import csv
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from attacks import build_attack
from attacks.temporal_translation import DEFAULT_MEAN, DEFAULT_STD, denormalize_video
from project_paths import PROJECT_ROOT

# 定义生成的明细CSV中每一行记录的基本字段（样本序号、唯一ID、视频ID、视角、真实标签、目标标签和对抗样本存储路径）
SAMPLE_BASE_FIELDS = ["index", "sample_id", "video_id", "camera", "true_label", "target_label", "adv_path",]

# 将输入的数值或带有除法的字符串（16/255）安全地解析为浮点数
def parse_float(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)

# 重定向运行时临时目录与matplotlib缓存目录到项目文件夹
def ensure_runtime_temp_dirs():
    tmp_dir = PROJECT_ROOT / "results" / "tmp"
    mpl_dir = tmp_dir / f"matplotlib-{os.getuid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

# 复制基础配置对象，并根据当前模型的特定配置更新多视角数量
def make_model_cfg(base_cfg, model_zoo, model_spec):
    cfg = SimpleNamespace(**base_cfg.__dict__)
    # DVANet/TBCNet 构建模型时会读取 cfg.num_views
    cfg.num_views = int(model_spec.get("num_views", model_zoo.get("num_views", getattr(base_cfg, "num_views", 3))))
    return cfg

# 根据模型名字和config里的checkpoint，把一个可用于攻击/评估的模型加载出来
# model_name   模型名，比如dvanet、tbcnet、i3d
# model_spec   当前模型在json里的配置，主要包含checkpoint路径
# base_cfg     数据集、类别数、视角数等基础配置
# model_zoo    整个config json读出来的内容
# device       cuda或cpu
def build_recognizer(model_name, model_spec, base_cfg, model_zoo, device):
    from models.build_model import build_model, load_model_checkpoint

    cfg = make_model_cfg(base_cfg, model_zoo, model_spec)
    model = build_model(model_name, cfg)
    load_model_checkpoint(
        model,
        model_spec["checkpoint"],
        model_name=model_name,
        use_ema=bool(model_spec.get("use_ema", False)),
        strict=bool(model_spec.get("strict", True)),
    )
    model.to(device)
    model.eval()
    return model

# 得到安全文件名字符串
def safe_number_token(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")

# 将归一化epsilon(0~1)换算为像素值尺度(0~255)，用于日志、元数据与uint8保存
def epsilon_scaled_value(args):
    return float(args.epsilon) * 255.0

# 生成人类可读的eps标签，如 0.5/255
def epsilon_label(args):
    return f"{safe_number_token(epsilon_scaled_value(args)).replace('p', '.')}/255"

# 为每个对抗样本生成唯一的.pt文件名
def safe_sample_name(index, sample_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("_")
    return f"{index:06d}_{safe or 'sample'}.pt"

# 将绝对路径转换为相对于项目根目录的相对路径，使日志和JSON记录更干净可读
def relpath(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)

# 将字典列表保存为CSV
def save_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# 全局汇总JSON
def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)

# 生成每个样本的汇总字段，动态拼接CSV的表头
def sample_summary_fields(eval_models):
    fields = list(SAMPLE_BASE_FIELDS)
    for model in eval_models:
        fields.extend(
            [
                f"clean_pred_{model}",
                f"adv_pred_{model}",
                f"clean_correct_{model}",
                f"adv_correct_{model}",
            ]
        )
    return fields

# 将多模型的黑盒评测结果与原始样本元数据进行关联对齐，组装成一张大宽表的行数据
def build_sample_summary_rows(metadata_rows, clean_predictions=None, adv_predictions=None, eval_models=None):
    clean_predictions = clean_predictions or []
    adv_predictions = adv_predictions or []
    eval_models = eval_models or []
    clean_by_model = {(row["model"], row["sample_id"]): row for row in clean_predictions}
    adv_by_model = {(row["model"], row["sample_id"]): row for row in adv_predictions}

    rows = []
    for meta in metadata_rows:
        sample_id = str(meta["sample_id"])
        row = {
            "index": int(meta["index"]),
            "sample_id": sample_id,
            "video_id": meta.get("video_id", ""),
            "camera": int(meta["camera"]),
            "true_label": int(meta["label"]),
            "target_label": optional_int(meta.get("target_label", -1)),
            "adv_path": meta["adv_path"],
        }
        for model in eval_models:
            clean_row = clean_by_model.get((model, sample_id), {})
            adv_row = adv_by_model.get((model, sample_id), {})
            row[f"clean_pred_{model}"] = clean_row.get("pred", "")
            row[f"adv_pred_{model}"] = adv_row.get("pred", "")
            row[f"clean_correct_{model}"] = clean_row.get("correct", "")
            row[f"adv_correct_{model}"] = adv_row.get("correct", "")
        rows.append(row)
    return rows

# 调用save_csv将组装好的总表落盘保存为sample_summary.csv
def save_sample_summary(run_dir, metadata_rows, clean_predictions=None, adv_predictions=None, eval_models=None):
    rows = build_sample_summary_rows(metadata_rows, clean_predictions, adv_predictions, eval_models)
    path = Path(run_dir) / "sample_summary.csv"
    save_csv(path, rows, sample_summary_fields(eval_models or []))
    return path

# 将张量从计算图分离并转移到CPU
def tensor_for_save(tensor, dtype_name):
    tensor = tensor.detach().cpu()
    if dtype_name == "float16":
        return tensor.half()
    return tensor.float()

# 为每一个对抗样本打包构造一个标准的.pt字典对象
def make_payload(video, label, meta, args, target_label=None):
    payload = {
        "video": tensor_for_save(video, args.save_dtype),
        "label": int(label),
        "sample_id": str(meta["sample_id"]),
        "meta": dict(meta),
        "normalization": {
            "mean": list(DEFAULT_MEAN),
            "std": list(DEFAULT_STD),
        },
        "layout": "T,C,H,W",
        "normalized": True,
    }
    if target_label is not None:
        payload["target_label"] = int(target_label)
        payload["target_action"] = int(target_label) + 1
        payload["target_strategy"] = args.target_strategy
    if args.save_video_01:
        video_01 = denormalize_video(video.unsqueeze(0), DEFAULT_MEAN, DEFAULT_STD).squeeze(0).clamp(0, 1)
        payload["video_01"] = tensor_for_save(video_01, args.save_dtype)
    return payload


# 当设置了--max_samples进行快速子集调试
def slice_batch(clips, labels, metas, remaining):
    if remaining <= 0 or clips.size(0) <= remaining:
        return clips, labels, metas
    return clips[:remaining], labels[:remaining], metas[:remaining]

# 解析和校验用户输入的待评测黑盒模型列表，如果没传该参数，则默认评测模型配置文件中的所有模型
def select_eval_models(model_zoo, eval_models):
    names = list(model_zoo["models"].keys())
    if not eval_models:
        return names
    requested = [name.strip() for name in eval_models.split(",") if name.strip()]
    missing = [name for name in requested if name not in model_zoo["models"]]
    if missing:
        raise ValueError(f"Eval models not found in config: {missing}")
    return requested

# 处理可选整型转换
def optional_int(value):
    if value is None or value == "":
        return None
    return int(value)

# 无定向返回原标签；定向攻击根据策略生成目标标签
def target_labels_for_batch(labels, num_actions, args, logits=None):
    labels = labels.long()
    if not args.targeted:
        return labels
    if num_actions < 2:
        raise ValueError(f"Targeted attack needs at least 2 actions, got {num_actions}")
    if args.target_strategy == "next":
        return (labels + 1) % num_actions
    if args.target_strategy in {"closest", "least_likely"}:
        if logits is None:
            raise ValueError(f"{args.target_strategy} target strategy requires source-model logits.")
        # 校验logits维度，要求shape是`[batch_size, 类别数]`，二维张量。维度不对抛异常
        if logits.dim() != 2:
            raise ValueError(f"Expected logits [B,num_classes], got {tuple(logits.shape)}")
        # 模型输出类别数≥2，道理同前面
        if logits.size(1) < 2:
            raise ValueError(f"{args.target_strategy} needs at least 2 classes, got {logits.size(1)}")
        # 标签不能小于 0，也不能大于等于类别总数，防止索引越界
        if labels.max().item() >= logits.size(1) or labels.min().item() < 0:
            raise ValueError(f"Labels are outside model output size {logits.size(1)}")
        # 复制 logits，`detach()`切断计算图
        masked_logits = logits.detach().clone()
        # 生成 batch 内的行索引：`[0,1,2,...,B-1]`，放到和 logits 相同的 GPU 上
        row_indices = torch.arange(labels.size(0), device=masked_logits.device)
        label_indices = labels.to(masked_logits.device)
        # 排除真实标签，保证定向目标不会退化为原始类别。
        if args.target_strategy == "closest":
            # clean-logit runner-up：源模型认为最接近真实类别的竞争类别。
            masked_logits[row_indices, label_indices] = float("-inf")
            return masked_logits.argmax(dim=1).to(labels.device)

        # least_likely：选择源模型最不可能的类别。
        masked_logits[row_indices, label_indices] = float("inf")
        return masked_logits.argmin(dim=1).to(labels.device)
    raise ValueError(f"Unsupported target_strategy: {args.target_strategy}")

# 把之前生成并保存的对抗样本重新包装成PyTorch数据集，用于二次评估
class AdvVideoDataset(Dataset):
    def __init__(self, metadata_rows):
        self.rows = list(metadata_rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = torch.load(PROJECT_ROOT / row["adv_path"], map_location="cpu")
        return payload["video"].float(), int(row["label"]), row

# 自定义collate_fn以处理不同形状的输入
def saved_collate(batch):
    videos, labels, rows = [], [], []
    for video, label, row in batch:
        videos.append(video)
        labels.append(label)
        rows.append(row)
    return torch.stack(videos, dim=0), torch.tensor(labels, dtype=torch.long), rows

# 统一不同模型的输出格式，最终都返回[B, 类别数]的logits
@torch.no_grad()
def logits_for_model(model, videos):
    outputs = model(videos.unsqueeze(1))
    if isinstance(outputs, dict):
        return outputs["per_view_logits"].mean(dim=1)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if outputs.dim() == 3:
        return outputs.mean(dim=1)
    return outputs

# 为一个模型生成批次预测明细行，返回(明细, 样本数, 正确数)
def prediction_rows_from_batch(model_name, labels, logits, batch_rows):
    preds = logits.argmax(dim=1)
    correct = preds.eq(labels)
    rows = []
    for i, meta_row in enumerate(batch_rows):
        sample_id = str(meta_row["sample_id"])
        target_label = optional_int(meta_row.get("target_label", ""))
        target_label = target_label if target_label is not None and target_label >= 0 else None
        target_hit = ""
        if target_label is not None:
            if target_label >= logits.size(1):
                raise ValueError(f"target_label {target_label} is outside model output size {logits.size(1)}")
            target_hit = int(preds[i].item() == target_label)
        rows.append(
            {
                "model": model_name,
                "sample_id": sample_id,
                "pred": int(preds[i].item()),
                "correct": int(correct[i].item()),
                "target_hit": target_hit,
            }
        )
    return rows, labels.size(0), correct.sum().item()

# 用保存的对抗样本评估单个模型，返回(正确数, 总数, 准确率, 逐样本明细)
def evaluate_adv_samples(model_name, model_spec, model_zoo, base_cfg, rows, device, args):
    dataset = AdvVideoDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        collate_fn=saved_collate,
    )
    model = build_recognizer(model_name, model_spec, base_cfg, model_zoo, device)
    if device.type == "cuda" and "," in args.gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    predictions = []
    total = 0
    correct_total = 0
    for videos, labels, batch_rows in tqdm(loader, desc=f"Evaluating adv/{model_name}"):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = logits_for_model(model, videos)
        batch_predictions, batch_total, batch_correct = prediction_rows_from_batch(
            model_name,
            labels,
            logits,
            batch_rows,
        )
        predictions.extend(batch_predictions)
        total += batch_total
        correct_total += batch_correct

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "split": "adv",
        "correct": int(correct_total),
        "total": int(total),
        "accuracy": correct_total / max(total, 1),
        "rows": predictions,
    }

# 评估干净样本
def evaluate_clean_split(model_name, model_spec, model_zoo, base_cfg, loader, device, args):
    model = build_recognizer(model_name, model_spec, base_cfg, model_zoo, device)
    if device.type == "cuda" and "," in args.gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    predictions = []
    total = 0
    correct_total = 0
    processed = 0
    for videos, labels, batch_rows in tqdm(loader, desc=f"Evaluating clean/{model_name}"):
        remaining = args.max_samples - processed if args.max_samples > 0 else 0
        videos, labels, batch_rows = slice_batch(videos, labels, batch_rows, remaining)
        if videos.size(0) == 0:
            break

        videos = videos.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        logits = logits_for_model(model, videos)
        batch_predictions, batch_total, batch_correct = prediction_rows_from_batch(
            model_name,
            labels,
            logits,
            batch_rows,
        )
        predictions.extend(batch_predictions)
        total += batch_total
        correct_total += batch_correct
        processed += batch_total

        if args.max_samples > 0 and processed >= args.max_samples:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "split": "clean",
        "correct": int(correct_total),
        "total": int(total),
        "accuracy": correct_total / max(total, 1),
        "rows": predictions,
    }

# 迁移总控
def evaluate_transfer(model_zoo, base_cfg, clean_loader, metadata_rows, eval_models, device, args):
    clean_predictions = []
    adv_predictions = []
    for model_name in eval_models:
        model_spec = model_zoo["models"][model_name]
        clean_result = evaluate_clean_split(model_name, model_spec, model_zoo, base_cfg, clean_loader, device, args)
        adv_result = evaluate_adv_samples(model_name, model_spec, model_zoo, base_cfg, metadata_rows, device, args)
        clean_predictions.extend(clean_result["rows"])
        adv_predictions.extend(adv_result["rows"])
        print(f"{model_name}: clean_acc={clean_result['accuracy']:.6f} adv_acc={adv_result['accuracy']:.6f}")

    summary_rows = summarize_transfer(clean_predictions, adv_predictions, eval_models, targeted=args.targeted)
    return clean_predictions, adv_predictions, summary_rows


def summarize_transfer(clean_rows, adv_rows, eval_models, targeted=False):
    clean_by_model = {model: {} for model in eval_models}
    adv_by_model = {model: {} for model in eval_models}

    for row in clean_rows:
        clean_by_model[row["model"]][row["sample_id"]] = row
    for row in adv_rows:
        adv_by_model[row["model"]][row["sample_id"]] = row

    all_sample_ids = set()
    for model in eval_models:
        all_sample_ids.update(clean_by_model[model].keys())
        all_sample_ids.update(adv_by_model[model].keys())

    common_clean_ids = [
        sid
        for sid in sorted(all_sample_ids)
        if all(
            sid in clean_by_model[model]
            and sid in adv_by_model[model]
            and int(clean_by_model[model][sid]["correct"]) == 1
            for model in eval_models
        )
    ]

    summary_rows = []
    for model in eval_models:
        clean_model_rows = clean_by_model[model]
        adv_model_rows = adv_by_model[model]
        sample_ids = sorted(set(clean_model_rows) & set(adv_model_rows))
        clean_correct = sum(int(clean_model_rows[sid]["correct"]) for sid in sample_ids)
        adv_correct = sum(int(adv_model_rows[sid]["correct"]) for sid in sample_ids)

        if targeted:
            success = sum(
                int(clean_model_rows[sid]["correct"]) == 1
                and optional_int(adv_model_rows[sid].get("target_hit", "")) == 1
                for sid in sample_ids
            )
            common_success = sum(
                sid in adv_model_rows and optional_int(adv_model_rows[sid].get("target_hit", "")) == 1
                for sid in common_clean_ids
            )
        else:
            success = sum(
                int(clean_model_rows[sid]["correct"]) == 1 and int(adv_model_rows[sid]["correct"]) == 0
                for sid in sample_ids
            )
            common_success = sum(
                sid in adv_model_rows and int(adv_model_rows[sid]["correct"]) == 0 for sid in common_clean_ids
            )

        summary_rows.append(
            {
                "model": model,
                "targeted": int(targeted),
                "total": len(sample_ids),
                "clean_correct": clean_correct,
                "clean_accuracy": clean_correct / max(len(sample_ids), 1),
                "adv_correct": adv_correct,
                "adv_accuracy": adv_correct / max(len(sample_ids), 1),
                "attack_success": success,
                "attack_success_rate": success / max(clean_correct, 1),
                "common_clean_total": len(common_clean_ids),
                "attack_success_common_clean": common_success,
                "attack_success_rate_common_clean": common_success / max(len(common_clean_ids), 1),
            }
        )
    return summary_rows

# 生成对抗样本：逐batch攻击、保存.pt、收集元信息，返回metadata_rows
def generate_examples(source_model, loader, device, run_dir, args, num_actions):
    # 在实验文件夹下创建`adv_samples`文件夹，**用来存放生成好的对抗样本pt文件**
    adv_dir = Path(run_dir) / "adv_samples"
    adv_dir.mkdir(parents=True, exist_ok=True)

    attacker = build_attack(
        args.attack_method,
        source_model,
        epsilon=args.epsilon,
        steps=args.steps,
        kernlen=args.kernlen,
        momentum=args.momentum,
        move_type=args.move_type,
        kernel_mode=args.kernel_mode,
        delay=args.delay,
        shift_batch_size=args.shift_batch_size,
        ti_kernlen=args.ti_kernlen,
        ti_nsig=args.ti_nsig,
        pgd_step_size=args.pgd_step_size,
        pgd_random_start=args.pgd_random_start,
        targeted=args.targeted,
        target_map_function=(lambda videos, labels: labels) if args.targeted else None,
    )

    metadata_rows = []
    processed = 0
    for clips, labels, metas in tqdm(loader, desc="Generating adversarial examples"):
        remaining = args.max_samples - processed if args.max_samples > 0 else 0
        clips, labels, metas = slice_batch(clips, labels, metas, remaining)
        if clips.size(0) == 0:
            break

        clean_videos = clips.float().to(device, non_blocking=True)
        labels_device = labels.long().to(device, non_blocking=True)
        # 确定攻击标签
        target_logits = None
        if args.targeted and args.target_strategy in {"closest", "least_likely"}:
            target_logits = logits_for_model(source_model, clean_videos)
        target_labels_device = target_labels_for_batch(labels_device, num_actions, args, target_logits)
        attack_labels_device = target_labels_device if args.targeted else labels_device
        adv_videos = attacker(clean_videos, attack_labels_device)
        saved_adv_videos = adv_videos

        for i, meta in enumerate(metas):
            index = processed + i
            filename = safe_sample_name(index, meta["sample_id"])
            adv_path = adv_dir / filename
            label = int(labels[i].item())
            target_label = int(target_labels_device[i].item()) if args.targeted else None
            torch.save(make_payload(saved_adv_videos[i], label, meta, args, target_label), adv_path)

            metadata_rows.append(
                {
                    "index": index,
                    "sample_id": str(meta["sample_id"]),
                    "video_id": str(meta.get("video_id", "")),
                    "camera": meta.get("camera", ""),
                    "label": label,
                    "target_label": -1 if target_label is None else target_label,
                    "adv_path": relpath(adv_path),
                }
            )

        processed += clips.size(0)
        if args.max_samples > 0 and processed >= args.max_samples:
            break

    return metadata_rows

# 打印信息
def print_run_header(args, model_zoo, base_cfg, dataset, device, run_dir, attack_name, num_actions):
    print("=" * 80)
    print(f"config: {args.config}")
    print(f"dataset: {model_zoo['dataset']}")
    print(f"training_mode: {model_zoo.get('training_mode', 'CS')}")
    print(f"annotations: {base_cfg.test_annotations}")
    print(f"loaded_samples: {len(dataset)}")
    print(f"source_model: {args.source_model}")
    print(f"attack_method: {args.attack_method}")
    print(f"targeted: {args.targeted}")
    if args.targeted:
        print(f"target_strategy: {args.target_strategy}")
        if args.target_strategy == "next":
            print(f"target_rule: A001->A002, ..., A{num_actions:03d}->A001")
        elif args.target_strategy == "closest":
            print("target_rule: source-model clean-logit runner-up, excluding the true label")
        elif args.target_strategy == "least_likely":
            print("target_rule: source-model least-likely class on clean input, excluding the true label")
    print(f"epsilon: {args.epsilon}")
    print(f"steps: {args.steps}")
    if attack_name.startswith("pgd"):
        pgd_step_size = args.pgd_step_size if args.pgd_step_size is not None else args.epsilon / args.steps
        print(f"pgd_step_size: {pgd_step_size}")
        print(f"pgd_random_start: {args.pgd_random_start}")
    print(f"device: {device}")
    print(f"output: {run_dir}")
    print("=" * 80)

# 把整个实验的所有信息打包成一个完整字典,summary.json
def build_metrics_summary(args, model_zoo, base_cfg, run_dir, metadata_rows, summary_rows, attack_name, num_actions):
    overall_metrics = {}
    for row in summary_rows:
        metrics = {
            "total": row["total"],
            "clean_correct": row["clean_correct"],
            "adv_correct": row["adv_correct"],
            "attack_success": row["attack_success"],
            "clean_accuracy": row["clean_accuracy"],
            "adv_accuracy": row["adv_accuracy"],
            "attack_success_rate": row["attack_success_rate"],
        }
        if args.targeted:
            metrics["target_success_rate"] = row["attack_success_rate"]
        else:
            metrics["asr"] = row["attack_success_rate"]
        overall_metrics[row["model"]] = metrics

    return {
        "experiment_config": {
            "config": str(Path(args.config).resolve()),
            "config_key": Path(args.config).stem.replace("model_zoo_", "", 1),
            "dataset": model_zoo["dataset"],
            "training_mode": model_zoo.get("training_mode", "CS"),
            "test_annotations": base_cfg.test_annotations,
            "num_actions": num_actions,
            "source_model": args.source_model,
            "attack_method": args.attack_method,
            "attack_key": attack_name,
            "targeted": args.targeted,
            "target_strategy": args.target_strategy if args.targeted else "",
            "epsilon": epsilon_label(args),
            "epsilon_value": args.epsilon,
            "steps": args.steps,
            "pgd_step_size": args.pgd_step_size if args.pgd_step_size is not None else args.epsilon / args.steps,
            "pgd_random_start": args.pgd_random_start,
            "kernlen": args.kernlen,
            "move_type": args.move_type,
            "kernel_mode": args.kernel_mode,
            "momentum": args.momentum,
            "delay": args.delay,
            "total_samples": len(metadata_rows),
            "save_dtype": args.save_dtype,
            "success_definition": (
                "clean_correct == 1 and adv_pred == target_label"
                if args.targeted
                else "clean_correct == 1 and adv_correct == 0"
            ),
        },
        "overall_metrics": overall_metrics,
        "outputs": {
            "run_dir": relpath(run_dir),
            "adv_samples": relpath(Path(run_dir) / "adv_samples"),
            "sample_summary": relpath(Path(run_dir) / "sample_summary.csv"),
            "metrics_summary": relpath(Path(run_dir) / "metrics_summary.json"),
        },
    }


def print_final_summary(sample_summary_path, metrics_summary_path, metadata_rows, summary_rows, targeted):
    print("=" * 80)
    print(f"generated_samples: {len(metadata_rows)}")
    print(f"sample_summary: {sample_summary_path}")
    if summary_rows:
        success_label = "target_success_rate" if targeted else "attack_success_rate"
        for row in summary_rows:
            print(
                f"{row['model']}: clean_acc={row['clean_accuracy']:.6f}, "
                f"adv_acc={row['adv_accuracy']:.6f}, "
                f"{success_label}={row['attack_success_rate']:.6f}"
            )
    print(f"metrics_summary: {metrics_summary_path}")
    print("=" * 80)

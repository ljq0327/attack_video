import argparse
import csv
import json
import os
import random
import sys


from project_paths import PROJECT_ROOT as PROJECT_ROOT_PATH, add_third_party_paths


PROJECT_ROOT = str(PROJECT_ROOT_PATH)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), f"matplotlib-{os.getuid()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

add_third_party_paths()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorchvideo.transforms import Normalize, ShortSideScale
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from configuration import build_config
from dataloader import frame_creation
from models.build_model import build_model, load_model_checkpoint


BASE_COLUMNS = [
    "sample_id",
    "row_index",
    "video_id",
    "subject",
    "action",
    "camera",
    "repetition",
    "setup",
    "start",
    "end",
    "segment_confidence",
    "label",
]

BASE_COLUMN_DEFAULTS = {
    "repetition": 0,
    "setup": 0,
    "start": "",
    "end": "",
    "segment_confidence": "",
}

MODEL_OUTPUT_COLUMNS = ["pred", "correct", "confidence", "true_confidence", "logit_margin"]
SUPPORTED_DATASETS = {"ntu_rgbd_120", "ntu_rgbd_60", "pkummd"}


def resolve_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_eval_transform():
    return Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCrop(224),
        ]
    )


def load_model_zoo(path):
    path = resolve_path(path)
    with open(path, "r") as f:
        model_zoo = json.load(f)

    models = {}
    for model_name, spec in model_zoo["models"].items():
        if isinstance(spec, str):
            spec = {"checkpoint": spec}
        spec = dict(spec)
        spec["checkpoint"] = resolve_path(spec["checkpoint"])
        if not os.path.exists(spec["checkpoint"]):
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {spec['checkpoint']}")
        models[model_name] = spec
    model_zoo["models"] = models
    model_zoo["_path"] = path
    return model_zoo


def build_cfg(model_zoo):
    cfg = build_config(
        model_zoo["dataset"],
        training_mode=model_zoo.get("training_mode", "CS"),
    )
    if "test_annotations" in model_zoo:
        cfg.test_annotations = model_zoo["test_annotations"]
    if "num_actions" in model_zoo:
        cfg.num_actions = int(model_zoo["num_actions"])
    if "num_views" in model_zoo:
        cfg.num_views = int(model_zoo["num_views"])
    cfg.test_annotations = resolve_path(cfg.test_annotations)
    cfg.train_annotations = resolve_path(cfg.train_annotations)
    return cfg


def int_or_default(value, default=0):
    if value is None or value == "":
        return default
    return int(value)


class FlatSampleDataset(Dataset):
    def __init__(self, cfg, annotations, transform=None, views=None):
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.videos_folder = cfg.videos_folder
        self.annotations = resolve_path(annotations)
        self.transform = transform
        self.views = set(views) if views else None
        self.height = 224
        self.width = 224
        self.num_frames = 16

        if self.dataset not in SUPPORTED_DATASETS:
            raise NotImplementedError(
                f"select_common_correct.py supports {sorted(SUPPORTED_DATASETS)}, got {self.dataset}."
            )

        with open(self.annotations, newline="") as f:
            reader = csv.DictReader(f)
            self._validate_columns(reader.fieldnames or [])
            self.rows = []
            for row_index, row in enumerate(reader):
                camera = int(row["camera"])
                if self.views is not None and camera not in self.views:
                    continue
                row = dict(row)
                self._normalize_row(row)
                row["row_index"] = row_index
                row["sample_id"] = self._sample_id(row)
                self.rows.append(row)

    def _validate_columns(self, fieldnames):
        required = {"video_id", "subject", "action", "camera"}
        if self.dataset == "pkummd":
            required.update({"start", "end", "confidence"})
        else:
            required.update({"repetition", "setup"})

        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Missing required columns in {self.annotations}: {missing}")

    def _normalize_row(self, row):
        if self.dataset == "pkummd":
            start = int(row["start"])
            end = int(row["end"])
            if start > end:
                start, end = end, start
            row["start"] = str(start)
            row["end"] = str(end)

    def _sample_id(self, row):
        if self.dataset == "pkummd":
            return "_".join(
                [
                    row["video_id"],
                    str(int(row["subject"])),
                    row["action"],
                    row["start"],
                    row["end"],
                ]
            )
        return row["video_id"]

    def _frame_row(self, row):
        if self.dataset == "pkummd":
            return [
                row["subject"],
                row["action"],
                row["video_id"],
                row["start"],
                row["end"],
            ]

        return [
            row["subject"],
            row["action"],
            row["video_id"],
            row["camera"],
            row["repetition"],
            row["setup"],
        ]

    def _meta(self, row):
        meta = {
            "sample_id": row["sample_id"],
            "row_index": int(row["row_index"]),
            "video_id": row["video_id"],
            "subject": int(row["subject"]),
            "action": int(row["action"]),
            "camera": int(row["camera"]),
            "repetition": int_or_default(row.get("repetition")),
            "setup": int_or_default(row.get("setup")),
            "start": row.get("start", ""),
            "end": row.get("end", ""),
            "segment_confidence": row.get("confidence", ""),
        }
        return meta

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = frame_creation(
            self._frame_row(row),
            self.dataset,
            self.videos_folder,
            self.height,
            self.width,
            self.num_frames,
            self.transform,
        )
        label = int(row["action"]) - 1
        return frames, label, self._meta(row)


def flat_collate(batch):
    frames = []
    labels = []
    metas = []
    for item in batch:
        clip, label, meta = item
        frames.append(clip)
        labels.append(label)
        metas.append(meta)
    return torch.stack(frames, dim=0), torch.tensor(labels, dtype=torch.long), metas


def build_loader(cfg, args):
    dataset = FlatSampleDataset(
        cfg,
        annotations=cfg.test_annotations,
        transform=build_eval_transform(),
        views=parse_int_list(args.views),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=flat_collate,
    )
    return loader, dataset


def maybe_limit_batches(loader, max_batches):
    if max_batches <= 0:
        yield from loader
        return
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        yield batch


def make_model_cfg(base_cfg, model_zoo, model_spec):
    cfg = type("", (), {})()
    cfg.__dict__.update(base_cfg.__dict__)
    cfg.num_views = int(model_spec.get("num_views", model_zoo.get("num_views", getattr(base_cfg, "num_views", 3))))
    return cfg


def build_recognizer(model_name, model_spec, base_cfg, model_zoo, device):
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


def prediction_path(prediction_dir, model_name):
    return os.path.join(prediction_dir, f"{model_name}.csv")


def write_prediction_header(writer):
    writer.writeheader()


@torch.no_grad()
def predict_one_model(model_name, model_spec, loader, base_cfg, model_zoo, device, output_csv, args):
    model = build_recognizer(model_name, model_spec, base_cfg, model_zoo, device)
    if device.type == "cuda" and "," in args.gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    total_seen = 0
    total_correct = 0
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = BASE_COLUMNS + ["pred", "correct", "confidence", "true_confidence", "logit_margin"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        write_prediction_header(writer)

        progress = tqdm(maybe_limit_batches(loader, args.max_batches), desc=f"Predicting {model_name}")
        for clips, labels, metas in progress:
            labels = labels.long()
            clips = clips.float().unsqueeze(1).to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            outputs = model(clips)
            logits = outputs["per_view_logits"].mean(dim=1)
            probs = F.softmax(logits, dim=1)
            confidence, preds = probs.max(dim=1)
            true_confidence = probs.gather(1, labels_device.unsqueeze(1)).squeeze(1)

            other_logits = logits.clone()
            row_ids = torch.arange(labels_device.size(0), device=device)
            other_logits[row_ids, labels_device] = -float("inf")
            best_other_logits = other_logits.max(dim=1).values
            true_logits = logits.gather(1, labels_device.unsqueeze(1)).squeeze(1)
            logit_margin = true_logits - best_other_logits

            correct = preds.eq(labels_device)
            total_seen += labels_device.size(0)
            total_correct += correct.sum().item()
            progress.set_postfix(acc=f"{total_correct / max(total_seen, 1):.4f}")

            preds_cpu = preds.detach().cpu()
            correct_cpu = correct.detach().cpu()
            confidence_cpu = confidence.detach().cpu()
            true_confidence_cpu = true_confidence.detach().cpu()
            logit_margin_cpu = logit_margin.detach().cpu()

            for i, meta in enumerate(metas):
                row = dict(meta)
                row["label"] = int(labels[i].item())
                row["pred"] = int(preds_cpu[i].item())
                row["correct"] = int(correct_cpu[i].item())
                row["confidence"] = float(confidence_cpu[i].item())
                row["true_confidence"] = float(true_confidence_cpu[i].item())
                row["logit_margin"] = float(logit_margin_cpu[i].item())
                writer.writerow(row)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "prediction_csv": output_csv,
        "total": total_seen,
        "correct": total_correct,
        "acc": total_correct / max(total_seen, 1),
    }


def load_prediction_csv(model_name, path):
    df = pd.read_csv(path)
    for column in BASE_COLUMNS:
        if column not in df.columns:
            df[column] = BASE_COLUMN_DEFAULTS.get(column, "")

    missing_output_columns = [column for column in MODEL_OUTPUT_COLUMNS if column not in df.columns]
    if missing_output_columns:
        raise ValueError(f"Missing prediction columns in {path}: {missing_output_columns}")

    keep_columns = BASE_COLUMNS + MODEL_OUTPUT_COLUMNS
    df = df[keep_columns].copy()
    rename_map = {
        "pred": f"{model_name}_pred",
        "correct": f"{model_name}_correct",
        "confidence": f"{model_name}_confidence",
        "true_confidence": f"{model_name}_true_confidence",
        "logit_margin": f"{model_name}_logit_margin",
    }
    return df.rename(columns=rename_map)


def merge_predictions(model_names, prediction_dir):
    merged = None
    for model_name in model_names:
        path = prediction_path(prediction_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prediction CSV for {model_name}: {path}")

        df = load_prediction_csv(model_name, path)
        model_columns = [col for col in df.columns if col not in BASE_COLUMNS]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["sample_id"] + model_columns], on="sample_id", how="inner")

    correct_columns = [f"{model_name}_correct" for model_name in model_names]
    confidence_columns = [f"{model_name}_confidence" for model_name in model_names]
    margin_columns = [f"{model_name}_logit_margin" for model_name in model_names]

    for column in correct_columns:
        merged[column] = merged[column].astype(int)

    merged["all_correct"] = merged[correct_columns].eq(1).all(axis=1)
    merged["min_confidence"] = merged[confidence_columns].min(axis=1)
    merged["mean_confidence"] = merged[confidence_columns].mean(axis=1)
    merged["min_logit_margin"] = merged[margin_columns].min(axis=1)
    merged["mean_logit_margin"] = merged[margin_columns].mean(axis=1)
    return merged


def group_records(num_actions, views, selection_group):
    if selection_group == "action_camera":
        for action in range(1, num_actions + 1):
            for camera in views:
                yield {"action": action, "camera": camera}
        return

    if selection_group == "action":
        for action in range(1, num_actions + 1):
            yield {"action": action}
        return

    raise ValueError(f"Unsupported selection_group: {selection_group}")


def group_mask(df, group_key):
    mask = df["action"] == group_key["action"]
    if "camera" in group_key:
        mask &= df["camera"] == group_key["camera"]
    return mask


def group_random_state(seed, group_key):
    camera = group_key.get("camera", 0)
    return seed + group_key["action"] * 10 + camera


def group_sort_columns(selection_group):
    if selection_group == "action_camera":
        return ["action", "camera", "sample_id"]
    if selection_group == "action":
        return ["action", "sample_id"]
    raise ValueError(f"Unsupported selection_group: {selection_group}")


def expected_selection_count(num_actions, views, selection_group, target_per_group):
    group_count = num_actions * len(views) if selection_group == "action_camera" else num_actions
    return group_count * target_per_group


def build_group_summary(merged, selected, num_actions, views, selection_group):
    records = []
    for group_key in group_records(num_actions, views, selection_group):
        group = merged[group_mask(merged, group_key)]
        selected_group = selected[group_mask(selected, group_key)]
        records.append(
            {
                **group_key,
                "total_candidates": int(len(group)),
                "all_correct_candidates": int(group["all_correct"].sum()) if len(group) else 0,
                "selected": int(len(selected_group)),
            }
        )
    return pd.DataFrame(records)


def select_samples(common, num_actions, views, selection_group, strategy, target_per_group, seed):
    selected_groups = []
    missing_records = []

    for group_key in group_records(num_actions, views, selection_group):
        group = common[group_mask(common, group_key)]
        if len(group) < target_per_group:
            missing_records.append(
                {
                    **group_key,
                    "available": int(len(group)),
                    "required": int(target_per_group),
                }
            )
            if len(group) == 0:
                continue

        if strategy == "random":
            take = group.sample(
                n=min(target_per_group, len(group)),
                random_state=group_random_state(seed, group_key),
            )
        elif strategy == "max_min_confidence":
            take = group.sort_values(
                ["min_confidence", "mean_confidence", "sample_id"],
                ascending=[False, False, True],
            ).head(target_per_group)
        elif strategy == "max_min_margin":
            take = group.sort_values(
                ["min_logit_margin", "mean_logit_margin", "sample_id"],
                ascending=[False, False, True],
            ).head(target_per_group)
        else:
            raise ValueError(f"Unsupported select strategy: {strategy}")

        selected_groups.append(take)

    if selected_groups:
        selected = pd.concat(selected_groups, ignore_index=True)
        selected = selected.sort_values(group_sort_columns(selection_group)).reset_index(drop=True)
    else:
        selected = common.iloc[0:0].copy()

    missing_columns = list(group_sort_columns(selection_group))
    missing_columns.remove("sample_id")
    missing_columns += ["available", "required"]
    return selected, pd.DataFrame(missing_records, columns=missing_columns)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser("Select samples correctly classified by all configured models.")
    parser.add_argument("--config", default="configs/model_zoo_ntu120_cset.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--views", default=None)
    parser.add_argument("--prediction_dir", default="results/predictions/ntu120_cset")
    parser.add_argument("--output_dir", default="results/selected_samples/ntu120_cset")
    parser.add_argument("--select_strategy", default="random", choices=["random", "max_min_confidence", "max_min_margin"])
    parser.add_argument("--selection_group", default=None, choices=["action_camera", "action"])
    parser.add_argument("--target_per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0, help="Use >0 for a quick smoke test.")
    parser.add_argument("--models", default=None, help="Optional comma-separated model subset.")
    parser.add_argument("--reuse_predictions", action="store_true")
    parser.add_argument("--only_merge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    model_zoo = load_model_zoo(args.config)
    base_cfg = build_cfg(model_zoo)
    views = parse_int_list(args.views) or parse_int_list(model_zoo.get("views")) or [1, 2, 3]
    args.views = views
    selection_group = args.selection_group or model_zoo.get("selection_group", "action_camera")
    target_per_group = args.target_per_group or int(model_zoo.get("target_per_group", 1))
    model_names = list(model_zoo["models"].keys())
    if args.models:
        requested = [name.strip() for name in args.models.split(",") if name.strip()]
        missing = [name for name in requested if name not in model_zoo["models"]]
        if missing:
            raise ValueError(f"Models not found in config: {missing}")
        model_names = requested

    os.makedirs(args.prediction_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_stats = []

    if not args.only_merge:
        loader, dataset = build_loader(base_cfg, args)
        print("=" * 60)
        print(f"config: {resolve_path(args.config)}")
        print(f"dataset: {model_zoo['dataset']}")
        print(f"training_mode: {model_zoo.get('training_mode', 'CS')}")
        print(f"test_annotations: {base_cfg.test_annotations}")
        print(f"samples: {len(dataset)}")
        print(f"models: {', '.join(model_names)}")
        print(f"selection_group: {selection_group}")
        print(f"target_per_group: {target_per_group}")
        print(f"device: {device}")
        print("=" * 60)

        for model_name in model_names:
            output_csv = prediction_path(args.prediction_dir, model_name)
            if args.reuse_predictions and os.path.exists(output_csv):
                print(f"Reuse prediction CSV for {model_name}: {output_csv}")
                continue

            stat = predict_one_model(
                model_name,
                model_zoo["models"][model_name],
                loader,
                base_cfg,
                model_zoo,
                device,
                output_csv,
                args,
            )
            run_stats.append(stat)
            print(f"{model_name}: acc={stat['acc']:.4f} ({stat['correct']}/{stat['total']})")

    merged = merge_predictions(model_names, args.prediction_dir)
    common = merged[merged["all_correct"]].copy()
    selected, missing_df = select_samples(
        common,
        num_actions=int(model_zoo.get("num_actions", base_cfg.num_actions)),
        views=views,
        selection_group=selection_group,
        strategy=args.select_strategy,
        target_per_group=target_per_group,
        seed=args.seed,
    )
    summary_df = build_group_summary(
        merged,
        selected,
        num_actions=int(model_zoo.get("num_actions", base_cfg.num_actions)),
        views=views,
        selection_group=selection_group,
    )

    merged_path = os.path.join(args.output_dir, "merged_predictions.csv")
    common_path = os.path.join(args.output_dir, "common_correct_all.csv")
    selected_path = os.path.join(args.output_dir, f"selected_{len(selected)}.csv")
    expected_selected = expected_selection_count(
        int(model_zoo.get("num_actions", base_cfg.num_actions)),
        views,
        selection_group,
        target_per_group,
    )
    selected_fixed_path = os.path.join(args.output_dir, f"selected_{expected_selected}.csv")
    missing_name = "missing_action_camera.csv" if selection_group == "action_camera" else "missing_action.csv"
    summary_name = "summary_by_action_camera.csv" if selection_group == "action_camera" else "summary_by_action.csv"
    missing_path = os.path.join(args.output_dir, missing_name)
    summary_path = os.path.join(args.output_dir, summary_name)
    run_summary_path = os.path.join(args.output_dir, "summary.json")

    merged.to_csv(merged_path, index=False)
    common.to_csv(common_path, index=False)
    selected.to_csv(selected_path, index=False)
    has_full_selection = len(selected) == expected_selected
    if has_full_selection and selected_fixed_path != selected_path:
        selected.to_csv(selected_fixed_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    run_summary = {
        "config": resolve_path(args.config),
        "dataset": model_zoo["dataset"],
        "training_mode": model_zoo.get("training_mode", "CS"),
        "test_annotations": base_cfg.test_annotations,
        "models": model_names,
        "views": views,
        "selection_group": selection_group,
        "select_strategy": args.select_strategy,
        "target_per_group": target_per_group,
        "total_samples": int(len(merged)),
        "common_correct_samples": int(len(common)),
        "selected_samples": int(len(selected)),
        "missing_groups": int(len(missing_df)),
        "run_stats": run_stats,
        "outputs": {
            "merged_predictions": merged_path,
            "common_correct_all": common_path,
            "selected": selected_path,
            "selected_full": selected_fixed_path if has_full_selection else None,
            "selected_360": selected_fixed_path if has_full_selection and expected_selected == 360 else None,
            "missing_groups": missing_path,
            "missing_action_camera": missing_path if selection_group == "action_camera" else None,
            "summary_by_group": summary_path,
            "summary_by_action_camera": summary_path if selection_group == "action_camera" else None,
        },
    }
    save_json(run_summary_path, run_summary)

    print("=" * 60)
    print(f"merged_samples: {len(merged)}")
    print(f"common_correct_samples: {len(common)}")
    print(f"selected_samples: {len(selected)}")
    print(f"missing_groups: {len(missing_df)}")
    print(f"selected_csv: {selected_path}")
    if has_full_selection:
        print(f"selected_{expected_selected}_csv: {selected_fixed_path}")
    print(f"summary_json: {run_summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

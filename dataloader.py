import csv
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from project_paths import PROJECT_ROOT, add_third_party_paths

add_third_party_paths()
# 定义系统白名单支持的数据集集合
SUPPORTED_DATASETS = {"ntu_rgbd_60", "ntu_rgbd_120", "pkummd"}
# 定义了各个数据集在服务器磁盘上的默认HDF5视频存储根目录路径
DEFAULT_VIDEO_FOLDERS = {
    "ntu_rgbd_60": "/mnt/data/ljq/datasets/16rgb_hdf5/",
    "ntu_rgbd_120": "/mnt/data/ljq/datasets/16rgb_hdf5/",
    "pkummd": "/mnt/data/ljq/datasets/pku-mmdhdf5/",
}
# 定义了CSV标注文件中可能缺失的可选字段的默认填充值
BASE_COLUMN_DEFAULTS = {
    "repetition": 0,
    "setup": 0,
    "start": "",
    "end": "",
    "confidence": "",
    "segment_confidence": "",
}
# 路径解析器
def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)
# 固定全局随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# 将输入的整数、列表或以逗号分隔的字符串安全地解析并转换为整型列表
def parse_int_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]

#构建标准的视频评估预处理流水线
def build_eval_transform():
    from pytorchvideo.transforms import Normalize, ShortSideScale
    from torchvision.transforms import CenterCrop, Compose

    return Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCrop(224),
        ]
    )


def load_model_zoo(path):
    # 调用前面定义的 resolve_path 函数，将传入的配置文件路径转换为绝对路径
    config_path = resolve_path(path)
    with open(config_path, "r") as f:
        # 从打开的文件对象`f`读取全部文本
        model_zoo = json.load(f)
    # 获取数据集字段
    dataset = model_zoo.get("dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise NotImplementedError(f"Unsupported dataset in {config_path}: {dataset}")
    # 检查是否漏掉了测试机标注文件
    if "test_annotations" not in model_zoo:
        raise ValueError(f"test_annotations must be set in {config_path}")
    model_zoo["test_annotations"] = resolve_path(model_zoo["test_annotations"])
    # 获取视频存储根目录
    videos_folder = model_zoo.get("videos_folder", DEFAULT_VIDEO_FOLDERS.get(dataset))
    if not videos_folder:
        raise ValueError(f"videos_folder must be set in {config_path}")
    # 将最终确认的视频文件夹路径转换为绝对路径 
    model_zoo["videos_folder"] = resolve_path(videos_folder)
    # 初始化一个空字典，用来存放后面处理完的、干净规范的模型配置详情
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
    model_zoo["_path"] = config_path
    return model_zoo


def build_cfg(model_zoo):
    required = ["dataset", "videos_folder", "test_annotations", "num_actions"]
    missing = [key for key in required if key not in model_zoo]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    cfg = SimpleNamespace(
        dataset=model_zoo["dataset"],
        training_mode=model_zoo.get("training_mode", "CS"),
        videos_folder=model_zoo["videos_folder"],
        test_annotations=model_zoo["test_annotations"],
        num_actions=int(model_zoo["num_actions"]),
        num_views=int(model_zoo.get("num_views", 3)),
        views=parse_int_list(model_zoo.get("views")),
    )

    for key in ("selection_group", "target_per_group"):
        if key in model_zoo:
            setattr(cfg, key, model_zoo[key])

    return cfg

def int_or_default(value, default=0):
    if value is None or value == "":
        return default
    return int(value)


class AttackSampleDataset(Dataset):
    def __init__(self, cfg, annotations=None, transform=None, views=None):
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.videos_folder = cfg.videos_folder
        self.annotations = resolve_path(annotations or cfg.test_annotations)
        self.transform = transform
        self.views = set(parse_int_list(views) or [])
        self.height = 224
        self.width = 224
        self.num_frames = 16

        if self.dataset not in SUPPORTED_DATASETS:
            raise NotImplementedError(f"dataloader.py supports {sorted(SUPPORTED_DATASETS)}, got {self.dataset}.")

        with open(self.annotations, newline="") as f:
            reader = csv.DictReader(f)
            # 调用校验函数检查CSV表头字段是否完整
            self._validate_columns(reader.fieldnames or [])
            self.rows = []
            for row_index, row in enumerate(reader):
                row = self._normalize_row(row)
                camera = self._camera_value(row)
                if self.views and camera not in self.views:
                    continue
                row["row_index"] = row_index
                row["sample_id"] = self._sample_id(row)
                self.rows.append(row)

    def _validate_columns(self, fieldnames):
        columns = set(fieldnames)
        required = {"subject", "action"}

        if self.dataset == "pkummd":
            required.update({"video_id", "camera", "start", "end", "confidence"})
        elif self.dataset in {"ntu_rgbd_60", "ntu_rgbd_120"}:
            required.update({"video_id", "camera", "repetition", "setup"})
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.annotations}: {missing}")

    def _normalize_row(self, row):
        normalized = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }
        # 确保起始帧小于等于结束帧，顺序颠倒时自动纠正
        if self.dataset == "pkummd":
            start = int(normalized["start"])
            end = int(normalized["end"])
            if start > end:
                start, end = end, start
            normalized["start"] = str(start)
            normalized["end"] = str(end)

        return normalized

    def _camera_value(self, row):
        return int(row["camera"])

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
            return [row["subject"], row["action"], row["video_id"], row["start"], row["end"],]

        return [
            row["subject"], row["action"], row["video_id"], row["camera"], row["repetition"], row["setup"],]

    def _label(self, row):
        return int(row["action"]) - 1
    # 打包当前样本的所有元数据到一个标准的字典中（供后续记录和 CSV 归档使用）
    def _meta(self, row):
        return {
            "sample_id": row["sample_id"],
            "row_index": int(row["row_index"]),
            "video_id": row["video_id"],
            "subject": int(row["subject"]),
            "action": int(row["action"]),
            "camera": self._camera_value(row),
            "repetition": int_or_default(row.get("repetition", BASE_COLUMN_DEFAULTS["repetition"])),
            "setup": int_or_default(row.get("setup", BASE_COLUMN_DEFAULTS["setup"])),
            "start": row.get("start", BASE_COLUMN_DEFAULTS["start"]),
            "end": row.get("end", BASE_COLUMN_DEFAULTS["end"]),
            "segment_confidence": row.get(
                "segment_confidence",
                row.get("confidence", BASE_COLUMN_DEFAULTS["segment_confidence"]),
            ),
        }
    # 返回总样本数量
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
        return frames, self._label(row), self._meta(row)


def attack_collate(batch):
    clips = []
    labels = []
    metas = []

    for clip, label, meta in batch:
        clips.append(clip)
        labels.append(label)
        metas.append(meta)

    return torch.stack(clips, dim=0), torch.tensor(labels, dtype=torch.long), metas

flat_collate = attack_collate


def build_loader(cfg, args):
    cli_views = parse_int_list(getattr(args, "views", None))
    cfg_views = parse_int_list(getattr(cfg, "views", None))
    views = cli_views if cli_views is not None else cfg_views

    dataset = AttackSampleDataset(
        cfg,
        annotations=cfg.test_annotations,
        transform=build_eval_transform(),
        views=views,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=attack_collate,
    )
    return loader, dataset

build_attack_loader = build_loader


def _first_existing_path(candidates):
    paths = [Path(candidate) for candidate in candidates]
    for path in paths:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"HDF5 file not found. Tried: {tried}")


def _read_hdf5_frames(file_path, num_frames):
    with h5py.File(file_path, "r") as h5_file:
        frames_dataset = h5_file["default"]
        total_frames = frames_dataset.shape[0]
        if total_frames <= 0:
            raise ValueError(f"No frames found in {file_path}")

        frame_indexer = np.linspace(0, total_frames - 1, num_frames).astype(int)
        return [frames_dataset[idx] for idx in frame_indexer]


def _prepare_frames(frames, transform):
    frames = torch.from_numpy(np.stack(frames)).float().div(255.0)
    if transform:
        frames = frames.transpose(0, 1)
        frames = transform(frames)
        frames = frames.transpose(0, 1)
    return frames


def _ntu_hdf5_path(videos_folder, video_id):
    video_id = str(video_id)
    names = []
    if video_id.endswith(".hdf5"):
        names.append(video_id)
    names.append(f"{video_id}.hdf5")
    if video_id.endswith(".avi"):
        names.append(f"{video_id[:-4]}.hdf5")
    else:
        names.append(f"{video_id}.avi.hdf5")
    return _first_existing_path(Path(videos_folder) / name for name in names)


def _pkummd_hdf5_path(videos_folder, subject, action, video_id, start_frame, end_frame):
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    file_name = f"{video_id}_{int(subject)}_{action}_{start_frame}_{end_frame}.hdf5"
    return _first_existing_path([Path(videos_folder) / file_name])


def frame_creation(row, dataset, videos_folder, height, width, num_frames, transform):
    if dataset in {"ntu_rgbd_60", "ntu_rgbd_120"}:
        subject, action, video_id = row[0], row[1], row[2]
        file_path = _ntu_hdf5_path(videos_folder, video_id)
    elif dataset == "pkummd":
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], row[3], row[4]
        file_path = _pkummd_hdf5_path(videos_folder, subject, action, video_id, start_frame, end_frame)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    frames = _read_hdf5_frames(file_path, num_frames)
    return _prepare_frames(frames, transform)

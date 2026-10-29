#!/usr/bin/env python3
"""Visualize clean/adversarial video tensors saved by attack_selected.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw


DEFAULT_MEAN = (0.45, 0.45, 0.45)
DEFAULT_STD = (0.225, 0.225, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean/adv/difference contact sheet from two saved .pt videos."
    )
    parser.add_argument("--clean", type=Path, required=True, help="Path to clean .pt file.")
    parser.add_argument("--adv", type=Path, required=True, help="Path to adversarial .pt file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <run_dir>/visualizations/<sample>_clean_vs_adv.png.",
    )
    parser.add_argument(
        "--diff_scale",
        type=float,
        default=20.0,
        help="Multiplier for the absolute pixel difference row.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Maximum number of frames to show. Uses evenly-spaced frames if the video is longer.",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=224,
        help="Width of each displayed frame in pixels.",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save per-frame clean/adv/diff triplets next to the contact sheet.",
    )
    parser.add_argument(
        "--frame_dir",
        type=Path,
        default=None,
        help="Directory for --save_frames output. Defaults to <output_stem>_frames.",
    )
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def default_output_path(clean_path: Path) -> Path:
    run_dir = clean_path.parent.parent if clean_path.parent.name == "clean" else clean_path.parent
    return run_dir / "visualizations" / f"{clean_path.stem}_clean_vs_adv.png"


def ensure_tchw(video: torch.Tensor, layout: str | None) -> torch.Tensor:
    video = video.detach().cpu().float()

    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 5D video, got shape {tuple(video.shape)}")
        video = video[0]

    if video.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got shape {tuple(video.shape)}")

    normalized_layout = (layout or "").replace(" ", "").replace("_", "").upper()
    if normalized_layout in {"TCHW", "T,C,H,W"}:
        pass
    elif normalized_layout in {"CTHW", "C,T,H,W"}:
        video = video.permute(1, 0, 2, 3)
    elif normalized_layout in {"THWC", "T,H,W,C"}:
        video = video.permute(0, 3, 1, 2)
    elif video.shape[1] in (1, 3):
        pass
    elif video.shape[0] in (1, 3):
        video = video.permute(1, 0, 2, 3)
    elif video.shape[-1] in (1, 3):
        video = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Could not infer video layout for shape {tuple(video.shape)}")

    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1)
    elif video.shape[1] > 3:
        video = video[:, :3]

    return video.contiguous()


def denormalize(video: torch.Tensor, payload: Dict[str, Any]) -> torch.Tensor:
    normalization = payload.get("normalization") or {}
    mean = torch.tensor(normalization.get("mean", DEFAULT_MEAN), dtype=video.dtype)
    std = torch.tensor(normalization.get("std", DEFAULT_STD), dtype=video.dtype)

    if mean.numel() != video.shape[1] or std.numel() != video.shape[1]:
        raise ValueError(
            f"Normalization channels do not match video channels: "
            f"video={video.shape[1]}, mean={mean.numel()}, std={std.numel()}"
        )

    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return video * std + mean


def load_video_01(path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = torch_load(path)
    if isinstance(payload, torch.Tensor):
        payload = {"video": payload, "layout": None, "normalized": False}
    if not isinstance(payload, dict):
        raise TypeError(f"{path} should contain a tensor or dict payload, got {type(payload)}")

    if "video_01" in payload:
        video = payload["video_01"]
        video = ensure_tchw(video, payload.get("layout"))
    elif "video" in payload:
        video = ensure_tchw(payload["video"], payload.get("layout"))
        if payload.get("normalized", True):
            video = denormalize(video, payload)
    else:
        raise KeyError(f"{path} does not contain 'video' or 'video_01'")

    if float(video.max()) > 2.0:
        video = video / 255.0
    return video.clamp(0.0, 1.0), payload


def choose_frame_indices(num_frames: int, max_frames: int) -> List[int]:
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    indices = torch.linspace(0, num_frames - 1, steps=max_frames).round().long().tolist()
    deduped: List[int] = []
    for idx in indices:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def tensor_to_image(frame: torch.Tensor, tile_width: int) -> Image.Image:
    frame = frame.clamp(0.0, 1.0)
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    if tile_width > 0 and image.width != tile_width:
        tile_height = round(image.height * tile_width / image.width)
        image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    return image


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    draw.text(xy, text, fill=fill)


def make_contact_sheet(
    clean: torch.Tensor,
    adv: torch.Tensor,
    indices: Sequence[int],
    diff_scale: float,
    tile_width: int,
) -> Image.Image:
    clean_images = [tensor_to_image(clean[i], tile_width) for i in indices]
    adv_images = [tensor_to_image(adv[i], tile_width) for i in indices]
    diff_images = [tensor_to_image((adv[i] - clean[i]).abs() * diff_scale, tile_width) for i in indices]

    tile_w, tile_h = clean_images[0].size
    label_w = 108
    top_h = 28
    pad = 10
    col_gap = 4
    row_gap = 12
    rows = [("clean", clean_images), ("adv", adv_images), (f"abs diff x{diff_scale:g}", diff_images)]

    width = label_w + pad * 2 + len(indices) * tile_w + (len(indices) - 1) * col_gap
    height = top_h + pad * 2 + len(rows) * tile_h + (len(rows) - 1) * row_gap
    sheet = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)

    for col, frame_idx in enumerate(indices):
        x = label_w + pad + col * (tile_w + col_gap)
        draw_text(draw, (x + 4, 6), f"{frame_idx:02d}", (45, 45, 45))

    for row_idx, (label, images) in enumerate(rows):
        y = top_h + pad + row_idx * (tile_h + row_gap)
        draw_text(draw, (pad, y + max(0, tile_h // 2 - 8)), label, (25, 25, 25))
        for col, image in enumerate(images):
            x = label_w + pad + col * (tile_w + col_gap)
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))

    return sheet


def save_frame_triplets(
    frame_dir: Path,
    clean: torch.Tensor,
    adv: torch.Tensor,
    indices: Iterable[int],
    diff_scale: float,
    tile_width: int,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        clean_img = tensor_to_image(clean[idx], tile_width)
        adv_img = tensor_to_image(adv[idx], tile_width)
        diff_img = tensor_to_image((adv[idx] - clean[idx]).abs() * diff_scale, tile_width)
        w, h = clean_img.size
        canvas = Image.new("RGB", (w * 3 + 16, h + 24), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        for col, (label, image) in enumerate((("clean", clean_img), ("adv", adv_img), ("diff", diff_img))):
            x = col * (w + 8)
            draw_text(draw, (x + 4, 4), label, (25, 25, 25))
            canvas.paste(image, (x, 24))
        canvas.save(frame_dir / f"frame_{idx:02d}.png")


def main() -> None:
    args = parse_args()
    clean, clean_payload = load_video_01(args.clean)
    adv, adv_payload = load_video_01(args.adv)

    num_frames = min(clean.shape[0], adv.shape[0])
    if clean.shape[1:] != adv.shape[1:]:
        raise ValueError(f"Shape mismatch after loading: clean={tuple(clean.shape)}, adv={tuple(adv.shape)}")
    if clean.shape[0] != adv.shape[0]:
        print(f"Warning: frame count mismatch, comparing first {num_frames} frames.")
        clean = clean[:num_frames]
        adv = adv[:num_frames]

    indices = choose_frame_indices(num_frames, args.max_frames)
    output = args.output or default_output_path(args.clean)
    output.parent.mkdir(parents=True, exist_ok=True)

    sheet = make_contact_sheet(clean, adv, indices, args.diff_scale, args.tile_width)
    sheet.save(output)

    if args.save_frames:
        frame_dir = args.frame_dir or output.with_suffix("").parent / f"{output.stem}_frames"
        save_frame_triplets(frame_dir, clean, adv, indices, args.diff_scale, args.tile_width)
        print(f"Saved frame triplets: {frame_dir}")

    delta = (adv - clean).abs()
    per_frame = delta.flatten(1).mean(dim=1)
    shown = ", ".join(f"{idx}:{float(per_frame[idx]):.6f}" for idx in indices)
    clean_id = clean_payload.get("sample_id", args.clean.stem)
    adv_id = adv_payload.get("sample_id", args.adv.stem)

    print(f"Saved comparison: {output}")
    print(f"Sample: clean={clean_id} adv={adv_id}")
    print(f"Shape: clean={tuple(clean.shape)} adv={tuple(adv.shape)}")
    print(f"Mean abs pixel diff: {float(delta.mean()):.6f}")
    print(f"Max abs pixel diff: {float(delta.max()):.6f}")
    print(f"Shown frame mean abs diff: {shown}")


if __name__ == "__main__":
    main()

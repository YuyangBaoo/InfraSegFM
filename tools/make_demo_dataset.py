"""Create a tiny synthetic InfraSegFM dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def draw_line(mask: np.ndarray, start, end, thickness: int = 2) -> None:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(y1 - y0), abs(x1 - x0), 1)
    for t in np.linspace(0.0, 1.0, steps + 1):
        y = int(round(y0 * (1 - t) + y1 * t))
        x = int(round(x0 * (1 - t) + x1 * t))
        y1b, y2b = max(0, y - thickness), min(mask.shape[0], y + thickness + 1)
        x1b, x2b = max(0, x - thickness), min(mask.shape[1], x + thickness + 1)
        mask[y1b:y2b, x1b:x2b] = 1


def make_sample(kind: str, image_size: int, seed: int):
    rng = np.random.default_rng(seed)
    h = w = image_size
    image = rng.normal(150, 18, size=(h, w, 3)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    if kind == "crack":
        x0 = int(rng.integers(w // 6, w // 3))
        x1 = int(rng.integers(2 * w // 3, 5 * w // 6))
        y0 = int(rng.integers(h // 5, h // 2))
        y1 = int(rng.integers(h // 2, 4 * h // 5))
        draw_line(mask, (y0, x0), (y1, x1), thickness=max(1, image_size // 64))
        image[mask > 0] = np.array([45, 45, 45], dtype=np.uint8)
    else:
        yy, xx = np.mgrid[:h, :w]
        cy = int(rng.integers(h // 3, 2 * h // 3))
        cx = int(rng.integers(w // 3, 2 * w // 3))
        ry = int(rng.integers(max(4, h // 10), max(5, h // 6)))
        rx = int(rng.integers(max(5, w // 8), max(6, w // 5)))
        mask[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = 1
        image[mask > 0] = np.array([70, 55, 45], dtype=np.uint8)

    return image, mask


def write_task(root: Path, split: str, task: str, kind: str, count: int, image_size: int, offset: int):
    img_dir = root / split / task / "npy_imgs"
    gt_dir = root / split / task / "npy_gts"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(count):
        image, mask = make_sample(kind, image_size, seed=offset + idx)
        name = f"DemoSource_{idx:03d}.npy"
        img_path = img_dir / name
        gt_path = gt_dir / name
        np.save(img_path, image)
        np.save(gt_path, mask)
        rows.append(
            {
                "split": split,
                "task": task,
                "image": str(img_path.relative_to(root)),
                "mask": str(gt_path.relative_to(root)),
                "foreground_pixels": int(mask.sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="demo_data")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--train_per_task", type=int, default=12)
    parser.add_argument("--val_per_task", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)

    tasks = [
        ("Handheld_ConcreteCrack", "crack"),
        ("VehicleProfiler_AsphaltCrack", "crack"),
        ("Handheld_ConcretePavementPothole", "pothole"),
        ("Aerial_RoadCrack", "crack"),
    ]

    rows = []
    for task_idx, (task, kind) in enumerate(tasks):
        rows.extend(write_task(out, "train", task, kind, args.train_per_task, args.image_size, 1000 + task_idx * 1000))
        rows.extend(write_task(out, "val", task, kind, args.val_per_task, args.image_size, 5000 + task_idx * 1000))

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "task", "image", "mask", "foreground_pixels"])
        writer.writeheader()
        writer.writerows(rows)

    readme = out / "README.md"
    readme.write_text(
        "# InfraSegFM Demo Data\n\n"
        "Synthetic mini benchmark for testing the InfraSegFM training and evaluation pipeline.\n\n"
        f"- image size: {args.image_size} x {args.image_size}\n"
        f"- tasks: {len(tasks)}\n"
        f"- train samples per task: {args.train_per_task}\n"
        f"- validation samples per task: {args.val_per_task}\n"
        f"- manifest: `manifest.csv`\n\n"
        "This dataset is only for software verification and is not used for reporting model performance.\n",
        encoding="utf-8",
    )
    print(f"Demo dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()

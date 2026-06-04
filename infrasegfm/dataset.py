"""Dataset utilities for InfraSegFM.

Supported task-folder layouts:

    TaskFolder/npy_imgs/*.npy   + TaskFolder/npy_gts/*.npy
    TaskFolder/images/*         + TaskFolder/masks/*

Images may be NumPy arrays or common image files. Masks are read as binary
foreground/background labels where values greater than zero are foreground.
"""

from __future__ import annotations

import os
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .taxonomy import (
    TASK_FOLDER_META,
    asset_level_1_dict,
    asset_level_1_map,
    asset_level_2_dict,
    asset_level_2_map,
    asset_level_3_dict,
    asset_level_3_map,
    platform_map,
    task_id,
)

IMG_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMG_EXTS


def _hash_source_prefix(prefix: str, buckets: int) -> int:
    if buckets <= 1 or not prefix:
        return 0
    h = 2166136261
    for ch in prefix:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return int(h % buckets)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        img = np.load(path)
    else:
        img = np.asarray(Image.open(path).convert("RGB"))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    elif img.ndim == 3 and img.shape[0] == 3:
        pass
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape {img.shape} in {path}")

    img = img.astype(np.float32, copy=False)
    # SAM's preprocessing expects RGB values on the 0..255 scale.
    if img.max(initial=0.0) <= 1.0:
        img = img * 255.0
    return img


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path).convert("L"))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _resize_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
    image_size: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if image_size is None:
        return img, mask
    image_size = int(image_size)
    if image_size <= 0:
        return img, mask

    if img.shape[-2:] == (image_size, image_size) and mask.shape[-2:] == (image_size, image_size):
        return img, mask

    img_t = torch.from_numpy(img).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    img_t = F.interpolate(img_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mask_t = F.interpolate(mask_t, size=(image_size, image_size), mode="nearest")
    return img_t.squeeze(0).numpy(), mask_t.squeeze(0).squeeze(0).numpy().astype(np.uint8)


def _compute_box(mask: np.ndarray, training: bool) -> np.ndarray:
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return np.array([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)

    x1, x2 = int(x.min()), int(x.max())
    y1, y2 = int(y.min()), int(y.max())
    if training:
        jitter = 3
        h, w = mask.shape
        x1 = max(0, x1 - random.randint(0, jitter))
        y1 = max(0, y1 - random.randint(0, jitter))
        x2 = min(w - 1, x2 + random.randint(0, jitter))
        y2 = min(h - 1, y2 + random.randint(0, jitter))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def resolve_task_folder_meta(task_folder: str) -> Tuple[int, int, int, int, int]:
    """Resolve task metadata indices from a folder name.

    Known folders use the explicit taxonomy. Unknown folders fall back to the
    ``<Platform>_<Task>`` convention and map the task to the model's unknown
    task embedding if it is not present in ``taxonomy.py``.
    """

    if task_folder in TASK_FOLDER_META:
        p0, l1, l2, l3, task = TASK_FOLDER_META[task_folder]
        return (
            int(platform_map[p0]),
            int(asset_level_1_map[l1]),
            int(asset_level_2_map[l2]),
            int(asset_level_3_map[l3]),
            int(task_id[task]),
        )

    if "_" in task_folder:
        platform_name, task_name = task_folder.split("_", 1)
    else:
        platform_name, task_name = "Handheld", task_folder
    if platform_name not in platform_map:
        platform_name = "Handheld"

    def find_level(level_dict: Dict[str, Sequence[str]], default_key: str) -> str:
        for key, tasks in level_dict.items():
            if task_name in tasks:
                return key
        return default_key

    l1 = find_level(asset_level_1_dict, next(iter(asset_level_1_dict)))
    l2 = find_level(asset_level_2_dict, next(iter(asset_level_2_dict)))
    l3 = find_level(asset_level_3_dict, next(iter(asset_level_3_dict)))
    return (
        int(platform_map[platform_name]),
        int(asset_level_1_map[l1]),
        int(asset_level_2_map[l2]),
        int(asset_level_3_map[l3]),
        int(task_id.get(task_name, -1)),
    )


def _find_data_dirs(task_dir: Path) -> Optional[Tuple[Path, Path]]:
    candidates = [
        ("npy_imgs", "npy_gts"),
        ("images", "masks"),
        ("imgs", "masks"),
        ("images", "labels"),
    ]
    for img_name, mask_name in candidates:
        img_dir = task_dir / img_name
        mask_dir = task_dir / mask_name
        if img_dir.is_dir() and mask_dir.is_dir():
            return img_dir, mask_dir
    return None


def _collect_pairs(task_dir: Path) -> List[Tuple[Path, Path]]:
    dirs = _find_data_dirs(task_dir)
    if dirs is None:
        return []
    img_dir, mask_dir = dirs

    images = {p.stem: p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS}
    masks = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.is_file() and p.suffix.lower() in MASK_EXTS}
    return [(images[stem], masks[stem]) for stem in sorted(images.keys() & masks.keys())]


class InfraSegDataset(Dataset):
    """Dataset over one split root containing one or more task folders."""

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        image_size: Optional[int] = 256,
        dataset_buckets: int = 1024,
        moe_cache_name: str = "_moe_routing_cache.json",
    ) -> None:
        self.data_root = Path(data_root)
        self.train = bool(train)
        self.image_size = image_size
        self.dataset_buckets = int(dataset_buckets)
        self.moe_cache_name = str(moe_cache_name)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        if _find_data_dirs(self.data_root) is not None:
            task_dirs = [self.data_root]
        else:
            task_dirs = [p for p in sorted(self.data_root.iterdir()) if p.is_dir()]

        self.samples: List[Tuple[Path, Path, str, Tuple[int, int, int, int, int]]] = []
        for task_dir in task_dirs:
            task_name = task_dir.name
            meta = resolve_task_folder_meta(task_name)
            for img_path, mask_path in _collect_pairs(task_dir):
                self.samples.append((img_path, mask_path, task_name, meta))

        if not self.samples:
            raise ValueError(
                f"No paired samples found under {self.data_root}. Expected task folders "
                "with npy_imgs/npy_gts or images/masks subdirectories."
            )

        self._moe_assign = self._load_moe_cache()

    def _load_moe_cache(self) -> Dict[str, Dict]:
        search_paths = [
            self.data_root / self.moe_cache_name,
            self.data_root.parent / self.moe_cache_name,
        ]
        for path in search_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                assignments = obj.get("assignments", {}) or {}
            except Exception:
                continue
            out = {}
            for key, value in assignments.items():
                norm = os.path.normpath(str(key))
                out[norm] = value
                out[os.path.abspath(norm)] = value
            return out
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def task_folders(self) -> List[str]:
        return [sample[2] for sample in self.samples]

    def __getitem__(self, index: int):
        img_path, mask_path, task_name, meta = self.samples[index]
        img = _read_image(img_path)
        mask = _read_mask(mask_path)
        img, mask = _resize_image_and_mask(img, mask, self.image_size)
        box = _compute_box(mask, self.train)

        platform_idx, l1_idx, l2_idx, l3_idx, task_idx = meta
        src_prefix = img_path.stem.split("_", 1)[0] if "_" in img_path.stem else ""
        src_bucket = _hash_source_prefix(src_prefix, self.dataset_buckets)

        data = {
            "img": torch.from_numpy(img).float(),
            "box": torch.from_numpy(box).float(),
            "platform": int(platform_idx),
            "context": (int(l1_idx), int(l2_idx), int(l3_idx), int(task_idx), int(src_bucket)),
            "name": str(mask_path),
            "task_folder": task_name,
        }
        rec = self._moe_assign.get(os.path.normpath(str(mask_path)))
        if rec is None:
            rec = self._moe_assign.get(os.path.abspath(os.path.normpath(str(mask_path))))
        if isinstance(rec, dict) and "expert" in rec:
            data["moe_target"] = int(rec.get("expert", -1))
            data["moe_cluster"] = int(rec.get("cluster", 0))
            data["moe_k"] = int(rec.get("k_used", 1))
        label = torch.from_numpy(mask[None, :, :]).long()
        return data, label


class InfraTaskDataset(InfraSegDataset):
    """Alias for evaluating a single task folder."""

    pass

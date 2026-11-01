# -*- coding: utf-8 -*-
"""Dataset loaders for InfraSegFM.

Directory layout
----------------
Each task is a folder containing:
  <TaskFolder>/npy_imgs/*.npy
  <TaskFolder>/npy_gts/*.npy

Example:
  <DATA_ROOT>/Handheld_ConcreteCrack/npy_imgs/xxx.npy
  <DATA_ROOT>/Handheld_ConcreteCrack/npy_gts/xxx.npy

The loader returns:
  data: {
    "img": Tensor[3,H,W] float32
    "box": Tensor[4] float32  (xyxy)
    "platform": int  (L0)
    "context": tuple[int,int,int,int,int]  (L1,L2,L3,L4,source_bucket)
    "name": str (gt path)
    "task_folder": str
    optional: moe_target / moe_cluster / moe_k (from routing cache)
  }
  label: Tensor[1,H,W] int64
"""

from __future__ import annotations

import os
import random
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.datainfo import (
    TASK_FOLDER_META,
    platform_map,
    asset_level_1_dict,
    asset_level_1_map,
    asset_level_2_dict,
    asset_level_2_map,
    asset_level_3_dict,
    asset_level_3_map,
    task_id,
)


def _safe_norm_img(img: np.ndarray) -> np.ndarray:
    """Normalize a npy image to float32 CHW in [0,1]."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    elif img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    mx = float(img.max()) if img.size > 0 else 0.0
    if mx > 1.0:
        img = img / 255.0
    return img


def _safe_load_gt(gt: np.ndarray) -> np.ndarray:
    if gt.dtype not in (np.uint8, np.int64, np.int32):
        gt = gt.astype(np.uint8)
    if gt.ndim == 3:
        gt = gt[..., 0]
    return gt


def _compute_box_from_mask(mask: np.ndarray, training: bool) -> np.ndarray:
    """Compute tight box from binary mask; optionally jitter during training."""
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return np.array([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)

    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()

    if training:
        h, w = mask.shape
        jitter = 3
        x1 = max(0, x1 - random.randint(0, jitter))
        y1 = max(0, y1 - random.randint(0, jitter))
        x2 = min(w - 1, x2 + random.randint(0, jitter))
        y2 = min(h - 1, y2 + random.randint(0, jitter))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _hash_source_prefix(prefix: str, buckets: int) -> int:
    """Stable hash -> [0, buckets). Used as a lightweight source bucket id."""
    if buckets <= 1:
        return 0
    if not prefix:
        return 0

    h = 2166136261
    for ch in prefix:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return int(h % buckets)




def _normalize_keywords(value) -> List[str]:
    """Normalize optional source-exclusion keywords from CLI/env/config."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    out = []
    for v in value:
        v = str(v).strip().lower()
        if v:
            out.append(v)
    return out


def _path_has_keyword(path: str, keywords: Sequence[str]) -> bool:
    """Case-insensitive substring match against full path and basename."""
    if not keywords:
        return False
    low = os.path.normpath(path).lower()
    base = os.path.basename(low)
    return any((kw in low) or (kw in base) for kw in keywords)


def resolve_task_folder_meta(task_folder: str) -> Tuple[int, int, int, int, int]:
    """Resolve (platform, L1, L2, L3, task) indices from TaskFolder name."""
    if task_folder in TASK_FOLDER_META:
        p0, l1, l2, l3, t = TASK_FOLDER_META[task_folder]
        return (
            int(platform_map[p0]),
            int(asset_level_1_map[l1]),
            int(asset_level_2_map[l2]),
            int(asset_level_3_map[l3]),
            int(task_id[t]),
        )

    # Fallback: parse pattern "<Platform>_<Task>"
    if "_" in task_folder:
        p0_str, task_str = task_folder.split("_", 1)
    else:
        p0_str, task_str = "Handheld", task_folder

    if p0_str not in platform_map:
        p0_str = "Handheld"

    def _find_level(level_dict: Dict[str, Sequence[str]], default_key: str) -> str:
        for k, v in level_dict.items():
            if task_str in v:
                return k
        return default_key

    l1_str = _find_level(asset_level_1_dict, default_key=list(asset_level_1_dict.keys())[0])
    l2_str = _find_level(asset_level_2_dict, default_key=list(asset_level_2_dict.keys())[0])
    l3_str = _find_level(asset_level_3_dict, default_key=list(asset_level_3_dict.keys())[0])

    return (
        int(platform_map[p0_str]),
        int(asset_level_1_map[l1_str]),
        int(asset_level_2_map[l2_str]),
        int(asset_level_3_map[l3_str]),
        int(task_id.get(task_str, -1)),
    )


class InfraSegDataset(Dataset):
    """Multi-task dataset under a split root (train/val/test)."""

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        sequence: bool = False,
        dataset_buckets: int = 1024,
        exclude_source_keywords: Optional[Sequence[str]] = None,
        moe_cache_name: Optional[str] = None,
        **kwargs,
    ):
        self.data_root = os.path.normpath(data_root)
        self.train = bool(train)
        self.sequence = bool(sequence)
        self.dataset_buckets = int(dataset_buckets)
        self.exclude_source_keywords = _normalize_keywords(
            exclude_source_keywords if exclude_source_keywords is not None else os.getenv("INFRASEG_EXCLUDE_SOURCE_KEYWORDS", "")
        )
        self.moe_cache_name = str(moe_cache_name or os.getenv("INFRASEG_MOE_CACHE_NAME", "_moe_routing_cache.json"))

        # Optional MoE routing cache.
        self._moe_assign: Dict[str, Dict] = {}
        self._moe_cache_path: Optional[str] = None
        for p in [
            os.path.join(self.data_root, self.moe_cache_name),
            os.path.join(os.path.dirname(self.data_root), self.moe_cache_name),
            os.path.join(self.data_root, "_moe_routing_cache.json"),
            os.path.join(os.path.dirname(self.data_root), "_moe_routing_cache.json"),
        ]:
            if p and os.path.exists(p):
                self._moe_cache_path = p
                break
        if self._moe_cache_path:
            try:
                import json

                with open(self._moe_cache_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                self._moe_assign = obj.get("assignments", {}) or {}
                print(
                    f"[InfraSegDataset] MoE cache loaded: {len(self._moe_assign)} assignments "
                    f"({os.path.basename(self._moe_cache_path)})"
                )
            except Exception:
                self._moe_assign = {}

        self.gt_paths: List[str] = []
        self.task_folders: List[str] = []
        self.meta_indices: List[Tuple[int, int, int, int, int]] = []

        for folder in sorted(os.listdir(self.data_root)):
            folder_path = join(self.data_root, folder)
            if not os.path.isdir(folder_path):
                continue

            imgs_dir = join(folder_path, "npy_imgs")
            gts_dir = join(folder_path, "npy_gts")
            if not (os.path.isdir(imgs_dir) and os.path.isdir(gts_dir)):
                continue

            meta = resolve_task_folder_meta(folder)

            for f in sorted(os.listdir(gts_dir)):
                if not f.endswith(".npy"):
                    continue
                gt_path = join(gts_dir, f)
                img_path = join(imgs_dir, f)
                if not os.path.exists(img_path):
                    continue
                if _path_has_keyword(gt_path, self.exclude_source_keywords) or _path_has_keyword(img_path, self.exclude_source_keywords):
                    continue
                self.gt_paths.append(gt_path)
                self.task_folders.append(folder)
                self.meta_indices.append(meta)

        if len(self.gt_paths) == 0:
            raise ValueError(
                f"[InfraSegDataset] No valid samples found under {data_root}. "
                f"Please check folder structure and npy_imgs/npy_gts pairing."
            )

        if self.exclude_source_keywords:
            print(f"[InfraSegDataset] Excluding source keywords: {self.exclude_source_keywords}")
        print(f"[InfraSegDataset] Loaded samples: {len(self.gt_paths)}")

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index: int):
        gt_path = self.gt_paths[index]
        folder = self.task_folders[index]
        img_path = gt_path.replace("npy_gts", "npy_imgs")

        img = _safe_norm_img(np.load(img_path))
        gt = _safe_load_gt(np.load(gt_path))
        box = _compute_box_from_mask(gt, self.train)

        platform_idx, l1_idx, l2_idx, l3_idx, task_idx = self.meta_indices[index]

        # Source bucket id (optional): filename pattern: <SourcePrefix>_<Stem>.npy
        fname = os.path.basename(gt_path)
        stem = fname[:-4] if fname.endswith(".npy") else fname
        src_prefix = stem.split("_", 1)[0] if "_" in stem else ""
        src_bucket = _hash_source_prefix(src_prefix, self.dataset_buckets)

        data: Dict = {
            "img": torch.from_numpy(img).float(),
            "box": torch.from_numpy(box).float(),
            "platform": int(platform_idx),
            "context": (int(l1_idx), int(l2_idx), int(l3_idx), int(task_idx), int(src_bucket)),
            "name": gt_path,
            "task_folder": folder,
        }

        if self._moe_assign:
            rec = self._moe_assign.get(gt_path, None)
            if rec is not None and isinstance(rec, dict) and ("expert" in rec):
                try:
                    data["moe_target"] = int(rec.get("expert", 0))
                    data["moe_cluster"] = int(rec.get("cluster", 0))
                    data["moe_k"] = int(rec.get("k_used", 1))
                except Exception:
                    pass

        return data, torch.from_numpy(gt[None, :, :]).long()


class InfraTaskDataset(Dataset):
    """Dataset for a single TaskFolder (used in task-wise evaluation)."""

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        sequence: bool = False,
        dataset_buckets: int = 1024,
        exclude_source_keywords: Optional[Sequence[str]] = None,
        moe_cache_name: Optional[str] = None,
        **kwargs,
    ):
        self.data_root = os.path.normpath(data_root)
        self.train = bool(train)
        self.sequence = bool(sequence)
        self.dataset_buckets = int(dataset_buckets)
        self.exclude_source_keywords = _normalize_keywords(
            exclude_source_keywords if exclude_source_keywords is not None else os.getenv("INFRASEG_EXCLUDE_SOURCE_KEYWORDS", "")
        )
        self.moe_cache_name = str(moe_cache_name or os.getenv("INFRASEG_MOE_CACHE_NAME", "_moe_routing_cache.json"))

        self._moe_assign: Dict[str, Dict] = {}
        self._moe_cache_path: Optional[str] = None
        for p in [
            os.path.join(self.data_root, self.moe_cache_name),
            os.path.join(os.path.dirname(self.data_root), self.moe_cache_name),
            os.path.join(os.path.dirname(os.path.dirname(self.data_root)), self.moe_cache_name),
            os.path.join(self.data_root, "_moe_routing_cache.json"),
            os.path.join(os.path.dirname(self.data_root), "_moe_routing_cache.json"),
            os.path.join(os.path.dirname(os.path.dirname(self.data_root)), "_moe_routing_cache.json"),
        ]:
            if p and os.path.exists(p):
                self._moe_cache_path = p
                break
        if self._moe_cache_path:
            try:
                import json

                with open(self._moe_cache_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                self._moe_assign = obj.get("assignments", {}) or {}
                print(
                    f"[InfraTaskDataset] MoE cache loaded: {len(self._moe_assign)} assignments "
                    f"({os.path.basename(self._moe_cache_path)})"
                )
            except Exception:
                self._moe_assign = {}

        imgs_dir = join(self.data_root, "npy_imgs")
        gts_dir = join(self.data_root, "npy_gts")
        if not (os.path.isdir(imgs_dir) and os.path.isdir(gts_dir)):
            raise ValueError(f"[InfraTaskDataset] Invalid task folder: {data_root}")

        self.gt_paths: List[str] = []
        for f in sorted(os.listdir(gts_dir)):
            if f.endswith(".npy"):
                gp = join(gts_dir, f)
                ip = join(imgs_dir, f)
                if os.path.exists(ip):
                    if _path_has_keyword(gp, self.exclude_source_keywords) or _path_has_keyword(ip, self.exclude_source_keywords):
                        continue
                    self.gt_paths.append(gp)

        if len(self.gt_paths) == 0:
            raise ValueError(f"[InfraTaskDataset] No samples found under {data_root}")

        self.task_folder = os.path.basename(self.data_root)
        self.meta = resolve_task_folder_meta(self.task_folder)

        if self.exclude_source_keywords:
            print(f"[InfraTaskDataset] Excluding source keywords: {self.exclude_source_keywords}")
        print(f"[InfraTaskDataset] Loaded samples: {len(self.gt_paths)}")

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index: int):
        gt_path = self.gt_paths[index]
        img_path = gt_path.replace("npy_gts", "npy_imgs")

        img = _safe_norm_img(np.load(img_path))
        gt = _safe_load_gt(np.load(gt_path))
        box = _compute_box_from_mask(gt, self.train)

        platform_idx, l1_idx, l2_idx, l3_idx, task_idx = self.meta

        fname = os.path.basename(gt_path)
        stem = fname[:-4] if fname.endswith(".npy") else fname
        src_prefix = stem.split("_", 1)[0] if "_" in stem else ""
        src_bucket = _hash_source_prefix(src_prefix, self.dataset_buckets)

        data: Dict = {
            "img": torch.from_numpy(img).float(),
            "box": torch.from_numpy(box).float(),
            "platform": int(platform_idx),
            "context": (int(l1_idx), int(l2_idx), int(l3_idx), int(task_idx), int(src_bucket)),
            "name": gt_path,
            "task_folder": self.task_folder,
        }

        if self._moe_assign:
            rec = self._moe_assign.get(gt_path, None)
            if rec is not None and isinstance(rec, dict) and ("expert" in rec):
                try:
                    data["moe_target"] = int(rec.get("expert", -1))
                    data["moe_cluster"] = int(rec.get("cluster", 0))
                    data["moe_k"] = int(rec.get("k_used", 1))
                except Exception:
                    pass

        return data, torch.from_numpy(gt[None, :, :]).long()

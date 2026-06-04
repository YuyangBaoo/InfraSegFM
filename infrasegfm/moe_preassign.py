# -*- coding: utf-8 -*-
"""
MoE pre-assignment (task-first, dynamic-K) for InfraSegFM.

Goal
----
Before training starts, we deterministically assign each training sample to a
"target expert" using a lightweight clustering rule:
  - first split by task folder
  - within each task, compute per-image style features (+ optional label stats)
  - choose dynamic K (per task) and run KMeans
  - enforce min cluster size (avoid tiny clusters)
  - map clusters to global expert ids (distinct within each task)

The result is stored as a cache file under:
  <train_root>/_moe_routing_cache.json

Dataset.py can then attach "moe_target" to each sample, and pretrain.py can
add an auxiliary routing supervision loss on the gates.

This module is **offline-only** and does NOT affect inference unless you choose
to keep the aux loss on during finetuning.

Dependencies:
  numpy, opencv-python, scikit-learn, tqdm
"""

from __future__ import annotations
import os
import json
import math
import zlib
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans



# Cache schema version: bump when routing cache format or feature logic changes
CACHE_VERSION = 2
# -----------------------
# Config (keep defaults stable)
# -----------------------
@dataclass
class PreassignConfig:
    """Configuration for offline MoE pre-assignment.

    Some call-sites in this repo pass extra knobs (e.g., `dataset_buckets`,
    `label_buckets`, `include_style_features`). We keep them here for
    backward-compatibility so that enabling forced routing in evaluation will
    not crash with unexpected kwargs.
    """

    # -------- Compatibility fields (safe to ignore if unused) --------
    dataset_buckets: int = 1024
    label_buckets: int = 256
    include_style_features: bool = True

    # ---------------- Core clustering controls ----------------
    k_max: int = 6
    n_min_for_cluster: int = 800
    d_min_for_cluster: int = 2

    allow_single_dataset_split: bool = True
    n_single_dataset_split: int = 4000
    style_var_y_std_thresh: float = 0.08   # mean_y std in [0,1]
    style_var_lap_std_thresh: float = 80.0

    n_big: int = 6000
    d_big: int = 6
    k_small: int = 2
    k_mid: int = 3
    k_large: int = 4
    k_xl: int = 5

    min_cluster_frac: float = 0.05
    min_cluster_count: int = 300

    include_label_features: bool = True
    ignore_label_ids: Tuple[int, ...] = (255,)

    num_workers: int = 16
    max_files_per_task: Optional[int] = None  # None = all
    exclude_source_keywords: Tuple[str, ...] = ()
    cache_name: str = "_moe_routing_cache.json"


def _normalize_keywords(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    out = []
    for v in value:
        v = str(v).strip().lower()
        if v:
            out.append(v)
    return tuple(out)


def _path_has_keyword(path: str, keywords: Tuple[str, ...]) -> bool:
    if not keywords:
        return False
    low = os.path.normpath(path).lower()
    base = os.path.basename(low)
    return any((kw in low) or (kw in base) for kw in keywords)


def _canonical_jsonable(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def _hash32(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def _safe_norm_img(arr: np.ndarray) -> np.ndarray:
    """Convert npy image to uint8 RGB for feature extraction."""
    a = arr
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        # (C,H,W)->(H,W,C)
        if a.shape[0] in (1, 3, 4) and a.shape[0] < a.shape[-1]:
            a = np.transpose(a, (1, 2, 0))
        if a.shape[-1] >= 3:
            a = a[..., :3]
        else:
            a = np.repeat(a, 3, axis=-1)
    a = a.astype(np.float32)
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _estimate_noise(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    diff = np.abs(g - blur)
    return float(np.median(diff))


def _colorfulness_hasler(rgb_u8: np.ndarray) -> float:
    img = rgb_u8.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _read_npy_rgb_u8(img_path: str) -> np.ndarray:
    img = np.load(img_path, allow_pickle=False)
    return _safe_norm_img(img)


def _read_npy_mask_i32(gt_path: str) -> np.ndarray:
    m = np.load(gt_path, allow_pickle=False)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim == 3 and m.shape[0] < 20:
        m = np.argmax(m, axis=0)
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {m.shape}")
    return m.astype(np.int32)


def extract_features_one(img_path: str, gt_path: str, cfg: PreassignConfig) -> Dict:
    rgb = _read_npy_rgb_u8(img_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    img_f = rgb.astype(np.float32) / 255.0
    y = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]

    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    out = {
        "img_path": img_path,
        "gt_path": gt_path,
        "mean_y": float(y.mean()),
        "std_y": float(y.std()),
        "mean_s": float(s.mean()),
        "std_s": float(s.std()),
        "mean_v": float(v.mean()),
        "std_v": float(v.std()),
        "lap_var": lap_var,
        "edge_density": edge_density,
        "noise_level": _estimate_noise(gray),
        "colorfulness": _colorfulness_hasler(rgb),
    }

    if cfg.include_label_features and os.path.exists(gt_path):
        try:
            m = _read_npy_mask_i32(gt_path)
            flat = m.reshape(-1)
            for ig in cfg.ignore_label_ids:
                flat = flat[flat != ig]
            if flat.size == 0:
                out.update({"gt_found": 1, "n_labels": 0, "dominant_label": -1, "fg_ratio": 0.0, "entropy_labels": 0.0})
            else:
                vals, counts = np.unique(flat, return_counts=True)
                n_labels = int(vals.size)
                dominant = int(vals[np.argmax(counts)])
                fg_ratio = float(np.mean(flat > 0))
                p = counts.astype(np.float64) / (counts.sum() + 1e-12)
                ent = float(-(p * np.log(p + 1e-12)).sum())
                out.update({"gt_found": 1, "n_labels": n_labels, "dominant_label": dominant, "fg_ratio": fg_ratio, "entropy_labels": ent})
        except Exception:
            out.update({"gt_found": 0, "n_labels": 0, "dominant_label": -1, "fg_ratio": 0.0, "entropy_labels": 0.0})
    else:
        out.update({"gt_found": 0, "n_labels": 0, "dominant_label": -1, "fg_ratio": 0.0, "entropy_labels": 0.0})

    return out


def decide_k0(n_images: int, n_datasets: int, cfg: PreassignConfig) -> int:
    if n_images < cfg.n_min_for_cluster or n_datasets < cfg.d_min_for_cluster:
        return 1

    if n_datasets <= 3:
        k = cfg.k_small
    elif n_datasets <= 5:
        k = cfg.k_mid
    elif n_datasets <= 8:
        k = cfg.k_large
    else:
        k = cfg.k_xl

    if n_images >= cfg.n_big and n_datasets >= cfg.d_big:
        k = max(k, cfg.k_xl)
    elif n_images >= (cfg.n_big // 2):
        k = max(k, cfg.k_large)

    return int(max(1, min(k, cfg.k_max)))


def maybe_split_single_dataset(features: List[Dict], k0: int, cfg: PreassignConfig) -> int:
    if not cfg.allow_single_dataset_split:
        return k0
    n = len(features)
    if n < cfg.n_single_dataset_split:
        return k0
    # estimate style variance from mean_y and lap_var
    y = np.array([r["mean_y"] for r in features], dtype=np.float32)
    lap = np.array([r["lap_var"] for r in features], dtype=np.float32)
    if float(y.std()) >= cfg.style_var_y_std_thresh or float(lap.std()) >= cfg.style_var_lap_std_thresh:
        return max(k0, 2)
    return k0


def _passes_min_cluster(labels: np.ndarray, cfg: PreassignConfig) -> bool:
    n = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)
    min_cnt = int(counts.min())
    min_frac = float(min_cnt / max(1, n))
    return (min_cnt >= cfg.min_cluster_count) or (min_frac >= cfg.min_cluster_frac)


def _kmeans_with_checks(X: np.ndarray, k0: int, cfg: PreassignConfig) -> Tuple[np.ndarray, int, np.ndarray]:
    n = X.shape[0]
    k0 = int(max(1, min(k0, cfg.k_max, n)))
    if k0 <= 1:
        return np.zeros((n,), dtype=np.int32), 1, np.zeros((1, X.shape[1]), dtype=np.float32)

    for k in range(k0, 0, -1):
        if k <= 1:
            return np.zeros((n,), dtype=np.int32), 1, np.zeros((1, X.shape[1]), dtype=np.float32)
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if _passes_min_cluster(labels, cfg):
            centers = km.cluster_centers_.astype(np.float32)
            return labels.astype(np.int32), int(k), centers

    return np.zeros((n,), dtype=np.int32), 1, np.zeros((1, X.shape[1]), dtype=np.float32)


def _assign_experts_for_task(task_name: str, k_used: int, expert_num: int) -> List[int]:
    """Deterministic expert id list of length k_used, distinct within task."""
    k_used = int(max(1, min(k_used, expert_num)))
    seed = _hash32(task_name)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(expert_num).tolist()
    return perm[:k_used]


def build_routing_cache(train_root: str, expert_num: int, cfg: PreassignConfig, force_rebuild: bool = False) -> str:
    """
    Build or reuse cache.
    Returns cache path.
    """
    cfg.exclude_source_keywords = _normalize_keywords(getattr(cfg, "exclude_source_keywords", ()))
    cache_name = str(getattr(cfg, "cache_name", "_moe_routing_cache.json") or "_moe_routing_cache.json")
    cache_path = os.path.join(train_root, cache_name)

    if (not force_rebuild) and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            meta = obj.get("meta", {}) or {}
            # Reuse cache ONLY when expert_num and config match, and schema version is current.
            if int(meta.get("expert_num", -1)) == int(expert_num) and int(meta.get("cache_version", 0)) == int(CACHE_VERSION):
                cached_cfg = meta.get("config", None)
                if isinstance(cached_cfg, dict):
                    # Compare through JSON form so tuple/list differences do not invalidate reusable caches.
                    if _canonical_jsonable(cached_cfg) == _canonical_jsonable(asdict(cfg)):
                        return cache_path
        except Exception:
            pass
    # scan task folders
    task_folders = []
    for name in sorted(os.listdir(train_root)):
        p = os.path.join(train_root, name)
        if not os.path.isdir(p):
            continue
        imgs_dir = os.path.join(p, "npy_imgs")
        gts_dir = os.path.join(p, "npy_gts")
        if os.path.isdir(imgs_dir) and os.path.isdir(gts_dir):
            task_folders.append((name, p))

    if not task_folders:
        raise RuntimeError(f"No task folders found under: {train_root}")

    assignments: Dict[str, Dict] = {}
    task_meta: Dict[str, Dict] = {}

    for task_name, task_dir in tqdm(task_folders, desc="MoE preassign (task)", ncols=100, mininterval=0.5):
        imgs_dir = os.path.join(task_dir, "npy_imgs")
        gts_dir = os.path.join(task_dir, "npy_gts")
        files = sorted([f for f in os.listdir(gts_dir) if f.endswith(".npy")])
        if cfg.max_files_per_task is not None:
            files = files[: int(cfg.max_files_per_task)]

        # paired paths
        pairs = []
        for f in files:
            gt_path = os.path.join(gts_dir, f)
            img_path = os.path.join(imgs_dir, f)
            if os.path.exists(img_path):
                if _path_has_keyword(gt_path, cfg.exclude_source_keywords) or _path_has_keyword(img_path, cfg.exclude_source_keywords):
                    continue
                pairs.append((img_path, gt_path))

        if not pairs:
            continue

        # dataset prefixes
        prefixes = []
        for _, gt_path in pairs:
            fname = os.path.basename(gt_path)
            stem = fname[:-4] if fname.endswith(".npy") else fname
            pref = stem.split("_", 1)[0] if "_" in stem else ""
            prefixes.append(pref)
        n_images = len(pairs)
        n_datasets = len(set(prefixes))

        # extract features in parallel
        feats: List[Dict] = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            futs = [ex.submit(extract_features_one, ip, gp, cfg) for ip, gp in pairs]
            pbar = tqdm(total=len(futs), desc=f"  Extract {task_name}", ncols=100, mininterval=0.5, leave=False)
            for fu in as_completed(futs):
                try:
                    feats.append(fu.result())
                except Exception:
                    pass
                pbar.update(1)
            pbar.close()

        if not feats:
            continue

        # build feature matrix
        cols_style = ["mean_y","std_y","mean_s","std_s","lap_var","edge_density","noise_level","colorfulness"]
        cols_label = ["gt_found","n_labels","dominant_label","fg_ratio","entropy_labels"]
        cols: List[str] = []
        if bool(getattr(cfg, "include_style_features", True)):
            cols += cols_style
        if bool(getattr(cfg, "include_label_features", True)):
            cols += cols_label
        if len(cols) == 0:
            # safety: always keep at least style features
            cols = cols_style

        X_raw = np.stack([np.array([r[c] for c in cols], dtype=np.float32) for r in feats], axis=0)
        mu = X_raw.mean(axis=0, keepdims=True)
        sd = X_raw.std(axis=0, keepdims=True) + 1e-6
        X = (X_raw - mu) / sd

        k0 = decide_k0(n_images, n_datasets, cfg)
        k0 = maybe_split_single_dataset(feats, k0, cfg)
        # Run KMeans with safety checks and keep centers so we can apply *training* routing to ID/ODD later.
        labels, k_used, centers = _kmeans_with_checks(X, k0, cfg)

        # map clusters -> expert ids (distinct within task)
        expert_list = _assign_experts_for_task(task_name, k_used, expert_num)
        cluster_to_expert = {int(c): int(expert_list[int(c) % len(expert_list)]) for c in range(k_used)}

        # persist per-sample assignment
        for r, c in zip(feats, labels.tolist()):
            gt_path = r["gt_path"]
            assignments[gt_path] = {
                "task": task_name,
                "cluster": int(c),
                "k_used": int(k_used),
                "expert": int(cluster_to_expert.get(int(c), expert_list[0])),
            }

        # task-level meta
        # distribution
        _, cnts = np.unique(labels, return_counts=True)
        dist = {str(i): int(cnts[i]) for i in range(len(cnts))}
        task_meta[task_name] = {
            "n_images": int(n_images),
            "n_datasets": int(n_datasets),
            "k0": int(k0),
            "k_used": int(k_used),
            "cluster_to_expert": {str(k): int(v) for k, v in cluster_to_expert.items()},
            "cluster_counts": dist,
            # routing model for applying training routing to new splits
            "cols": cols,
            "x_mean": mu.reshape(-1).astype(np.float32).tolist(),
            "x_std": sd.reshape(-1).astype(np.float32).tolist(),
            "centers": centers.astype(np.float32).tolist(),
        }

    obj = {
        "meta": {
            "train_root": train_root,
            "expert_num": int(expert_num),
            "cache_version": int(CACHE_VERSION),
            "config": asdict(cfg),
        },
        "tasks": task_meta,
        "assignments": assignments,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return cache_path


def apply_routing_from_train_cache(
    train_cache_path: str,
    eval_root: str,
    expert_num: int,
    cfg: PreassignConfig,
    force_rebuild: bool = False,
) -> str:
    """Apply *training* routing model to a new split (ID/ODD) and write cache.

    This avoids "transductive" clustering on the evaluation split itself.

    Requirements
    ------------
    - `train_cache_path` must contain per-task `cols`, `x_mean`, `x_std`, `centers`,
      and `cluster_to_expert`.
    - `eval_root` follows the same task layout: <eval_root>/<Task>/npy_imgs + npy_gts.
    """

    train_cache_path = os.path.normpath(train_cache_path)
    eval_root = os.path.normpath(eval_root)
    cfg.exclude_source_keywords = _normalize_keywords(getattr(cfg, "exclude_source_keywords", ()))
    cache_name = str(getattr(cfg, "cache_name", "_moe_routing_cache.json") or "_moe_routing_cache.json")
    out_cache_path = os.path.join(eval_root, cache_name)

    if (not force_rebuild) and os.path.exists(out_cache_path):
        try:
            with open(out_cache_path, "r", encoding="utf-8") as f:
                _obj = json.load(f)
            meta = _obj.get("meta", {}) or {}
            if (
                int(meta.get("expert_num", -1)) == int(expert_num)
                and meta.get("source_train_cache") == train_cache_path
                and int(meta.get("cache_version", 0)) == int(CACHE_VERSION)
            ):
                return out_cache_path
        except Exception:
            pass
    with open(train_cache_path, "r", encoding="utf-8") as f:
        train_obj = json.load(f)
    train_tasks = train_obj.get("tasks", {}) or {}

    # scan eval task folders
    task_folders = []
    for name in sorted(os.listdir(eval_root)):
        p = os.path.join(eval_root, name)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, "npy_imgs")) and os.path.isdir(os.path.join(p, "npy_gts")):
            task_folders.append((name, p))
    if not task_folders:
        raise RuntimeError(f"No task folders found under: {eval_root}")

    assignments: Dict[str, Dict] = {}
    task_meta: Dict[str, Dict] = {}

    for task_name, task_dir in tqdm(task_folders, desc="MoE apply(train->eval)", ncols=100, mininterval=0.5):
        tmeta = train_tasks.get(task_name, None)
        if not isinstance(tmeta, dict):
            # task not seen in training cache -> do NOT force routing (expert=-1 => use gate)
            centers = np.zeros((1, 8), dtype=np.float32)
            cols = ["mean_y","std_y","mean_s","std_s","lap_var","edge_density","noise_level","colorfulness"]
            mu = np.zeros((1, len(cols)), dtype=np.float32)
            sd = np.ones((1, len(cols)), dtype=np.float32)
            cluster_to_expert = {0: -1}
            k_used = 1
        else:
            cols = list(tmeta.get("cols", []))
            if len(cols) == 0:
                cols = ["mean_y","std_y","mean_s","std_s","lap_var","edge_density","noise_level","colorfulness"]
            mu = np.array(tmeta.get("x_mean", []), dtype=np.float32).reshape(1, -1)
            sd = np.array(tmeta.get("x_std", []), dtype=np.float32).reshape(1, -1)
            centers = np.array(tmeta.get("centers", []), dtype=np.float32)
            c2e_raw = tmeta.get("cluster_to_expert", {}) or {}
            cluster_to_expert = {int(k): int(v) for k, v in c2e_raw.items()}
            k_used = int(tmeta.get("k_used", max(1, centers.shape[0])))
            if centers.ndim != 2 or centers.shape[0] < 1:
                centers = np.zeros((1, len(cols)), dtype=np.float32)
                k_used = 1

        imgs_dir = os.path.join(task_dir, "npy_imgs")
        gts_dir = os.path.join(task_dir, "npy_gts")
        files = sorted([f for f in os.listdir(gts_dir) if f.endswith(".npy")])
        if cfg.max_files_per_task is not None:
            files = files[: int(cfg.max_files_per_task)]

        pairs = []
        for fn in files:
            gp = os.path.join(gts_dir, fn)
            ip = os.path.join(imgs_dir, fn)
            if os.path.exists(ip):
                if _path_has_keyword(gp, cfg.exclude_source_keywords) or _path_has_keyword(ip, cfg.exclude_source_keywords):
                    continue
                pairs.append((ip, gp))
        if not pairs:
            continue

        feats: List[Dict] = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            futs = [ex.submit(extract_features_one, ip, gp, cfg) for ip, gp in pairs]
            pbar = tqdm(total=len(futs), desc=f"  Feat {task_name}", ncols=100, mininterval=0.5, leave=False)
            for fu in as_completed(futs):
                try:
                    feats.append(fu.result())
                except Exception:
                    pass
                pbar.update(1)
            pbar.close()

        if not feats:
            continue

        X_raw = np.stack([np.array([r[c] for c in cols], dtype=np.float32) for r in feats], axis=0)
        if mu.shape[1] != X_raw.shape[1]:
            # safety: re-init normalization if shape mismatch
            mu = X_raw.mean(axis=0, keepdims=True)
            sd = X_raw.std(axis=0, keepdims=True) + 1e-6
        X = (X_raw - mu) / (sd + 1e-6)

        # assign nearest center
        # (N,C) vs (K,C)
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1).astype(np.int32)

        # persist per-sample assignment
        for r, c in zip(feats, labels.tolist()):
            gt_path = r["gt_path"]
            c = int(c)
            assignments[gt_path] = {
                "task": task_name,
                "cluster": int(c),
                "k_used": int(k_used),
                "expert": int(cluster_to_expert.get(c, -1)),
                "source": "train_cache",
            }

        # meta for this eval split
        _, cnts = np.unique(labels, return_counts=True)
        dist = {str(i): int(cnts[i]) for i in range(len(cnts))}
        task_meta[task_name] = {
            "k_used": int(k_used),
            "cluster_to_expert": {str(k): int(v) for k, v in cluster_to_expert.items()},
            "cluster_counts": dist,
        }

    obj = {
        "meta": {
            "eval_root": eval_root,
            "expert_num": int(expert_num),
            "source_train_cache": train_cache_path,
            "cache_version": int(CACHE_VERSION),
            "config": asdict(cfg),
        },
        "tasks": task_meta,
        "assignments": assignments,
    }

    with open(out_cache_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_cache_path

# -*- coding: utf-8 -*-
from __future__ import annotations

"""
InfraSegFM internal evaluating script (CivilDB/Windows compatible)

Adds visualization (triplet) export:
- For every evaluated sample (or up to --vis_max), save a triplet PNG:
    [Input | GT mask | Pred mask]
- Organized by task folder:
    <work_dir>/vis/ID/<dataset>/<task>/... or <work_dir>/vis/ID/<dataset>/<task>/<sequence>/...

Notes:
- This script assumes data["img"] is in [0,1] range. If your dataset normalizes with mean/std,
  the saved input images may look dark; in that case, tell me your normalization and I’ll add
  a reversible denormalize in 2 lines.
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
import time
import faulthandler
from typing import Dict, List, Optional

join = os.path.join

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torchvision.utils as vutils
import pandas as pd
from tqdm import tqdm

faulthandler.enable(all_threads=True)

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import SAMDecoderTuner, InfraSegFM
from data.dataset import InfraTaskDataset
from utils.moe_preassign import PreassignConfig, build_routing_cache, apply_routing_from_train_cache
from utils.metric import SegmentMetrics


# -------------------------
# Arg parser
# -------------------------
parser = argparse.ArgumentParser("InfraSegFM internal evaluating", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--model_weight", type=str, default="./playground/InfraSegFM/infrasegfm_vit_b.pth",
                    help="path to InfraSegFM model weight")
parser.add_argument("--method", type=str, default="infrasegfm")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--adapter_mode", type=str, default=None, choices=["moe", "shared", "lora", "medsa"],
                    help="Adapter architecture. If unset, read from checkpoint moe_cfg (defaults to 'moe').")

# MoE routing at evaluation time:
#   0  = disable (default, recommended)
#   1  = enable (uses moe_target from cache to force expert routing)
#  -1  = auto (follow checkpoint/flag; mostly for debugging/ablation)
parser.add_argument("--force_preassign_eval", type=int, default=0, choices=[-1, 0, 1])
# Optional overrides (if not set, will read from checkpoint moe_cfg when available)
parser.add_argument("--moe_topk", type=int, default=None)
parser.add_argument("--moe_temp", type=float, default=None)
parser.add_argument("--moe_noise_std", type=float, default=None)
parser.add_argument("--moe_style_bn", type=int, default=None)
parser.add_argument("--moe_style_dropout", type=float, default=None)
parser.add_argument("--moe_disable_style", type=int, default=None, choices=[0, 1],
                    help="Ablation/eval: zero out online style embedding before routing. If unset, read from checkpoint moe_cfg.")
# data
parser.add_argument("--data_path", type=str, default="./playground/InfraSegDB/eval/ID",
                    help="path to InfraSegDB data folder")
parser.add_argument("--metric", type=str, default=["dsc", "hd"], nargs='+',
                    help="evaluation metrics (e.g dsc, hd)")
# eval
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1],
                    help="1 enables DataLoader pinned memory; 0 disables it for Windows stability.")

# visualization
parser.add_argument("--vis", action="store_true",
                    help="save visualization triplets (img | gt | pred)")
parser.add_argument("--vis_max", type=int, default=10_000_000,
                    help="maximum number of visualized samples (very large by default)")
parser.add_argument("--vis_stride", type=int, default=1,
                    help="save every N samples (default: save all)")


# -------------------------
# Helpers
# -------------------------
def _is_task_folder(p: str) -> bool:
    """A task folder must contain both npy_gts and npy_imgs subfolders."""
    return os.path.isdir(join(p, "npy_gts")) and os.path.isdir(join(p, "npy_imgs"))


def _find_train_moe_cache(model_weight: str) -> str | None:
    """Try to locate the training-time MoE routing cache.

    We keep this heuristic so the user can run evaluation with the *same
    command line* as before.

    Search order:
      1) <model_weight_dir>/moe_train_cache_path.txt
      2) ./playground/InfraSegDBv1/train/_moe_routing_cache.json
      3) ./playground/InfraSegDBv1/val/_moe_routing_cache.json
      4) Parent dirs of model_weight (up to 4 levels) for *_moe_routing_cache.json
    """
    try:
        w = os.path.normpath(model_weight)
        wdir = os.path.dirname(w)

        # (1) explicit pointer file
        p_txt = os.path.join(wdir, "moe_train_cache_path.txt")
        if os.path.exists(p_txt):
            try:
                with open(p_txt, "r", encoding="utf-8") as f:
                    p = f.read().strip()
                if p and os.path.exists(p):
                    return os.path.normpath(p)
            except Exception:
                pass

        # (2-3) default InfraSegDBv1 layout
        for cand in [
            os.path.join(".", "playground", "InfraSegDBv1", "train", "_moe_routing_cache.json"),
            os.path.join(".", "playground", "InfraSegDBv1", "val", "_moe_routing_cache.json"),
        ]:
            if os.path.exists(cand):
                return os.path.normpath(cand)

        # (4) walk up weight dir
        cur = wdir
        for _ in range(4):
            cand = os.path.join(cur, "_moe_routing_cache.json")
            if os.path.exists(cand):
                return os.path.normpath(cand)
            cur = os.path.dirname(cur)
    except Exception:
        return None
    return None


def _normalize_task_dir(path: str) -> str:
    """
    Normalize the input path to be a task folder path.

    Acceptable inputs:
      - task folder: .../TaskName
      - sequence folder: .../TaskName/SequenceName
      - accidental npy file: .../TaskName/npy_gts/xxx.npy (or npy_imgs)

    Returns:
      folder path expected to contain `npy_gts` and `npy_imgs`.
    """
    p = os.path.normpath(path)

    # If it's a file path, go up one level
    if os.path.isfile(p):
        p = os.path.dirname(p)

    base = os.path.basename(p).lower()

    # If it's inside npy_gts or npy_imgs, go up to task folder
    if base in ["npy_gts", "npy_imgs"]:
        p = os.path.dirname(p)

    # In case the string ends with .npy but os.path.isfile failed (rare)
    if p.lower().endswith(".npy"):
        # .../task/npy_gts/xxx.npy -> .../task
        p = os.path.dirname(os.path.dirname(p))

    return p


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    """
    NaN-safe mean for 1D tensor (or flattened tensor).
    If all values are non-finite, returns NaN (visible but will not crash).
    """
    x = x.flatten()
    m = torch.isfinite(x)
    if m.any():
        return x[m].mean()
    return torch.tensor(float("nan"), device=x.device)


def dice_empty_aware(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred, gt: Bool tensors of shape [B, 1, H, W]
    Return: [B] float tensor.

    Rules:
      - GT=0 & Pred=0 -> 1
      - GT=0 & Pred>0 -> 0
      - GT>0 -> standard Dice
    """
    pred = pred.bool()
    gt = gt.bool()

    pred_sum = pred.flatten(1).sum(dim=1).float()
    gt_sum = gt.flatten(1).sum(dim=1).float()
    inter = (pred & gt).flatten(1).sum(dim=1).float()

    d = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)

    both_empty = (pred_sum == 0) & (gt_sum == 0)
    gt_empty_pred_nonempty = (gt_sum == 0) & (pred_sum > 0)

    d[both_empty] = 1.0
    d[gt_empty_pred_nonempty] = 0.0
    return d


def _safe_stem_from_name(name: str, data_root: str) -> str:
    """
    Make a stable, Windows-safe filename stem from an arbitrary path/string.
    Uses relpath to keep it readable, then flattens separators.
    """
    try:
        p = os.path.normpath(str(name))
        rel = os.path.relpath(p, os.path.normpath(data_root))
    except Exception:
        rel = str(name)

    rel = rel.replace("npy_gts" + os.sep, "")
    rel = rel.replace("npy_imgs" + os.sep, "")
    rel = rel.replace("/", os.sep).replace("\\", os.sep)

    # Flatten folder structure into filename
    rel = rel.replace(os.sep, "__")

    # Drop extension if exists
    stem = os.path.splitext(rel)[0]

    # Extra safety: remove characters that can be annoying on Windows
    bad = ['<', '>', ':', '"', '|', '?', '*']
    for ch in bad:
        stem = stem.replace(ch, "_")

    return stem


# -------------------------
# Core evaluate
# -------------------------
def evaluate(
    model,
    metric,
    dataloader,
    img_size,
    img_transform,
    box_transform,
    dataset,
    task,
    sequence: Optional[str] = None,
    result_total=None,
    meta=None,
    args=None,
    vis_root: Optional[str] = None,
):
    model.eval()
    device = torch.device(args.device)

    # store per-sample tensors (for aggregation)
    result_task: Dict[str, List[torch.Tensor]] = {m: [] for m in args.metric}

    # visualization counter (global per task call)
    saved_vis = 0
    seen_vis = 0

    pbar = tqdm(dataloader)
    if sequence is not None:
        pbar.set_description(f"Evaluating - {dataset} {task} {sequence}")
    else:
        pbar.set_description(f"Evaluating - {dataset} {task}")

    with torch.no_grad():
        for data, label in pbar:
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    (data["box"].reshape(-1, 2, 2)),
                    data["img"].shape[-2:]
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            label = label.to(device, non_blocking=True, dtype=torch.bool)  # [B,1,H,W]


            # Optional: enforce offline expert routing (few-shot eval)
            if getattr(args, "force_preassign_eval", 0) == 1 and ("moe_target" in data) and ("force_expert" not in data):
                data["force_expert"] = data["moe_target"]
            if "force_expert" in data:
                try:
                    data["force_expert"] = data["force_expert"].to(device, non_blocking=True, dtype=torch.long)
                except Exception:
                    pass
            mask_pred = model(data)

            if mask_pred.shape[-1] != label.shape[-1]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)

            mask_prob = torch.sigmoid(mask_pred)
            mask = (mask_prob > 0.5).bool()  # [B, K, H, W]

            # For multi-mask outputs:
            # - DSC uses empty-aware rule (no NaN from 0/0)
            # - Other metrics (HD) still come from metric module (may return NaN; we'll nan-aggregate)
            per_mask: Dict[str, List[torch.Tensor]] = {m: [] for m in args.metric}

            K = mask.shape[1]
            for k in range(K):
                pred_k = mask[:, k].unsqueeze(1)  # [B,1,H,W]

                # DSC
                if "dsc" in args.metric:
                    dsc_k = dice_empty_aware(pred_k, label)  # [B]
                    per_mask["dsc"].append(dsc_k)

                # Other metrics via metric module (expects [B,1,H,W] bool)
                other_metrics = [m for m in args.metric if m != "dsc"]
                if other_metrics:
                    out = metric(pred_k, label)
                    for m in other_metrics:
                        per_mask[m].append(out[m])  # typically [B]

            # Select best mask by DSC (if DSC exists), else default first mask
            if "dsc" in args.metric:
                dsc_stack = torch.stack(per_mask["dsc"], dim=0)  # [K,B]
                best_dsc, best_idx = dsc_stack.max(dim=0)       # [B], [B]
            else:
                best_idx = torch.zeros(label.shape[0], device=device, dtype=torch.long)
                best_dsc = None

            # Build per-batch final metrics and display
            metric_dict = {}
            batch_metric_values: Dict[str, torch.Tensor] = {}

            for m in args.metric:
                if m == "dsc":
                    result = best_dsc  # [B]
                else:
                    m_stack = torch.stack(per_mask[m], dim=0)  # [K,B]
                    result = m_stack[best_idx, torch.arange(m_stack.shape[1], device=device)]  # [B]

                batch_metric_values[m] = result
                result_task[m].append(result)
                result_total[m].append(result)

                metric_dict[m] = _nanmean(result).item()

            pbar.set_postfix(metric_dict)

            # record instance-level results
            for i, name in enumerate(data["name"]):
                try:
                    file_rel = os.path.relpath(os.path.normpath(name), os.path.normpath(args.data_path))
                except Exception:
                    file_rel = str(name)
                file_rel = file_rel.replace("npy_gts" + os.sep, "")
                meta["File"].append(file_rel)
                for m in args.metric:
                    meta[m.upper()].append(batch_metric_values[m][i].item())

            # -----------------------------
            # Visualization: triplet export
            # -----------------------------
            if args.vis and vis_root is not None and saved_vis < args.vis_max:
                # Get best mask as [B,1,H,W]
                best_mask = mask[torch.arange(mask.size(0), device=device), best_idx].unsqueeze(1).float()
                gt = label.float()

                for i in range(label.size(0)):
                    if saved_vis >= args.vis_max:
                        break

                    # stride-based sampling over seen items
                    if (seen_vis % max(1, args.vis_stride)) != 0:
                        seen_vis += 1
                        continue
                    seen_vis += 1

                    # Build output directory by dataset/task/(sequence)
                    out_dir = join(vis_root, dataset, task)
                    if sequence is not None:
                        out_dir = join(out_dir, sequence)
                    os.makedirs(out_dir, exist_ok=True)

                    stem = _safe_stem_from_name(data["name"][i], args.data_path)

                    # Input image: [C,H,W]
                    img = data["img"][i].detach().cpu().float()
                    if img.dim() == 2:
                        img = img.unsqueeze(0)
                    if img.size(0) == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.size(0) > 3:
                        img = img[:3]

                    # Assume [0,1] for clean visualization; clamp to avoid artifacts
                    img = img.clamp(0, 1)

                    # Masks: make 3-channel for concatenation; keep crisp binary (0/1)
                    gt_i = gt[i].detach().cpu()
                    pr_i = best_mask[i].detach().cpu()
                    gt3 = gt_i.repeat(3, 1, 1).clamp(0, 1)
                    pr3 = pr_i.repeat(3, 1, 1).clamp(0, 1)

                    # Triplet horizontally: (W dimension)
                    triplet = torch.cat([img, gt3, pr3], dim=2)  # [3,H,3W]

                    # Save
                    out_path = join(out_dir, f"{stem}__triplet.png")
                    vutils.save_image(triplet, out_path, normalize=False)

                    saved_vis += 1


    # aggregate per task (nan-safe, especially for HD)
    out_task = {k: _nanmean(torch.cat(v)).item() for k, v in result_task.items()}
    return out_task


# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device(args.device)

    # -----------------------------
    # Load checkpoint FIRST so eval-time MoE routing matches training.
    # Keeps your previous eval command unchanged.
    # -----------------------------
    if not os.path.isfile(args.model_weight):
        raise FileNotFoundError(f"model weight {args.model_weight} not found!")
    print(f"load model from {args.model_weight}")
    ckpt = torch.load(args.model_weight, map_location=device)

    # -----------------------------
    # Resolve MoE config (CLI > ckpt.moe_cfg > defaults)
    # -----------------------------
    moe_cfg = {}
    try:
        if isinstance(ckpt, dict):
            moe_cfg = ckpt.get("moe_cfg", {}) or {}
    except Exception:
        moe_cfg = {}

    if getattr(args, "moe_topk", None) is None:
        args.moe_topk = int(moe_cfg.get("moe_topk", 2))
    if getattr(args, "moe_temp", None) is None:
        args.moe_temp = float(moe_cfg.get("moe_temp", 1.0))
    if getattr(args, "moe_noise_std", None) is None:
        args.moe_noise_std = float(moe_cfg.get("moe_noise_std", 0.0))
    if getattr(args, "moe_style_bn", None) is None:
        args.moe_style_bn = int(moe_cfg.get("moe_style_bn", 1))
    if getattr(args, "moe_style_dropout", None) is None:
        args.moe_style_dropout = float(moe_cfg.get("moe_style_dropout", 0.10))
    if getattr(args, "moe_disable_style", None) is None:
        args.moe_disable_style = int(moe_cfg.get("moe_disable_style", 0))
    if getattr(args, "adapter_mode", None) is None:
        args.adapter_mode = str(moe_cfg.get("adapter_mode", "moe"))

    # -----------------------------
    # Resolve forced routing during eval (auto: ckpt first, then flag file)
    # -----------------------------
    weight_dir = os.path.dirname(args.model_weight)
    flag_path = os.path.join(weight_dir, "force_preassign.flag")
    flag_exists = os.path.exists(flag_path)

    force_ckpt_present = isinstance(ckpt, dict) and ("force_preassign" in ckpt)
    force_from_ckpt = bool(ckpt.get("force_preassign", False)) if isinstance(ckpt, dict) else False

    if int(getattr(args, "force_preassign_eval", -1)) in (0, 1):
        force_preassign_eval = bool(int(getattr(args, "force_preassign_eval", -1)))
    else:
        # AUTO: checkpoint wins; only fall back to flag when checkpoint has no force_preassign field
        force_preassign_eval = (force_from_ckpt if force_ckpt_present else flag_exists)

    setattr(args, "force_preassign_eval", int(force_preassign_eval))

    print(
        f"[Eval] MoE cfg: topk={int(args.moe_topk)}, temp={float(args.moe_temp)}, "
        f"noise_std={float(args.moe_noise_std)}, style_bn={int(args.moe_style_bn)}, "
        f"style_dropout={float(args.moe_style_dropout)}, disable_style={int(args.moe_disable_style)}, force_preassign_eval={int(force_preassign_eval)}"
    )
    if flag_exists and force_ckpt_present and (not force_from_ckpt):
        print("[Eval] Note: force_preassign.flag detected but checkpoint says force_preassign=False; ignoring flag (use --force_preassign_eval 1 to override).")
    if force_preassign_eval:
        print("[Eval] MoE forced routing ENABLED (few-shot).")

    # -----------------------------
    # Build model
    # -----------------------------
    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=checkpoint)

    method = str(getattr(args, "method", "infrasegfm")).lower().strip()
    use_decoder_tune = ("sam" in method) and ("moe" not in method) and ("seg" not in method)

    if use_decoder_tune:
        model = SAMDecoderTuner(sam_model).to(device)
        use_moe = False
    else:
        model = InfraSegFM(
            sam_model,
            args.bottleneck_dim,
            args.embedding_dim,
            args.expert_num,
            gate_topk=int(getattr(args, "moe_topk", 2)),
            gate_temperature=float(getattr(args, "moe_temp", 1.0)),
            gate_noise_std=float(getattr(args, "moe_noise_std", 0.0)),
            style_use_std=bool(int(getattr(args, "moe_style_bn", 1))),
            style_dropout=float(getattr(args, "moe_style_dropout", 0.10)),
            disable_style=bool(int(getattr(args, "moe_disable_style", 0))),
            adapter_mode=str(getattr(args, "adapter_mode", None) or "moe"),
        ).to(device)
        use_moe = True

    seg_metric = SegmentMetrics(args.metric).to(device)

    model = nn.DataParallel(model, device_ids=args.device_ids)
    seg_metric = nn.DataParallel(seg_metric, device_ids=args.device_ids)

    # Load weights AFTER model is built with correct MoE config.
    model.module.load_parameters(ckpt["model"])

    # Output dir
    work_dir = join(weight_dir, "internal")
    os.makedirs(work_dir, exist_ok=True)

    # Visualization root (saved together with MD/CSV under work_dir)
    vis_root = join(work_dir, "vis", "ID")
    if args.vis:
        os.makedirs(vis_root, exist_ok=True)
        print(f"[VIS] Saving triplets to: {vis_root}")

    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    result_total = {}
    meta = {"File": []}
    for m in args.metric:
        result_total[m] = []
        meta[m.upper()] = []

    time_start = time.time()

    out_md = join(work_dir, "ID.md")
    print(f"save evaluation result to {out_md}")

    data_path = os.path.normpath(args.data_path)

    with open(out_md, mode="w", encoding="utf-8") as f:
        f.write("# internal evaluation\n\n")

        # (A) Detect flat task layout
        children = [join(data_path, d) for d in sorted(os.listdir(data_path))
                    if os.path.isdir(join(data_path, d))]
        flat_tasks = [p for p in children if _is_task_folder(p)]

        if len(flat_tasks) > 0:
            dataset_name = "CivilDB"
            f.write(f"- {dataset_name}\n")

            # Build routing cache at split-root for flat layout when forced routing is enabled.
            # (0-shot keeps original gate; few-shot checkpoints enable forced routing automatically.)
            if getattr(args, "force_preassign_eval", 0) == 1:
                try:
                    cfg = PreassignConfig(
                        dataset_buckets=int(os.getenv("MOE_DATASET_BUCKETS", "1024")),
                        label_buckets=int(os.getenv("MOE_LABEL_BUCKETS", "256")),
                        include_style_features=True,
                        include_label_features=True,
                        num_workers=max(1, int(getattr(args, "num_workers", 2))),
                    )
                    train_cache = _find_train_moe_cache(getattr(args, "model_weight", ""))
                    if train_cache and os.path.exists(train_cache):
                        _cache = apply_routing_from_train_cache(train_cache, data_path, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                        print(f"[Eval] MoE cache ready (train->eval): {_cache}")
                    else:
                        _cache = build_routing_cache(data_path, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                        print(f"[Eval] MoE cache ready (fit eval): {_cache}")
                except Exception as _e:
                    print(f"[Eval] MoE cache build failed (continue): {_e}")

            for task_dir in flat_tasks:
                task = os.path.basename(os.path.normpath(task_dir))

                test_dataset = InfraTaskDataset(task_dir, train=False)
                if len(test_dataset) == 0:
                    continue

                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                )

                metric_task = evaluate(
                    model, seg_metric, test_dataloader,
                    img_size, img_transform, box_transform,
                    dataset_name, task,
                    result_total=result_total, meta=meta, args=args,
                    vis_root=vis_root if args.vis else None,
                )
                result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                f.write(f"  - {task}: {result_task}\n")

            # summarize (nan-safe)
            result_total_mean = {k: _nanmean(torch.cat(v)).item() for k, v in result_total.items()}
            result_total_str = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in result_total_mean.items()])
            f.write("- ALL\n")
            f.write(f"  - Mean: {result_total_str}\n")

        else:
            # (B) Original two-level dataset/task layout
            for dataset in sorted(os.listdir(data_path)):
                dataset_path = join(data_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue


                # Build routing cache for this dataset root if forced routing is enabled
                if getattr(args, "force_preassign_eval", 0) == 1:
                    try:
                        cfg = PreassignConfig(
                            dataset_buckets=int(os.getenv("MOE_DATASET_BUCKETS", "1024")),
                            label_buckets=int(os.getenv("MOE_LABEL_BUCKETS", "256")),
                            include_style_features=True,
                            include_label_features=True,
                            num_workers=max(1, int(getattr(args, "num_workers", 2))),
                        )
                        train_cache = _find_train_moe_cache(getattr(args, "model_weight", ""))
                        if train_cache and os.path.exists(train_cache):
                            _cache = apply_routing_from_train_cache(train_cache, dataset_path, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                            print(f"[Eval] MoE cache ready (train->eval): {_cache}")
                        else:
                            _cache = build_routing_cache(dataset_path, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                            print(f"[Eval] MoE cache ready (fit eval): {_cache}")
                    except Exception as _e:
                        print(f"[Eval] MoE cache build failed (continue): {_e}")
                f.write(f"- {dataset}\n")

                for task in sorted(os.listdir(dataset_path)):
                    task_path = join(dataset_path, task)
                    if not os.path.isdir(task_path):
                        continue

                    # Case 1: task folder contains npy_gts directly
                    if _is_task_folder(task_path):
                        test_dataset = InfraTaskDataset(task_path, train=False)
                        if len(test_dataset) == 0:
                            continue

                        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                        )

                        metric_task = evaluate(
                            model, seg_metric, test_dataloader,
                            img_size, img_transform, box_transform,
                            dataset, task,
                            result_total=result_total, meta=meta, args=args,
                            vis_root=vis_root if args.vis else None,
                        )
                        result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                        f.write(f"  - {task}: {result_task}\n")

                    # Case 2: task folder contains sequences
                    else:
                        f.write(f"  - {task}\n")
                        for sequence in sorted(os.listdir(task_path)):
                            sequence_path = join(task_path, sequence)

                            task_dir = _normalize_task_dir(sequence_path)
                            if not _is_task_folder(task_dir):
                                continue

                            test_dataset = InfraTaskDataset(task_dir, train=False, sequence=True)
                            if len(test_dataset) == 0:
                                continue

                            test_dataloader = DataLoader(
                                test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                            )

                            metric_task = evaluate(
                                model, seg_metric, test_dataloader,
                                img_size, img_transform, box_transform,
                                dataset, task, sequence,
                                result_total=result_total, meta=meta, args=args,
                                vis_root=vis_root if args.vis else None,
                            )
                            result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                            f.write(f"    - {sequence}: {result_task}\n")

            # summarize (nan-safe)
            result_total_mean = {k: _nanmean(torch.cat(v)).item() for k, v in result_total.items()}
            result_total_str = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in result_total_mean.items()])
            f.write("- ALL\n")
            f.write(f"  - Mean: {result_total_str}\n")

    time_end = time.time()
    print(f"Time cost: {time_end - time_start:.0f} s")

    # record instance-level results
    df = pd.DataFrame(meta)
    df.to_csv(join(work_dir, "ID.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved instance metrics to: {join(work_dir, 'ID.csv')}")

    if args.vis:
        print(f"[VIS] Triplets saved under: {vis_root}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# -*- coding: utf-8 -*-
from __future__ import annotations

"""
InfraSegFM external evaluating script (Windows compatible)

Supports BOTH:
A) Flat task layout:
   <data_path>/<shift_type>/
      TaskA/
        npy_gts/
        npy_imgs/
      TaskB/...

B) Original external layout:
   <data_path>/<shift_type>/
      Task/
        Dataset/
          inference/ (or sequences/*/inference/)

Enhancements kept:
1) VIS outputs are FLATTENED into <out_dir>/vis (no deep folders).
2) Empty-aware Dice to avoid 0/0 -> NaN.

Removed (per your request):
- --threshold (binarization threshold)
- --dsc_tol and tdsc (tolerant dice)
Threshold is now FIXED at 0.5 to match the original reference behavior.
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
import time

# Gentle mode: duty-cycle the GPU between steps to suppress power transients
# on marginal PSUs. Zero effect on results, only on wall time.
_STEP_SLEEP_MS = max(0, int(os.environ.get("INFRASEGFM_STEP_SLEEP_MS", "0") or 0))


def _step_throttle():
    if _STEP_SLEEP_MS:
        time.sleep(_STEP_SLEEP_MS / 1000.0)

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


parser = argparse.ArgumentParser("InfraSegFM external evaluating", add_help=True)

# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--model_weight", type=str, default="./playground/InfraSegFM/infrasegfm_vit_b.pth")
parser.add_argument("--method", type=str, default="infrasegfm")
parser.add_argument("--seed", type=int, default=None, help="Recorded seed; leave absent for an unknown historical seed")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--adapter_mode", type=str, default=None, choices=["moe", "shared", "lora", "medsa"],
                    help="Adapter architecture. If unset, read from checkpoint moe_cfg (defaults to 'moe').")

# MoE routing at evaluation time:
#   0  = disable (default, recommended for real OOD measurement)
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
parser.add_argument("--data_path", type=str, default="./playground/eval/OOD")
parser.add_argument("--shift_type", type=str, default="cross_site")
parser.add_argument("--metric", type=str, default=["dsc", "hd"], nargs='+',
                    help="metrics: dsc, hd")

# eval
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0], nargs='+')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1],
                    help="1 enables DataLoader pinned memory; 0 disables it for Windows stability.")

# output
parser.add_argument("--out_dir", type=str, default=None)

# visualization
parser.add_argument("--vis", action="store_true")
parser.add_argument("--vis_max", type=int, default=10_000_000)
parser.add_argument("--vis_stride", type=int, default=1)


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten()
    m = torch.isfinite(x)
    if m.any():
        return x[m].mean()
    return torch.tensor(float("nan"), device=x.device)


def dice_empty_aware(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred, gt: Bool tensors of shape [B, 1, H, W]
    Rule:
      - GT=0 & Pred=0 -> 1
      - GT=0 & Pred>0 -> 0
      - GT>0 -> standard dice
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


def _safe_stem_from_name(name: str, shift_root: str) -> str:
    """
    Create a reasonably unique filename stem from data["name"].
    Keep __-joined relpath to avoid collisions when flattening vis outputs.
    """
    try:
        p = os.path.normpath(str(name))
        rel = os.path.relpath(p, os.path.normpath(shift_root))
    except Exception:
        rel = str(name)

    rel = rel.replace("/", os.sep).replace("\\", os.sep)
    rel = rel.replace("npy_gts" + os.sep, "").replace("npy_imgs" + os.sep, "")
    rel = rel.replace("inference" + os.sep, "")
    rel = rel.replace(os.sep, "__")
    stem = os.path.splitext(rel)[0]

    for ch in ['<', '>', ':', '"', '|', '?', '*']:
        stem = stem.replace(ch, "_")
    return stem


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_")


def _is_task_folder(p: str) -> bool:
    return os.path.isdir(join(p, "npy_gts")) and os.path.isdir(join(p, "npy_imgs"))


def _find_train_moe_cache(model_weight: str) -> str | None:
    """Best-effort locate the training routing cache (for forced routing eval)."""
    try:
        w = os.path.normpath(model_weight)
        wdir = os.path.dirname(w)

        p_txt = os.path.join(wdir, "moe_train_cache_path.txt")
        if os.path.exists(p_txt):
            try:
                with open(p_txt, "r", encoding="utf-8") as f:
                    p = f.read().strip()
                if p and os.path.exists(p):
                    return os.path.normpath(p)
            except Exception:
                pass

        for cand in [
            os.path.join(".", "playground", "InfraSegDBv1", "train", "_moe_routing_cache.json"),
            os.path.join(".", "playground", "InfraSegDBv1", "val", "_moe_routing_cache.json"),
        ]:
            if os.path.exists(cand):
                return os.path.normpath(cand)

        cur = wdir
        for _ in range(4):
            cand = os.path.join(cur, "_moe_routing_cache.json")
            if os.path.exists(cand):
                return os.path.normpath(cand)
            cur = os.path.dirname(cur)
    except Exception:
        return None
    return None


def evaluate(
    model,
    metric,
    dataloader,
    img_size,
    img_transform,
    box_transform,
    shift_root: str,
    group1: str,
    group2: str,
    sequence: Optional[str] = None,
    result_total=None,
    meta=None,
    args=None,
    vis_root: Optional[str] = None,
):
    """
    group1/group2 are used for logging:
      - flat mode: group1="(flat)" group2=task
      - normal mode: group1=task group2=dataset
    """
    model.eval()
    device = torch.device(args.device)

    # FIXED threshold to match original repo
    THRESH = 0.5

    result_task: Dict[str, List[torch.Tensor]] = {m: [] for m in args.metric}

    saved_vis = 0
    seen_vis = 0

    pbar = tqdm(dataloader)
    desc = f"Evaluating - {group1} {group2}" if sequence is None else f"Evaluating - {group1} {group2} {sequence}"
    pbar.set_description(desc)

    # metric module is only used for non-DSC metrics like hd
    other_metrics = [m for m in args.metric if m != "dsc"]

    with torch.no_grad():
        for data, label in pbar:
            _step_throttle()
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    data["box"].reshape(-1, 2, 2),
                    data["img"].shape[-2:]
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)

            # Optional: enforce offline expert routing (few-shot eval)
            if getattr(args, "force_preassign_eval", 0) == 1 and ("moe_target" in data) and ("force_expert" not in data):
                data["force_expert"] = data["moe_target"]
            if "force_expert" in data:
                try:
                    data["force_expert"] = data["force_expert"].to(device, non_blocking=True, dtype=torch.long)
                except Exception:
                    pass
            label = label.to(device, non_blocking=True, dtype=torch.bool)  # [B,1,H,W]

            mask_pred = model(data)
            if mask_pred.shape[-2:] != label.shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)

            mask_prob = torch.sigmoid(mask_pred)
            mask = (mask_prob > THRESH).bool()  # [B,K,H,W]
            K = mask.shape[1]

            # per-mask metrics
            per_mask: Dict[str, List[torch.Tensor]] = {m: [] for m in args.metric}

            for k in range(K):
                pred_k = mask[:, k].unsqueeze(1)  # [B,1,H,W]

                # DSC (empty-aware)
                if "dsc" in args.metric:
                    per_mask["dsc"].append(dice_empty_aware(pred_k, label))

                # Other metrics (e.g., HD) via metric module
                if other_metrics:
                    out = metric(pred_k, label)  # returns dict for metrics
                    for m in other_metrics:
                        per_mask[m].append(out[m])

            # Select best mask by DSC (same spirit as original)
            if "dsc" not in args.metric:
                raise ValueError("This script expects 'dsc' in --metric for best-mask selection.")
            dsc_stack = torch.stack(per_mask["dsc"], dim=0)  # [K,B]
            _, best_idx = dsc_stack.max(dim=0)

            metric_dict = {}
            batch_metric_values: Dict[str, torch.Tensor] = {}

            for m in args.metric:
                m_stack = torch.stack(per_mask[m], dim=0)  # [K,B]
                result = m_stack[best_idx, torch.arange(m_stack.shape[1], device=device)]  # [B]

                batch_metric_values[m] = result
                result_task[m].append(result)
                result_total[m].append(result)
                metric_dict[m] = _nanmean(result).item()

            pbar.set_postfix(metric_dict)

            # record per-instance
            for i, name in enumerate(data["name"]):
                try:
                    file_rel = os.path.relpath(os.path.normpath(name), os.path.normpath(shift_root))
                except Exception:
                    file_rel = str(name)
                file_rel = file_rel.replace("npy_gts" + os.sep, "").replace("inference" + os.sep, "")
                meta["File"].append(file_rel)
                meta["Task"].append(group2 if group1 == "(flat)" else group1)
                meta["Seed"].append(args.seed)
                for m in args.metric:
                    meta[m.upper()].append(batch_metric_values[m][i].item())

            # VIS (flatten to vis_root only)
            if args.vis and vis_root is not None and saved_vis < args.vis_max:
                best_mask = mask[torch.arange(mask.size(0), device=device), best_idx].unsqueeze(1).float()
                gt = label.float()

                for i in range(label.size(0)):
                    if saved_vis >= args.vis_max:
                        break
                    if (seen_vis % max(1, args.vis_stride)) != 0:
                        seen_vis += 1
                        continue
                    seen_vis += 1

                    out_dir = vis_root
                    os.makedirs(out_dir, exist_ok=True)

                    stem = _safe_stem_from_name(data["name"][i], shift_root)

                    img = data["img"][i].detach().cpu().float()
                    if img.dim() == 2:
                        img = img.unsqueeze(0)
                    if img.size(0) == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.size(0) > 3:
                        img = img[:3]
                    img = img.clamp(0, 1)

                    gt3 = gt[i].detach().cpu().repeat(3, 1, 1).clamp(0, 1)
                    pr3 = best_mask[i].detach().cpu().repeat(3, 1, 1).clamp(0, 1)

                    triplet = torch.cat([img, gt3, pr3], dim=2)
                    vutils.save_image(triplet, join(out_dir, f"{stem}__triplet.png"), normalize=False)
                    saved_vis += 1

    out_task = {k: _nanmean(torch.cat(v)).item() for k, v in result_task.items()}
    return out_task


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
    # Output dir base
    # -----------------------------
    if args.out_dir is not None:
        base_dir = os.path.normpath(args.out_dir)
    else:
        base_dir = os.path.dirname(args.model_weight)

    # -----------------------------
    # Resolve forced routing during eval (auto: ckpt first, then flag file)
    # -----------------------------
    flag_path = os.path.join(base_dir, "force_preassign.flag")
    flag_exists = os.path.exists(flag_path)

    force_ckpt_present = isinstance(ckpt, dict) and ("force_preassign" in ckpt)
    force_from_ckpt = bool(ckpt.get("force_preassign", False)) if isinstance(ckpt, dict) else False

    if int(getattr(args, "force_preassign_eval", -1)) in (0, 1):
        force_preassign_eval = bool(int(getattr(args, "force_preassign_eval", -1)))
    else:
        # AUTO: checkpoint wins; only fall back to flag when checkpoint has no force_preassign field
        force_preassign_eval = (force_from_ckpt if force_ckpt_present else flag_exists)

    # AUTO_DISABLE_FORCE_ON_UNSEEN_SHIFTS:
    # For CrossTask / RealWorld shifts, tasks are often unseen. Forced routing can bias the evaluation.
    # Unless the user explicitly sets --force_preassign_eval=1, we disable forcing for these shifts.
    try:
        user_set = int(getattr(args, "force_preassign_eval", 0)) in (0, 1)
    except Exception:
        user_set = True
    if (not user_set):
        st = str(getattr(args, "shift_type", "")).lower()
        if ("cross_task" in st) or ("crosstask" in st) or ("realworld" in st) or ("real_world" in st):
            force_preassign_eval = False

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
            gate_topk=int(args.moe_topk),
            gate_temperature=float(args.moe_temp),
            gate_noise_std=float(args.moe_noise_std),
            style_use_std=bool(int(args.moe_style_bn)),
            style_dropout=float(args.moe_style_dropout),
            disable_style=bool(int(getattr(args, "moe_disable_style", 0))),
            adapter_mode=str(getattr(args, "adapter_mode", None) or "moe"),
        ).to(device)
        use_moe = True

    # metric module is only used for non-DSC metrics like hd
    other_metrics = [m for m in args.metric if m != "dsc"]
    seg_metric = None if len(other_metrics) == 0 else SegmentMetrics(other_metrics).to(device)

    model = nn.DataParallel(model, device_ids=args.device_ids)
    if seg_metric is not None:
        seg_metric = nn.DataParallel(seg_metric, device_ids=args.device_ids)

    # Load weights AFTER model is built with correct MoE config.
    model.module.load_parameters(ckpt["model"])

    # output dir
    work_dir = base_dir
    if force_preassign_eval:
        work_dir = join(work_dir, "external")
    os.makedirs(work_dir, exist_ok=True)

    vis_root = join(work_dir, "vis")
    if args.vis:
        os.makedirs(vis_root, exist_ok=True)
        print(f"[VIS] Saving triplets to: {os.path.normpath(vis_root)}")

    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    result_total = {m: [] for m in args.metric}
    meta = {"File": [], "Task": [], "Seed": []}
    for m in args.metric:
        meta[m.upper()] = []

    time_start = time.time()

    shift_root = join(os.path.normpath(args.data_path), args.shift_type)
    if not os.path.isdir(shift_root):
        raise FileNotFoundError(f"Shift folder not found: {shift_root}")

    safe_shift = _safe_name(args.shift_type)
    safe_datap = os.path.basename(os.path.normpath(args.data_path))

    out_md = join(work_dir, f"{safe_datap}-{safe_shift}.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    print(f"save evaluation result to {os.path.normpath(out_md)}")

    # Detect flat tasks
    children = [join(shift_root, d) for d in sorted(os.listdir(shift_root)) if os.path.isdir(join(shift_root, d))]
    flat_tasks = [p for p in children if _is_task_folder(p)]

    with open(out_md, mode="w", encoding="utf-8") as f:
        f.write(f"# external {args.shift_type} evaluation\n\n")

        total_eval_cnt = 0

        if len(flat_tasks) > 0:
            f.write(f"- FLAT_TASKS\n")

            # Build routing cache at shift-root for flat layout when forced routing is enabled.
            # This makes external ODD evaluation consistent with few-shot training that uses moe_target.
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
                        _cache = apply_routing_from_train_cache(train_cache, shift_root, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                        print(f"[Eval] MoE cache ready (train->eval): {os.path.normpath(_cache)}")
                    else:
                        _cache = build_routing_cache(shift_root, int(getattr(args, "expert_num", 4)), cfg=cfg, force_rebuild=False)
                        print(f"[Eval] MoE cache ready (fit eval): {os.path.normpath(_cache)}")
                except Exception as _e:
                    print(f"[Eval] MoE cache build failed (continue): {_e}")

            for task_dir in flat_tasks:
                task = os.path.basename(os.path.normpath(task_dir))

                # IMPORTANT: skip empty-shell tasks safely
                try:
                    test_dataset = InfraTaskDataset(task_dir, train=False)
                except ValueError as e:
                    print(f"[SKIP] {task_dir} -> {e}")
                    continue

                if len(test_dataset) == 0:
                    continue

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                )

                metric_task = evaluate(
                    model, seg_metric, test_loader,
                    img_size, img_transform, box_transform,
                    shift_root, "(flat)", task,
                    result_total=result_total, meta=meta, args=args,
                    vis_root=vis_root if args.vis else None
                )
                total_eval_cnt += len(test_dataset)
                result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                f.write(f"  - {task}: {result_task}\n")

        else:
            # Original layout
            for task in sorted(os.listdir(shift_root)):
                task_path = join(shift_root, task)
                if not os.path.isdir(task_path):
                    continue
                f.write(f"- {task}\n")

                for dataset in sorted(os.listdir(task_path)):
                    dataset_path = join(task_path, dataset)
                    if not os.path.isdir(dataset_path):
                        continue

                    # has inference
                    if "inference" in os.listdir(dataset_path):
                        infer_path = join(dataset_path, "inference")
                        try:
                            test_dataset = InfraTaskDataset(infer_path, train=False)
                        except ValueError as e:
                            print(f"[SKIP] {infer_path} -> {e}")
                            continue

                        if len(test_dataset) == 0:
                            continue

                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                        )

                        metric_task = evaluate(
                            model, seg_metric, test_loader,
                            img_size, img_transform, box_transform,
                            shift_root, task, dataset,
                            result_total=result_total, meta=meta, args=args,
                            vis_root=vis_root if args.vis else None
                        )
                        total_eval_cnt += len(test_dataset)
                        result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                        f.write(f"  - {dataset}: {result_task}\n")

                    else:
                        f.write(f"  - {dataset}\n")
                        for sequence in sorted(os.listdir(dataset_path)):
                            sequence_path = join(dataset_path, sequence)
                            infer_path = join(sequence_path, "inference")
                            if not os.path.isdir(infer_path):
                                continue

                            try:
                                test_dataset = InfraTaskDataset(infer_path, train=False, sequence=True)
                            except ValueError as e:
                                print(f"[SKIP] {infer_path} -> {e}")
                                continue

                            if len(test_dataset) == 0:
                                continue

                            test_loader = DataLoader(
                                test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0,
                            )

                            metric_task = evaluate(
                                model, seg_metric, test_loader,
                                img_size, img_transform, box_transform,
                                shift_root, task, dataset, sequence,
                                result_total=result_total, meta=meta, args=args,
                                vis_root=vis_root if args.vis else None
                            )
                            total_eval_cnt += len(test_dataset)
                            result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                            f.write(f"    - {sequence}: {result_task}\n")

        # Summarize (avoid empty cat crash)
        any_added = any(len(v) > 0 for v in result_total.values())
        f.write(f"\n- ALL\n")
        if not any_added:
            f.write("  - Mean: N/A (no samples evaluated)\n")
            print("[ERROR] No samples were evaluated. Please check directory layout and InfraTaskDataset loading rules.")
        else:
            result_total_mean = {k: _nanmean(torch.cat(v)).item() for k, v in result_total.items()}
            result_total_str = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in result_total_mean.items()])
            f.write(f"  - Mean: {result_total_str}\n")

        f.write(f"\n- META\n  - evaluated_samples: {total_eval_cnt}\n")

    # save csv
    out_csv = join(work_dir, f"{safe_datap}-{safe_shift}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(meta).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved instance metrics to: {os.path.normpath(out_csv)}")

    time_end = time.time()
    print(f"Time cost: {time_end - time_start:.0f} s")

    if args.vis:
        print(f"[VIS] Triplets saved under: {os.path.normpath(vis_root)}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

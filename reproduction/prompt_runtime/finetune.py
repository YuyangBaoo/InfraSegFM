# -*- coding: utf-8 -*-
"""
Fine-tuning script (flat train/val layout, Windows compatible)

Data layout expected (matches your screenshot):
  <data_path>/
    train/        (or train5/train15/train25/train50/train100)
    val/

This script:
- loads SAM + InfraSegFM or decoder-tuning baseline
- optionally resumes from --resume (model_best.pth etc.)
- trains on <data_path>/<train_split>, validates on <data_path>/<val_split>
- saves outputs under: <work_dir>/<task_name>/
  - output.log
  - tensorboard logs
  - model_latest.pth / model_best.pth / model_lowest.pth / model_epochXX.pth
  - train_loss.png / lr.png / val_dsc.png
  - result.log (per-epoch summary)

Notes:
- If your images are 512x512 but SAM image_size=256, inputs are resized to 256 for training.
"""

import os
import re
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

import argparse
import json
import time

# Gentle mode: duty-cycle the GPU between steps to suppress power transients
# on marginal PSUs. Zero effect on results, only on wall time.
_STEP_SLEEP_MS = max(0, int(os.environ.get("INFRASEGFM_STEP_SLEEP_MS", "0") or 0))


def _step_throttle():
    if _STEP_SLEEP_MS:
        time.sleep(_STEP_SLEEP_MS / 1000.0)

import faulthandler
import traceback
from datetime import datetime, timedelta

import numpy as np
import random
import torch

faulthandler.enable(all_threads=True)
_FAULT_LOG_HANDLE = None


def enable_fault_log(log_dir: str):
    global _FAULT_LOG_HANDLE
    try:
        os.makedirs(log_dir, exist_ok=True)
        _FAULT_LOG_HANDLE = open(os.path.join(log_dir, "faulthandler.log"), "a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)
    except Exception:
        faulthandler.enable(all_threads=True)


def log_unhandled_exception(args, exc: BaseException) -> None:
    try:
        out_dir = os.path.join(args.work_dir, args.task_name)
        os.makedirs(out_dir, exist_ok=True)
        tb = traceback.format_exc()
        with open(os.path.join(out_dir, "crash.log"), "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"exception: {type(exc).__name__}: {exc}\n")
            f.write(tb)
            if not tb.endswith("\n"):
                f.write("\n")
        try:
            from utils.runtime_logging import write_heartbeat
            write_heartbeat(
                out_dir,
                "finetune.py",
                "unhandled_exception",
                error=f"{type(exc).__name__}: {exc}",
                traceback=tb[-4000:],
            )
        except Exception:
            pass
    except Exception:
        pass


def make_amp_grad_scaler():
    if hasattr(torch, "GradScaler"):
        try:
            return torch.GradScaler("cuda")
        except TypeError:
            return torch.GradScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def cuda_amp_autocast():
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except TypeError:
            return torch.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


# -------------------------
# Stable Dice (empty-aware)
# -------------------------
def dice_batch(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_bin: (B,1,H,W) 0/1 float
    gt_bin:   (B,1,H,W) 0/1 float
    returns:  (B,) dice
    Rules:
      - pred and gt both empty => dice=1
      - otherwise standard dice (+eps)
    """
    pred = pred_bin.float()
    gt = gt_bin.float()

    pred_f = pred.flatten(1)
    gt_f = gt.flatten(1)

    inter = (pred_f * gt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + gt_f.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)
    dice = torch.where(denom == 0, torch.ones_like(dice), dice)
    return dice


def set_seeds(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==========================
# ✅ Optimizer helper: smaller LR for mask_decoder + grad clip + nonfinite guard
# ==========================
def build_optimizer(model_dp: torch.nn.Module, args):
    mask_params = []
    other_params = []
    for n, p in model_dp.named_parameters():
        if not p.requires_grad:
            continue
        if "sam.mask_decoder" in n or ("mask_decoder" in n and "sam." in n):
            mask_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": float(args.lr)})
    if mask_params:
        groups.append({"params": mask_params, "lr": float(args.lr) * float(getattr(args, "mask_decoder_lr_mult", 0.1))})

    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))



# -------------------------
# Arg parser
# -------------------------
parser = argparse.ArgumentParser("InfraSegFM fine-tuning (flat train/val)", add_help=True)

# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--method", type=str, default="infrasegfm")
parser.add_argument("--seed", type=int, default=2025, help="Independent fine-tuning random seed")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--task_drop_p", type=float, default=0.0,
                    help="Training-time task-identity dropout (see pretrain.py). Usually 0 for few-shot finetuning.")
parser.add_argument("--adapter_mode", type=str, default="moe", choices=["moe", "shared", "lora", "medsa"],
                    help="'moe' = routed experts (default); 'shared' = parameter-matched single shared adapter control (no gate, no style, no routing); 'lora' = SAMed-style LoRA (r=4 on q/v); 'medsa' = Med-SA serial adapters.")
# MoE anti-collapse (gate regularization)
parser.add_argument("--moe_topk", type=int, default=2,
                    help="top-k experts used by the gate (0 = dense softmax, original behavior)")
parser.add_argument("--moe_temp", type=float, default=1.0,
                    help="gate softmax temperature (smaller -> peakier; >1 -> smoother)")
parser.add_argument("--moe_noise_std", type=float, default=0.0,
                    help="std of Gaussian noise added to gate logits during training (0 = off)")
parser.add_argument("--moe_lb_coef", type=float, default=0.01,
                    help="load-balance loss coefficient (0 = disable)")
parser.add_argument("--moe_ent_coef", type=float, default=0.01,
                    help="entropy bonus coefficient (0 = disable)")
parser.add_argument("--moe_style_bn", type=int, default=1,
                    help="1: use std in online style descriptor; 0: use mean absolute deviation")
parser.add_argument("--moe_style_dropout", type=float, default=0.10,
                    help="dropout applied to online style embedding")
parser.add_argument("--moe_disable_style", type=int, default=0, choices=[0, 1],
                    help="Ablation: zero out online style embedding before routing (NoStyle).")

# routing pre-assign (task-first, dynamic-K)
parser.add_argument("--moe_preassign", type=int, default=1, choices=[0,1],
                    help="Pre-assign a routing target for each training sample using task-first dynamic-K clustering (writes cache under <data_path>/train). (default: 1)")
parser.add_argument("--moe_preassign_force", action="store_true",
                    help="Force rebuild the routing cache even if it already exists.")
parser.add_argument("--moe_route_sup_coef", type=float, default=0.05,
                    help="Routing supervision loss coefficient (0 = disable). Uses 'moe_target' from dataset cache.")
parser.add_argument("--moe_route_warmup", type=int, default=2,
                    help="Warmup epochs for routing supervision (linearly ramps from 0 to moe_route_sup_coef).")
parser.add_argument("--moe_route_cap_ratio", type=float, default=0.25,
                    help="Clip routing supervision term to at most (moe_route_cap_ratio * seg_loss) to avoid destabilizing training when coef is large.")
parser.add_argument("--moe_route_log", type=int, default=1, choices=[0,1],
                    help="Log routing supervision stats (route_loss/route_coef/route_term and invalid targets) to tqdm and TensorBoard. (default: 1)")

# data imbalance
parser.add_argument("--task_balanced_sampling", type=int, default=1,
                    help="1: balance sampling across TaskFolders (recommended); 0: plain shuffle")

# TaskFolder imbalance handling (mixed sampling)
parser.add_argument("--task_sampling_alpha", type=float, default=0.25,
                    help="mixing factor in [0,1] for TaskFolder reweighting. 0=natural, 1=fully balanced.")
parser.add_argument("--task_sampling_power", type=float, default=1.0,
                    help="strength of inverse-frequency reweighting: 1/(count**power). 1.0=standard, 0.5=milder.")
parser.add_argument("--task_sampling_num_samples", type=int, default=0,
                    help="samples drawn per epoch when using WeightedRandomSampler (0 => len(dataset)).")

# MoE warmup (recommended)
parser.add_argument("--moe_warmup_epochs", type=int, default=5,
                    help="linearly warm up MoE auxiliary loss + gate noise for first N epochs (0 disables warmup).")

# data
parser.add_argument("--data_path", type=str, default="./playground/InfraSegDBCrossSite",
                    help="root folder that contains train/val (or train5/train15/...)")
parser.add_argument("--train_split", type=str, default="train",
                    help="folder name for training split (e.g train, train5, train15, train25, train50, train100)")
parser.add_argument("--val_split", type=str, default="val",
                    help="folder name for validation split (default: val)")
parser.add_argument("--exclude_source_keywords", type=str, nargs="*", default=[],
                    help="Ablation/data filter: exclude paired .npy samples whose image or mask path contains any keyword, e.g. SynBridgeCrack.")
parser.add_argument("--moe_cache_name", type=str, default="_moe_routing_cache.json",
                    help="MoE routing-cache filename. Use variant-specific names for ablations to avoid stale cache reuse.")

# env / output
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./playground",
                    help="output root directory")
parser.add_argument("--task_name", type=str, default="InfraSegFM_finetune",
                    help="output folder name under work_dir")

# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1],
                    help="1 enables DataLoader pinned memory; 0 disables it for Windows stability.")
parser.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint (model_best.pth etc.) to initialize fine-tuning")

# optimizer
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mask_decoder_lr_mult", type=float, default=0.1,
                    help="learning-rate multiplier for SAM mask_decoder params (default: 0.1)")
parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="global grad-norm clip (0 disables). Recommended: 1.0")
parser.add_argument("--skip_nonfinite", type=int, default=1, choices=[0,1],
                    help="skip optimizer step if total_loss is nan/inf (default: 1)")
parser.add_argument("--auto_resume", type=int, default=1, choices=[0,1],
                    help="auto-resume from <work_dir>/<task_name>/model_latest.pth if present and --resume is not set (default: 1)")
parser.add_argument("--save_every", type=int, default=0,
                    help="save numbered epoch checkpoints every N epochs; 0 saves only the final numbered checkpoint. model_latest/model_best are still maintained.")
parser.add_argument("--selected_only", action="store_true", help="Keep only validation-selected model weights; optimizer resume is disabled")
parser.add_argument("--heartbeat_interval", type=int, default=100,
                    help="write training heartbeat JSON every N batches; 0 disables batch heartbeats.")
parser.add_argument("--disable_tqdm", type=int, default=int(os.environ.get("INFRASEGFM_DISABLE_TQDM", "0") or 0),
                    choices=[0, 1],
                    help="disable tqdm progress bars for cleaner logged runs. Heartbeat status is still written.")
parser.add_argument("--use_amp", action="store_true", default=False,
                    help="whether to use AMP")


def main(args):
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.transforms import Resize
    from tqdm import tqdm

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from segment_anything import sam_model_registry, sam_model_checkpoint
    from segment_anything.utils.transforms import ResizeLongestSide

    from model import SAMDecoderTuner, InfraSegFM
    from data.dataset import InfraSegDataset
    from utils.loss import DiceBCELoss
    from utils.logger import get_logger
    from utils.moe_loss import moe_aux_loss
    from utils.moe_preassign import build_routing_cache, PreassignConfig
    from utils.runtime_logging import atomic_torch_save as legacy_atomic_torch_save, write_heartbeat
    from utils.sampler import build_task_mixed_sampler

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from baseline.checkpoint_io import atomic_torch_save as selected_atomic_torch_save, selected_payload
    if args.selected_only:
        args.auto_resume = 0

    def atomic_torch_save(torch_module, state, path):
        if args.selected_only:
            return selected_atomic_torch_save(torch_module, selected_payload(state), path)
        return legacy_atomic_torch_save(torch_module, state, path)

    set_seeds(args.seed)

    device = torch.device(args.device)

    # -------------------------
    # Build model
    # -------------------------
    ckpt_path = os.path.join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=ckpt_path)

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
            style_use_std=bool(getattr(args, "moe_style_bn", 1)),
            style_dropout=float(getattr(args, "moe_style_dropout", 0.10)),
            disable_style=bool(int(getattr(args, "moe_disable_style", 0))),
            adapter_mode=str(getattr(args, "adapter_mode", None) or "moe"),
            task_drop_p=float(getattr(args, "task_drop_p", 0.0)),
        ).to(device)
        use_moe = True

    model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # -------------------------
    # Output dirs / logging
    # -------------------------
    out_dir = os.path.join(args.work_dir, args.task_name)
    os.makedirs(out_dir, exist_ok=True)
    enable_fault_log(out_dir)

    logger = get_logger(log_file=os.path.join(out_dir, "output.log"))
    logger.info(f"args: {json.dumps(vars(args), indent=2, ensure_ascii=False)}")
    logger.info("Model: %s" % str(model))
    logger.info("Number of total parameters: %d" % (sum(p.numel() for p in model.parameters())))
    logger.info("Number of trainable parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    tb_writer = SummaryWriter(log_dir=out_dir)
    heartbeat_interval = max(0, int(getattr(args, "heartbeat_interval", 100) or 0))
    disable_tqdm = bool(int(getattr(args, "disable_tqdm", 0) or 0))
    tqdm_mininterval = float(os.environ.get("INFRASEGFM_TQDM_MININTERVAL", "30") or 30)
    write_heartbeat(out_dir, "finetune.py", "init", torch_module=torch, args=vars(args))

    # -------------------------
    # Data
    # -------------------------
    train_root = os.path.join(args.data_path, args.train_split)
    val_root = os.path.join(args.data_path, args.val_split)

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train split not found: {train_root}")
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"Val split not found: {val_root}")

    # ==========================
    # Few-shot routing mode (auto-detected)
    # If your train split is train5/train15/train25/train50/train100, we:
    #   - force the MoE gate to follow offline pre-assigned experts (moe_target)
    #   - drop a flag file next to checkpoints so evaluation can auto-enable the same routing
    # 0-shot / full train uses the original gate (no forcing) by default.
    # ==========================
    _train_base = os.path.basename(os.path.normpath(train_root)).lower()
    is_fewshot = bool(re.match(r"train(5|15|25|50|100)$", _train_base))

    # ------------------------------------------------------------
    # Few-shot MoE forcing schedule
    # ------------------------------------------------------------
    # New default: only force routing during the first `moe_warmup_epochs` epochs,
    # then let the gate learn from segmentation loss (real inference behavior).
    #
    # Set --moe_warmup_epochs 0 to disable forcing completely.
    # ------------------------------------------------------------
    force_preassign_train = bool(is_fewshot)
    force_preassign_epochs = int(getattr(args, "moe_warmup_epochs", 0) or 0)
    if force_preassign_epochs < 0:
        force_preassign_epochs = 0

    # For backward compatibility: this field used to indicate "eval should force".
    # We now keep eval in gate mode by default (external eval defaults force_preassign_eval=0).
    force_preassign = False

    if force_preassign_train and (force_preassign_epochs > 0):
        try:
            with open(os.path.join(out_dir, "force_preassign.flag"), "w", encoding="utf-8") as _f:
                _f.write("1\n")
        except Exception:
            pass

    
    # ==========================
    # ✅ MoE routing pre-assign (task-first, dynamic-K)
    # This writes <data_path>/train/_moe_routing_cache.json and dataset.py will attach `moe_target`.
    # ==========================
    if int(getattr(args, "moe_preassign", 1)) == 1 and use_moe and str(getattr(args, "adapter_mode", None) or "moe") == "moe":
        try:
            cache_name = str(getattr(args, "moe_cache_name", "_moe_routing_cache.json"))
            cache_path_expected = os.path.join(train_root, cache_name)
            logger.info(f"MoE preassign start: cache={cache_path_expected}, exists={os.path.exists(cache_path_expected)}")
            write_heartbeat(
                out_dir,
                "finetune.py",
                "moe_preassign_start",
                torch_module=torch,
                cache_path=cache_path_expected,
                cache_exists=os.path.exists(cache_path_expected),
                force_rebuild=bool(args.moe_preassign_force),
            )
            cfg = PreassignConfig(
                include_label_features=True,   # label-aware pre-assign (offline)
                num_workers=max(1, int(getattr(args, "num_workers", 2))),
                exclude_source_keywords=tuple(getattr(args, "exclude_source_keywords", []) or ()),
                cache_name=str(getattr(args, "moe_cache_name", "_moe_routing_cache.json")),
            )
            cache_path = build_routing_cache(train_root, args.expert_num, cfg=cfg, force_rebuild=bool(args.moe_preassign_force))
            logger.info(f"MoE preassign cache ready: {cache_path}")
            write_heartbeat(
                out_dir,
                "finetune.py",
                "moe_preassign_done",
                torch_module=torch,
                cache_path=cache_path,
                cache_size=os.path.getsize(cache_path) if os.path.exists(cache_path) else None,
            )
            # Pointer file for evaluation (apply training routing to ID/ODD) without changing CLI.
            try:
                with open(os.path.join(out_dir, "moe_train_cache_path.txt"), "w", encoding="utf-8") as wf:
                    wf.write(os.path.normpath(cache_path))
            except Exception:
                pass
            if force_preassign_train and (force_preassign_epochs > 0):
                write_heartbeat(
                    out_dir,
                    "finetune.py",
                    "moe_preassign_val_start",
                    torch_module=torch,
                    cache_path=os.path.join(val_root, cache_name),
                    cache_exists=os.path.exists(os.path.join(val_root, cache_name)),
                    force_rebuild=bool(args.moe_preassign_force),
                )
                cache_val_path = build_routing_cache(val_root, args.expert_num, cfg=cfg, force_rebuild=bool(args.moe_preassign_force))
                logger.info(f"MoE preassign cache (val) ready: {cache_val_path}")
                write_heartbeat(
                    out_dir,
                    "finetune.py",
                    "moe_preassign_val_done",
                    torch_module=torch,
                    cache_path=cache_val_path,
                    cache_size=os.path.getsize(cache_val_path) if os.path.exists(cache_val_path) else None,
                )
        except Exception as e:
            logger.warning(f"MoE preassign failed (continue without it): {e}")
    train_dataset = InfraSegDataset(
        train_root,
        train=True,
        exclude_source_keywords=getattr(args, "exclude_source_keywords", []),
        moe_cache_name=getattr(args, "moe_cache_name", "_moe_routing_cache.json"),
    )
    val_dataset = InfraSegDataset(
        val_root,
        train=False,
        exclude_source_keywords=getattr(args, "exclude_source_keywords", []),
        moe_cache_name=getattr(args, "moe_cache_name", "_moe_routing_cache.json"),
    )

    logger.info(f"Train split: {train_root} | samples: {len(train_dataset)}")
    logger.info(f"Val split:   {val_root} | samples: {len(val_dataset)}")

    pin_memory = bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0
    if args.task_balanced_sampling == 1:
        sampler = build_task_mixed_sampler(train_dataset.task_folders, alpha=args.task_sampling_alpha, power=args.task_sampling_power, num_samples=(args.task_sampling_num_samples or None))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -------------------------
    # Optim / loss
    # -------------------------
    optimizer = build_optimizer(model, args)
    criterion = DiceBCELoss(sigmoid=True, squared_pred=True, reduction="none")

    # AMP
    scaler = make_amp_grad_scaler() if args.use_amp else None

    # Resize helpers
    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0.0
    best_epoch = -1

    loss_log = []
    lr_log = []
    dsc_log = []

    loaded_latest = False
    latest = os.path.join(out_dir, 'model_latest.pth')
    if int(getattr(args, 'auto_resume', 1)) == 1 and os.path.isfile(latest):
        try:
            logger.info(f'Auto-resume checkpoint found: {latest}')
            ckpt = torch.load(latest, map_location=device)
            if 'model' in ckpt:
                model.module.load_parameters(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if args.use_amp and (scaler is not None) and ('scaler' in ckpt):
                scaler.load_state_dict(ckpt['scaler'])
            resume_phase = str(ckpt.get("phase", "epoch_complete"))
            resume_epoch = int(ckpt.get('epoch', -1))
            start_epoch = resume_epoch + (1 if resume_phase == "epoch_complete" else 0)
            best_loss = float(ckpt.get("best_loss", best_loss))
            best_dsc = float(ckpt.get("best_dsc", best_dsc))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            loss_log = list(ckpt.get("loss_log", loss_log) or loss_log)
            lr_log = list(ckpt.get("lr_log", lr_log) or lr_log)
            dsc_log = list(ckpt.get("dsc_log", dsc_log) or dsc_log)
            loaded_latest = True
            logger.info(f'Auto-resume done. phase={resume_phase}, start_epoch={start_epoch}')
            write_heartbeat(
                out_dir,
                "finetune.py",
                "resume_loaded",
                torch_module=torch,
                resume_path=latest,
                resume_epoch=resume_epoch,
                resume_phase=resume_phase,
                start_epoch=start_epoch,
                best_epoch=best_epoch,
                best_dsc=best_dsc,
            )
        except Exception as e:
            logger.warning(f'Auto-resume failed, falling back to --resume/base initialization. err={e}')

    if (not loaded_latest) and args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        logger.info(f"Loading resume checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # Compatible with your training/eval checkpoints:
        # ckpt["model"] is expected by model.module.load_parameters(...)
        model.module.load_parameters(ckpt["model"])
        # If it has optimizer, you can choose to load or not; for fine-tune, usually start fresh.
        # If you want to load optimizer too, uncomment:
        # if "optimizer" in ckpt:
        #     optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = 0  # fine-tune usually restarts epoch counting
        logger.info("Resume loaded (model parameters). Optimizer is freshly initialized for fine-tuning.")

    if loaded_latest and start_epoch >= args.num_epochs and 'ckpt' in locals() and isinstance(ckpt, dict):
        final_marker = os.path.join(out_dir, f"model_{max(args.num_epochs - 1, 0)}.pth")
        if not os.path.isfile(final_marker):
            ckpt["phase"] = "epoch_complete"
            atomic_torch_save(torch, ckpt, final_marker)
            logger.info(f"Final epoch marker recreated from model_latest: {final_marker}")
            write_heartbeat(
                out_dir,
                "finetune.py",
                "final_marker_recreated",
                torch_module=torch,
                path=final_marker,
                start_epoch=start_epoch,
                num_epochs=args.num_epochs,
            )

    # -------------------------
    # Train loop
    # -------------------------
    time_start = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        write_heartbeat(
            out_dir,
            "finetune.py",
            "epoch_start",
            torch_module=torch,
            epoch=epoch,
            num_epochs=args.num_epochs,
        )
        force_now = bool(force_preassign_train and (force_preassign_epochs > 0) and (epoch < force_preassign_epochs))
        # -------- Train --------
        model.train()
        epoch_loss = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.num_epochs}] Train", disable=disable_tqdm, mininterval=tqdm_mininterval)
        for data, label in pbar:
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # resize to SAM encoder size
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    data["box"].reshape(-1, 2, 2),
                    data["img"].shape[-2:]
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            # Optional: enforce offline expert routing (few-shot)
            if force_now and ("moe_target" in data) and ("force_expert" not in data):
                data["force_expert"] = data["moe_target"]
            if "force_expert" in data:
                try:
                    data["force_expert"] = data["force_expert"].to(device, non_blocking=True, dtype=torch.long)
                except Exception:
                    pass
            label = label.to(device, non_blocking=True)

            # Forward (MoE variant can optionally return gates)
            if use_moe:
                if args.use_amp:
                    with cuda_amp_autocast():
                        out = model(data, return_gates=True)
                else:
                    out = model(data, return_gates=True)
                mask_pred = out["masks"]
                gates = out.get("gates", [])
            else:
                if args.use_amp:
                    with cuda_amp_autocast():
                        mask_pred = model(data)
                else:
                    mask_pred = model(data)
                gates = []

            # ensure (H,W) match
            if mask_pred.shape[-2:] != label.shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)

            # multi-mask: take minimum loss across masks
            losses = []
            for i in range(model.module.sam.mask_decoder.num_multimask_outputs):
                output = mask_pred[:, i].unsqueeze(1)
                loss_i = criterion(output.float(), label)
                losses.append(loss_i)
            loss = torch.stack(losses, dim=0).min(dim=0)[0].mean()

            moe_out = moe_aux_loss(
                gates,
                lb_coef=args.moe_lb_coef,
                ent_coef=args.moe_ent_coef,
            )

            # ==========================
            # ✅ Routing supervision (optional, uses dataset-provided moe_target)
            # Encourage gates to match the pre-assigned expert targets.
            # Notes:
            #   - With top-k gating, some experts can have near-zero prob; we clamp for stability.
            #   - If moe_route_sup_coef is set too large, route loss can dominate and hurt DSC.
            #     We therefore (optionally) cap the route term to a fraction of seg loss.
            # ==========================
            route_loss = None
            route_term = None
            invalid_tgt = 0

            if force_now and args.moe_route_sup_coef > 0 and gates and isinstance(data, dict) and ("moe_target" in data):
                try:
                    tgt = data["moe_target"].to(device, non_blocking=True).long()  # [B]
                    E = int(gates[0].shape[1])
                    invalid_tgt = int(((tgt < 0) | (tgt >= E)).sum().item())
                    tgt = tgt.clamp_(0, E - 1)

                    nlls = []
                    for g in gates:
                        if g is None:
                            continue
                        g_clamp = torch.clamp(g, min=1e-8)
                        p = g_clamp.gather(1, tgt.view(-1, 1)).squeeze(1)  # [B]
                        nlls.append((-torch.log(p)).mean())
                    if nlls:
                        route_loss = torch.stack(nlls).mean()  # scalar tensor
                except Exception:
                    route_loss = None
                    invalid_tgt = 0

            # warmup schedule (epoch is 0-indexed)
            if args.moe_route_sup_coef > 0 and args.moe_route_warmup > 0:
                w = min(1.0, float(epoch) / float(max(1, args.moe_route_warmup)))
            else:
                w = 1.0
            route_coef = float(args.moe_route_sup_coef) * float(w)

            # route term with optional cap
            if (route_loss is not None) and (route_coef > 0):
                route_term = route_loss * float(route_coef)
                cap = float(getattr(args, "moe_route_cap_ratio", 0.25)) * loss.detach()
                route_term = torch.minimum(route_term, cap)
            else:
                route_term = 0.0

            total_loss = loss + moe_out.total.to(loss.device) + (route_term if isinstance(route_term, torch.Tensor) else 0.0)

            # Non-finite loss guard (prevents NaN from corrupting optimizer states)
            if int(getattr(args, 'skip_nonfinite', 1)) == 1 and (not torch.isfinite(total_loss)):
                logger.warning(f'Non-finite loss encountered at epoch={epoch}, step={step}: {float(total_loss.detach().item())}')
                optimizer.zero_grad(set_to_none=True)
                if args.use_amp and (scaler is not None):
                    scaler.update()
                continue

            if args.use_amp:
                scaler.scale(total_loss).backward()
                # Grad clip must happen on unscaled grads
                scaler.unscale_(optimizer)
                if float(getattr(args, 'grad_clip_norm', 1.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, 'grad_clip_norm', 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if float(getattr(args, 'grad_clip_norm', 1.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, 'grad_clip_norm', 1.0)))
                optimizer.step()

            _step_throttle()
            epoch_loss += float(loss.item())
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            pbar.set_postfix({
            "lr": lr,
            "loss": float(loss.item()),
            "moe": float(moe_out.total.detach().item()) if hasattr(moe_out.total, "item") else 0.0,
            "route": float(route_term.detach().item()) if hasattr(route_term, "detach") else float(route_term),
            })

            # tensorboard (batch)
            epoch_1000x = int((epoch + step / max(1, len(train_loader))) * 1000)
            tb_writer.add_scalar("batch/lr", lr, epoch_1000x)
            tb_writer.add_scalar("batch/loss", float(loss.item()), epoch_1000x)
            tb_writer.add_scalar("batch/moe_total", float(moe_out.total.detach().item()), epoch_1000x)
            tb_writer.add_scalar("batch/moe_lb", float(moe_out.lb.detach().item()), epoch_1000x)
            tb_writer.add_scalar("batch/moe_ent", float(moe_out.ent.detach().item()), epoch_1000x)

            if int(getattr(args, 'moe_route_log', 1)) == 1:
                try:
                    tb_writer.add_scalar('batch/route_term', float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term), epoch_1000x)
                    tb_writer.add_scalar('batch/route_loss', float(route_loss.detach().item()) if hasattr(route_loss, 'detach') else 0.0, epoch_1000x)
                    tb_writer.add_scalar('batch/route_coef', float(route_coef), epoch_1000x)
                    tb_writer.add_scalar('batch/route_invalid_tgt', float(invalid_tgt), epoch_1000x)
                except Exception:
                    pass
            if heartbeat_interval and (step == 1 or step % heartbeat_interval == 0 or step == len(train_loader)):
                write_heartbeat(
                    out_dir,
                    "finetune.py",
                    "train_batch",
                    torch_module=torch,
                    epoch=epoch,
                    step=step,
                    total_steps=len(train_loader),
                    lr=float(lr),
                    loss=float(loss.item()),
                    total_loss=float(total_loss.detach().item()),
                    moe=float(moe_out.total.detach().item()) if hasattr(moe_out.total, "item") else 0.0,
                    route=float(route_term.detach().item()) if hasattr(route_term, "detach") else float(route_term),
                    force_preassign=bool(force_now),
                )


        epoch_loss = epoch_loss / max(step, 1)
        lr_log.append(lr)
        loss_log.append(epoch_loss)

        tb_writer.add_scalar("epoch/lr", lr, epoch + 1)
        tb_writer.add_scalar("epoch/loss", epoch_loss, epoch + 1)

        logger.info(f"Epoch [{epoch}] Train - lr: {lr:.6g}, loss: {epoch_loss:.6f}")
        print(f"Time: {datetime.now().strftime('%Y/%m/%d-%H:%M')}, Epoch: {epoch}, Loss: {epoch_loss}")
        is_lowest = epoch_loss < best_loss
        if is_lowest:
            best_loss = epoch_loss

        # save latest
        ckpt = {
            "seed": args.seed,
            "args": vars(args),
            "model": model.module.save_parameters(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "moe_cfg": {
                "moe_topk": int(getattr(args, "moe_topk", 2)),
                "moe_temp": float(getattr(args, "moe_temp", 1.0)),
                "moe_noise_std": float(getattr(args, "moe_noise_std", 0.0)),
                "moe_style_bn": int(bool(getattr(args, "moe_style_bn", 1))),
                "moe_style_dropout": float(getattr(args, "moe_style_dropout", 0.10)),
                "moe_disable_style": int(bool(int(getattr(args, "moe_disable_style", 0)))),
                "exclude_source_keywords": list(getattr(args, "exclude_source_keywords", []) or []),
                "moe_cache_name": str(getattr(args, "moe_cache_name", "_moe_routing_cache.json")),
                "adapter_mode": str(getattr(args, "adapter_mode", None) or "moe"),
            },
            "force_preassign": False,
            "force_preassign_train": bool(force_preassign_train),
            "force_preassign_warmup_epochs": int(force_preassign_epochs),
            "moe_train_cache": None,
            "phase": "train_complete",
            "best_loss": float(best_loss),
            "best_dsc": float(best_dsc),
            "best_epoch": int(best_epoch),
            "loss_log": loss_log,
            "lr_log": lr_log,
            "dsc_log": dsc_log,
        }

        # attach training routing cache path (if we have it)
        try:
            p_txt = os.path.join(out_dir, "moe_train_cache_path.txt")
            if os.path.exists(p_txt):
                with open(p_txt, "r", encoding="utf-8") as rf:
                    ckpt["moe_train_cache"] = rf.read().strip() or None
        except Exception:
            pass
        if args.use_amp and (scaler is not None):
            try:
                ckpt["scaler"] = scaler.state_dict()
            except Exception:
                pass
        # -------- Val (DSC) --------
        model.eval()
        epoch_dsc_sum = 0.0
        n = 0

        pbarv = tqdm(val_loader, desc=f"Epoch [{epoch}/{args.num_epochs}] Val", disable=disable_tqdm, mininterval=tqdm_mininterval)
        with torch.no_grad():
            for val_step, (data, label) in enumerate(pbarv, start=1):
                _step_throttle()
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch(
                        data["box"].reshape(-1, 2, 2),
                        data["img"].shape[-2:]
                    ).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])

                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)

                # NOTE: do NOT force routing during validation.
                # Validation should reflect real inference behavior (gate decides experts).

                # label must always be moved, independent of force routing
                label = label.to(device, non_blocking=True)

                mask_pred = model(data)
                if mask_pred.shape[-2:] != label.shape[-2:]:
                    mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)

                mask_prob = torch.sigmoid(mask_pred)
                mask_bin = (mask_prob > 0.5).float()      # (B,M,H,W)
                label_bin = (label > 0).float()           # (B,1,H,W)

                # choose best mask by DSC
                dsc_list = []
                for i in range(model.module.sam.mask_decoder.num_multimask_outputs):
                    pred_i = mask_bin[:, i].unsqueeze(1)
                    dsc_i = dice_batch(pred_i, label_bin)
                    dsc_list.append(dsc_i)
                dsc = torch.stack(dsc_list, dim=0).max(dim=0)[0]  # (B,)

                epoch_dsc_sum += float(dsc.sum().item())
                n += int(dsc.numel())
                pbarv.set_postfix({"dsc": float(dsc.mean().item())})
                if heartbeat_interval and (val_step == 1 or val_step % heartbeat_interval == 0 or val_step == len(val_loader)):
                    write_heartbeat(
                        out_dir,
                        "finetune.py",
                        "val_batch",
                        torch_module=torch,
                        epoch=epoch,
                        step=val_step,
                        total_steps=len(val_loader),
                        batch_dsc=float(dsc.mean().item()),
                        samples_seen=int(n),
                    )

        epoch_dsc = epoch_dsc_sum / max(n, 1)
        dsc_log.append(epoch_dsc)
        tb_writer.add_scalar("epoch/dsc", epoch_dsc, epoch + 1)

        logger.info(f"Epoch [{epoch}] Val - dsc: {epoch_dsc:.6f}")
        print(f"Time: {datetime.now().strftime('%Y/%m/%d-%H:%M')}, Epoch: {epoch}, DSC: {epoch_dsc}")

        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            best_epoch = epoch
            ckpt["best_dsc"] = float(best_dsc)
            ckpt["best_epoch"] = int(best_epoch)
            ckpt["dsc_log"] = dsc_log
            atomic_torch_save(torch, ckpt, os.path.join(out_dir, "model_best.pth"))

        ckpt["phase"] = "epoch_complete"
        ckpt["best_loss"] = float(best_loss)
        ckpt["best_dsc"] = float(best_dsc)
        ckpt["best_epoch"] = int(best_epoch)
        ckpt["loss_log"] = loss_log
        ckpt["lr_log"] = lr_log
        ckpt["dsc_log"] = dsc_log
        if not args.selected_only:
            atomic_torch_save(torch, ckpt, os.path.join(out_dir, "model_latest.pth"))
            write_heartbeat(
                out_dir,
                "finetune.py",
                "checkpoint_saved",
                epoch=epoch,
                path=os.path.join(out_dir, "model_latest.pth"),
                checkpoint_phase="epoch_complete",
            )
        if is_lowest and not args.selected_only:
            atomic_torch_save(torch, ckpt, os.path.join(out_dir, "model_lowest.pth"))

        save_every = max(0, int(getattr(args, "save_every", 0) or 0))
        save_numbered = (epoch == args.num_epochs - 1) or (save_every > 0 and (epoch + 1) % save_every == 0)
        if save_numbered and not args.selected_only:
            atomic_torch_save(torch, ckpt, os.path.join(out_dir, f"model_{epoch}.pth"))

        # The portable baseline package keeps numeric logs; no plot files.

        # epoch summary log file
        with open(os.path.join(out_dir, "result.log"), mode="w", encoding="utf-8") as f:
            for e in range(len(loss_log)):
                f.write(f"Epoch [{e}] - LR: {lr_log[e]}, Loss: {loss_log[e]}, DSC: {dsc_log[e]}\n")
            f.write(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}\n")

        tb_writer.flush()
        write_heartbeat(
            out_dir,
            "finetune.py",
            "epoch_end",
            epoch=epoch,
            lr=float(lr),
            loss=float(epoch_loss),
            dsc=float(epoch_dsc),
            best_epoch=int(best_epoch),
            best_dsc=float(best_dsc),
        )

    total_time = time.time() - time_start
    logger.info(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}")
    logger.info(f"Time cost: {str(timedelta(seconds=int(total_time)))}")
    write_heartbeat(
        out_dir,
        "finetune.py",
        "complete",
        best_epoch=int(best_epoch),
        best_dsc=float(best_dsc),
        time_cost=str(timedelta(seconds=int(total_time))),
    )


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        log_unhandled_exception(args, exc)
        raise

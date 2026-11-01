# -*- coding: utf-8 -*-
"""
InfraSegFM pre-training script
"""

# ==========================
# ✅ 1) 彻底避免 Tk/Tcl 崩溃：强制 Matplotlib 无GUI后端
# 必须放在 import pyplot 之前
# ==========================
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

import argparse
join = os.path.join
import time
import json
import faulthandler
import traceback

import numpy as np
import random
import torch

from datetime import datetime, timedelta

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
        out_dir = join(args.work_dir, args.task_name)
        os.makedirs(out_dir, exist_ok=True)
        tb = traceback.format_exc()
        with open(join(out_dir, "crash.log"), "a", encoding="utf-8") as f:
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
                "pretrain.py",
                "unhandled_exception",
                error=f"{type(exc).__name__}: {exc}",
                traceback=tb[-4000:],
            )
        except Exception:
            pass
    except Exception:
        pass

# 注意：不再依赖 SegmentMetrics 来算 dsc，避免内部 0/0 nan
# from utils.metric import SegmentMetrics


# ==========================
# ✅ 2) 方案A：稳定 Dice/DSC 计算（带 eps + 全空处理）
# ==========================
def dice_batch(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_bin: (B,1,H,W)  0/1 或 bool
    gt_bin:   (B,1,H,W)  0/1
    return:   (B,)       每个样本 dice
    规则：
      - pred 与 gt 都全空 => dice=1
      - 其他情况按标准 dice（加 eps 防止 0/0）
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================
# ✅ Optimizer helper: smaller LR for mask_decoder to prevent late-epoch drift
# ==========================
def build_optimizer(model_dp, args):
    """Build AdamW with param groups.
    - mask_decoder uses lr * mask_decoder_lr_mult (default 0.1)
    - others use lr
    """
    mask_params = []
    other_params = []
    for n, p in model_dp.named_parameters():
        if not p.requires_grad:
            continue
        if 'sam.mask_decoder' in n or ('mask_decoder' in n and 'sam.' in n):
            mask_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({'params': other_params, 'lr': float(args.lr), 'initial_lr': float(args.lr)})
    if mask_params:
        lr_m = float(args.lr) * float(getattr(args, 'mask_decoder_lr_mult', 0.1))
        groups.append({'params': mask_params, 'lr': lr_m, 'initial_lr': lr_m})

    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


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


# setup parser
parser = argparse.ArgumentParser("InfraSegFM training", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--task_name", type=str, default="CivilSegX")
parser.add_argument("--method", type=str, default="infrasegfm")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--task_drop_p", type=float, default=0.0,
                    help="Training-time task-identity dropout: replace task id with the unknown bucket with this probability, teaching the gate open-set routing (0 = off).")
parser.add_argument("--lr_schedule", type=str, default="const", choices=["const", "cosine"],
                    help="Learning-rate schedule. 'cosine' decays each param group from its base LR to lr_min_ratio*base over num_epochs.")
parser.add_argument("--lr_min_ratio", type=float, default=0.05,
                    help="Final LR as a fraction of base LR for cosine schedule.")
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
# data
parser.add_argument("--data_path", type=str, default="./playground/InfraSegDB",
                    help="path to InfraSegDB data folder")
parser.add_argument("--exclude_source_keywords", type=str, nargs="*", default=[],
                    help="Ablation/data filter: exclude paired .npy samples whose image or mask path contains any keyword, e.g. SynBridgeCrack.")
parser.add_argument("--moe_cache_name", type=str, default="_moe_routing_cache.json",
                    help="MoE routing-cache filename. Use variant-specific names for ablations to avoid stale cache reuse.")
# env
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./playground")
# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1],
                    help="1 enables DataLoader pinned memory; 0 disables it for Windows stability.")
parser.add_argument("--resume", type=str, default=None,
                    help="resume training from checkpoint")
# optimizer
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                    help="learning rate (absolute lr default: 0.001)")
parser.add_argument("--weight_decay", type=float, default=0.01,
                    help="weight decay (default: 0.01)")
parser.add_argument("--mask_decoder_lr_mult", type=float, default=0.1,
                    help="learning-rate multiplier for SAM mask_decoder params (default: 0.1).")
parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="global grad clipping (norm). 0 disables. (default: 1.0)")
parser.add_argument("--auto_resume", type=int, default=1, choices=[0,1],
                    help="auto resume from <work_dir>/<task_name>/model_latest.pth if exists and --resume is not set. (default: 1)")
parser.add_argument("--save_every", type=int, default=0,
                    help="save numbered epoch checkpoints every N epochs; 0 saves only the final numbered checkpoint. model_latest/model_best are still maintained.")
parser.add_argument("--heartbeat_interval", type=int, default=100,
                    help="write training heartbeat JSON every N batches; 0 disables batch heartbeats.")
parser.add_argument("--disable_tqdm", type=int, default=int(os.environ.get("INFRASEGFM_DISABLE_TQDM", "0") or 0),
                    choices=[0, 1],
                    help="disable tqdm progress bars for cleaner logged runs. Heartbeat status is still written.")
parser.add_argument("--skip_nonfinite", type=int, default=1, choices=[0,1],
                    help="skip optimizer step when loss is NaN/Inf to prevent corruption. (default: 1)")
parser.add_argument("--use_amp", action="store_true", default=False,
                    help="whether to use amp")
parser.add_argument("--task_balanced_sampling", type=int, default=1,
                    help="1: balance sampling across TaskFolders (recommended for highly imbalanced datasets); 0: plain shuffle")

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
    from utils.runtime_logging import atomic_torch_save, write_heartbeat
    from utils.sampler import build_task_mixed_sampler

    set_seeds(2025)
    device = torch.device(args.device)

    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    # 这里 image_size=256 是你当前代码的设定：会把 512 输入缩到 256
    # 如果你想原生训练 512，可以改为 image_size=512，但显存会更高。
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=checkpoint)

    method = str(getattr(args, "method", "infrasegfm")).lower().strip()
    # Lightweight heuristic: treat names containing 'sam' (but not 'moe') as the decoder-tuning baseline.
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
            adapter_mode=str(getattr(args, "adapter_mode", "moe")),
            task_drop_p=float(getattr(args, "task_drop_p", 0.0)),
        ).to(device)
        use_moe = True

    model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    work_dir = join(args.work_dir, args.task_name)
    os.makedirs(work_dir, exist_ok=True)
    enable_fault_log(work_dir)
    log_writer = SummaryWriter(log_dir=work_dir)
    logger = get_logger(log_file=os.path.join(work_dir, 'output.log'))
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")
    heartbeat_interval = max(0, int(getattr(args, "heartbeat_interval", 100) or 0))
    disable_tqdm = bool(int(getattr(args, "disable_tqdm", 0) or 0))
    tqdm_mininterval = float(os.environ.get("INFRASEGFM_TQDM_MININTERVAL", "30") or 30)
    write_heartbeat(work_dir, "pretrain.py", "init", torch_module=torch, args=vars(args))

    logger.info("Model: %s" % str(model))
    logger.info("Number of total parameters: %d" % (sum(p.numel() for p in model.parameters())))
    logger.info("Number of trainable parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = build_optimizer(model, args)
    criterion = DiceBCELoss(sigmoid=True, squared_pred=True, reduction='none')
    logger.info("Criterion: %s" % str(criterion))

    
    # ==========================
    # ✅ MoE routing pre-assign (task-first, dynamic-K)
    # This writes <data_path>/train/_moe_routing_cache.json and dataset.py will attach `moe_target`.
    # ==========================
    if int(getattr(args, "moe_preassign", 1)) == 1 and use_moe and str(getattr(args, "adapter_mode", "moe")) == "moe":
        try:
            cache_name = str(getattr(args, "moe_cache_name", "_moe_routing_cache.json"))
            cache_path_expected = os.path.join(join(args.data_path, "train"), cache_name)
            logger.info(f"MoE preassign start: cache={cache_path_expected}, exists={os.path.exists(cache_path_expected)}")
            write_heartbeat(
                work_dir,
                "pretrain.py",
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
            cache_path = build_routing_cache(join(args.data_path, "train"), expert_num=args.expert_num, cfg=cfg, force_rebuild=args.moe_preassign_force)
            logger.info(f"MoE preassign cache ready: {cache_path}")
            write_heartbeat(
                work_dir,
                "pretrain.py",
                "moe_preassign_done",
                torch_module=torch,
                cache_path=cache_path,
                cache_size=os.path.getsize(cache_path) if os.path.exists(cache_path) else None,
            )
            # Write a pointer file so evaluation can apply training routing to ID/ODD without
            # changing command line.
            try:
                with open(os.path.join(work_dir, "moe_train_cache_path.txt"), "w", encoding="utf-8") as wf:
                    wf.write(os.path.normpath(cache_path))
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"MoE preassign failed (continue without it): {e}")
    train_dataset = InfraSegDataset(
        join(args.data_path, "train"),
        train=True,
        exclude_source_keywords=getattr(args, "exclude_source_keywords", []),
        moe_cache_name=getattr(args, "moe_cache_name", "_moe_routing_cache.json"),
    )
    logger.info(f"Number of training samples: {len(train_dataset)}")
    pin_memory = bool(int(getattr(args, "pin_memory", 0))) and int(args.num_workers) > 0
    if args.task_balanced_sampling == 1:
        sampler = build_task_mixed_sampler(train_dataset.task_folders, alpha=args.task_sampling_alpha, power=args.task_sampling_power, num_samples=args.task_sampling_num_samples)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_dataset = InfraSegDataset(
        join(args.data_path, "val"),
        train=False,
        exclude_source_keywords=getattr(args, "exclude_source_keywords", []),
        moe_cache_name=getattr(args, "moe_cache_name", "_moe_routing_cache.json"),
    )
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)  # 输入缩放到 256
    box_transform = ResizeLongestSide(img_size)

    num_epochs = args.num_epochs
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0
    best_epoch = -1
    loss_log = []
    lr_log = []
    dsc_log = []

    # -------------------------
    # Resume (explicit --resume or auto-resume from model_latest)
    # -------------------------
    resume_path = None
    if args.resume is not None and os.path.isfile(args.resume):
        resume_path = args.resume
    elif int(getattr(args, 'auto_resume', 1)) == 1:
        cand = join(work_dir, 'model_latest.pth')
        if os.path.isfile(cand):
            resume_path = cand

    if resume_path is not None:
        logger.info(f'Loading resume checkpoint: {resume_path}')
        checkpoint = torch.load(resume_path, map_location=device)
        if 'epoch' in checkpoint:
            resume_phase = str(checkpoint.get("phase", "epoch_complete"))
            resume_epoch = int(checkpoint['epoch'])
            start_epoch = resume_epoch + (1 if resume_phase == "epoch_complete" else 0)
        if 'model' in checkpoint:
            model.module.load_parameters(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_dsc = float(checkpoint.get("best_dsc", best_dsc))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        loss_log = list(checkpoint.get("loss_log", loss_log) or loss_log)
        lr_log = list(checkpoint.get("lr_log", lr_log) or lr_log)
        dsc_log = list(checkpoint.get("dsc_log", dsc_log) or dsc_log)
        if args.use_amp and ('scaler' in checkpoint) and ('scaler' not in locals()):
            pass
        logger.info(f'Resume done. start_epoch={start_epoch}')
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "resume_loaded",
            torch_module=torch,
            resume_path=resume_path,
            resume_epoch=int(checkpoint.get("epoch", -1)),
            resume_phase=str(checkpoint.get("phase", "epoch_complete")),
            start_epoch=start_epoch,
            best_epoch=best_epoch,
            best_dsc=best_dsc,
        )

    scaler = None
    if args.use_amp:
        # ✅ 建议写法：torch.amp.GradScaler('cuda')
        scaler = make_amp_grad_scaler()

        # load scaler state if present in resume checkpoint
        try:
            if 'checkpoint' in locals() and isinstance(checkpoint, dict) and ('scaler' in checkpoint):
                scaler.load_state_dict(checkpoint['scaler'])
                logger.info('AMP scaler state loaded from checkpoint.')
        except Exception as e:
            logger.warning(f'Failed to load AMP scaler state: {e}')

    if start_epoch >= num_epochs and 'checkpoint' in locals() and isinstance(checkpoint, dict):
        final_marker = join(work_dir, f"model_{max(num_epochs - 1, 0)}.pth")
        if not os.path.isfile(final_marker):
            checkpoint["phase"] = "epoch_complete"
            atomic_torch_save(torch, checkpoint, final_marker)
            logger.info(f"Final epoch marker recreated from model_latest: {final_marker}")
            write_heartbeat(
                work_dir,
                "pretrain.py",
                "final_marker_recreated",
                torch_module=torch,
                path=final_marker,
                start_epoch=start_epoch,
                num_epochs=num_epochs,
            )

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        if str(getattr(args, "lr_schedule", "const")) == "cosine":
            import math as _math
            t = float(epoch) / float(max(1, num_epochs - 1))
            _factor = float(args.lr_min_ratio) + (1.0 - float(args.lr_min_ratio)) * 0.5 * (1.0 + _math.cos(_math.pi * t))
            for _g in optimizer.param_groups:
                base = float(_g.get("initial_lr", _g["lr"]))
                _g["lr"] = base * _factor
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "epoch_start",
            torch_module=torch,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        # ==========================
        # Train
        # ==========================
        epoch_loss = 0
        step = 0
        model.train()
        pbar_train = tqdm(train_dataloader, disable=disable_tqdm, mininterval=tqdm_mininterval)
        pbar_train.set_description(f"Epoch [{epoch}/{num_epochs}] Train")
        for data, label in pbar_train:
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # 你的数据是 512x512，但 SAM image_size=256 -> 这里会做 resize
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    data["box"].reshape(-1, 2, 2),
                    data["img"].shape[-2:]
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
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

            # ==========================
            # ✅ 3) 512x512 的隐患修复：插值用 (H,W) 而不是只用 W
            # 去掉 antialias，明确 align_corners=False 更稳
            # ==========================
            if mask_pred.shape[-2:] != label.shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)

            losses = []
            for i in range(model.module.sam.mask_decoder.num_multimask_outputs):
                output = mask_pred[:, i].unsqueeze(1)
                loss_i = criterion(output.float(), label)
                losses.append(loss_i)
            loss = torch.stack(losses, dim=0).min(dim=0)[0]
            loss = loss.mean()

            # MoE auxiliary loss (cheap; helps prevent expert collapse)
            # Warmup is important on highly imbalanced multi-task data:
            # let the segmentation objective settle first, then gradually turn on MoE regularization.
            if args.moe_warmup_epochs and args.moe_warmup_epochs > 0:
                moe_scale = min(1.0, float(epoch + 1) / float(args.moe_warmup_epochs))
            else:
                moe_scale = 1.0

            moe_out = moe_aux_loss(
                gates,
                lb_coef=args.moe_lb_coef * moe_scale,
                ent_coef=args.moe_ent_coef * moe_scale,
                topk_for_load=1,
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

            if args.moe_route_sup_coef > 0 and gates and isinstance(data, dict) and ("moe_target" in data):
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
                if args.use_amp:
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

            epoch_loss += loss.item()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            pbar_train.set_postfix({
            "lr": lr,
            "loss": float(loss.item()),
            "moe": float(moe_out.total.detach().item()) if hasattr(moe_out.total, 'item') else 0.0,
            "route": float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term),
            })

            epoch_1000x = int((epoch + step / len(train_dataloader)) * 1000)
            log_writer.add_scalar('batch/lr', lr, epoch_1000x)
            log_writer.add_scalar('batch/loss', float(loss.item()), epoch_1000x)
            log_writer.add_scalar('batch/moe_total', float(moe_out.total.detach().item()), epoch_1000x)
            log_writer.add_scalar('batch/moe_lb', float(moe_out.lb.detach().item()), epoch_1000x)
            log_writer.add_scalar('batch/moe_ent', float(moe_out.ent.detach().item()), epoch_1000x)
            if int(getattr(args, 'moe_route_log', 1)) == 1:
                try:
                    log_writer.add_scalar('batch/route_term', float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term), epoch_1000x)
                    log_writer.add_scalar('batch/route_loss', float(route_loss.detach().item()) if hasattr(route_loss, 'detach') else 0.0, epoch_1000x)
                    log_writer.add_scalar('batch/route_coef', float(route_coef), epoch_1000x)
                    log_writer.add_scalar('batch/route_invalid_tgt', float(invalid_tgt), epoch_1000x)
                except Exception:
                    pass
            if heartbeat_interval and (step == 1 or step % heartbeat_interval == 0 or step == len(train_dataloader)):
                write_heartbeat(
                    work_dir,
                    "pretrain.py",
                    "train_batch",
                    torch_module=torch,
                    epoch=epoch,
                    step=step,
                    total_steps=len(train_dataloader),
                    lr=float(lr),
                    loss=float(loss.item()),
                    total_loss=float(total_loss.detach().item()),
                    moe=float(moe_out.total.detach().item()) if hasattr(moe_out.total, "item") else 0.0,
                    route=float(route_term.detach().item()) if hasattr(route_term, "detach") else float(route_term),
                )

        lr_log.append(lr)
        epoch_loss /= step
        loss_log.append(epoch_loss)
        log_writer.add_scalar('epoch/lr', lr, epoch + 1)
        log_writer.add_scalar('epoch/loss', epoch_loss, epoch + 1)
        print(f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        is_lowest = epoch_loss < best_loss
        if is_lowest:
            best_loss = epoch_loss

        # save models
        ckpt = {
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
                "adapter_mode": str(getattr(args, "adapter_mode", "moe")),
            },
            "force_preassign": False,
            "phase": "train_complete",
            "best_loss": float(best_loss),
            "best_dsc": float(best_dsc),
            "best_epoch": int(best_epoch),
            "loss_log": loss_log,
            "lr_log": lr_log,
            "dsc_log": dsc_log,
        }
        if args.use_amp and (scaler is not None):
            try:
                ckpt["scaler"] = scaler.state_dict()
            except Exception:
                pass
        # ==========================
        # Eval
        # ==========================
        epoch_dsc = 0.0
        size = 0
        model.eval()
        pbar_val = tqdm(val_dataloader, disable=disable_tqdm, mininterval=tqdm_mininterval)
        pbar_val.set_description(f"Epoch [{epoch}/{num_epochs}] Val")

        with torch.no_grad():
            for val_step, (data, label) in enumerate(pbar_val, start=1):
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch(
                        data["box"].reshape(-1, 2, 2),
                        data["img"].shape[-2:]
                    ).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])

                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                mask_pred = model(data)

                if mask_pred.shape[-2:] != label.shape[-2:]:
                    mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)

                # mask_pred: (B, M, H, W) logits
                mask_prob = torch.sigmoid(mask_pred)
                mask_bin = (mask_prob > 0.5).float()  # 0/1 float，避免 bool 引发内部类型问题
                label_bin = (label > 0).float()

                # 多 mask 输出：取 dice 最大的那个
                dsc_ambiguous = []
                for idx in range(model.module.sam.mask_decoder.num_multimask_outputs):
                    pred_i = mask_bin[:, idx].unsqueeze(1)  # (B,1,H,W)
                    dsc_i = dice_batch(pred_i, label_bin)   # (B,)
                    dsc_ambiguous.append(dsc_i)

                dsc = torch.stack(dsc_ambiguous, dim=0).max(dim=0)[0]  # (B,)

                epoch_dsc += dsc.sum().item()
                size += dsc.shape[0]
                pbar_val.set_postfix({"dsc": float(dsc.mean().item())})
                if heartbeat_interval and (val_step == 1 or val_step % heartbeat_interval == 0 or val_step == len(val_dataloader)):
                    write_heartbeat(
                        work_dir,
                        "pretrain.py",
                        "val_batch",
                        torch_module=torch,
                        epoch=epoch,
                        step=val_step,
                        total_steps=len(val_dataloader),
                        batch_dsc=float(dsc.mean().item()),
                        samples_seen=int(size),
                    )

        epoch_dsc = epoch_dsc / max(size, 1)
        dsc_log.append(epoch_dsc)
        log_writer.add_scalar('epoch/dsc', epoch_dsc, epoch + 1)
        print(f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {epoch_dsc}')

        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            best_epoch = epoch
            # Save extra metadata for evaluation-time forced routing.
            moe_train_cache = None
            try:
                p_txt = os.path.join(work_dir, "moe_train_cache_path.txt")
                if os.path.exists(p_txt):
                    with open(p_txt, "r", encoding="utf-8") as rf:
                        moe_train_cache = rf.read().strip()
            except Exception:
                moe_train_cache = None

            atomic_torch_save(
                torch,
                {
                    "model": model.module.save_parameters(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "force_preassign": False,
                    "moe_train_cache": moe_train_cache,
                    "moe_cfg": {
                        "moe_topk": int(getattr(args, "moe_topk", 2)),
                        "moe_temp": float(getattr(args, "moe_temp", 1.0)),
                        "moe_noise_std": float(getattr(args, "moe_noise_std", 0.0)),
                        "moe_style_bn": int(bool(getattr(args, "moe_style_bn", 1))),
                        "moe_style_dropout": float(getattr(args, "moe_style_dropout", 0.10)),
                        "moe_disable_style": int(bool(int(getattr(args, "moe_disable_style", 0)))),
                        "exclude_source_keywords": list(getattr(args, "exclude_source_keywords", []) or []),
                        "moe_cache_name": str(getattr(args, "moe_cache_name", "_moe_routing_cache.json")),
                    },
                },
                join(work_dir, "model_best.pth"),
            )

        ckpt["phase"] = "epoch_complete"
        ckpt["best_loss"] = float(best_loss)
        ckpt["best_dsc"] = float(best_dsc)
        ckpt["best_epoch"] = int(best_epoch)
        ckpt["loss_log"] = loss_log
        ckpt["lr_log"] = lr_log
        ckpt["dsc_log"] = dsc_log
        atomic_torch_save(torch, ckpt, join(work_dir, "model_latest.pth"))
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "checkpoint_saved",
            epoch=epoch,
            path=join(work_dir, "model_latest.pth"),
            checkpoint_phase="epoch_complete",
        )

        if is_lowest:
            atomic_torch_save(torch, ckpt, join(work_dir, "model_lowest.pth"))

        save_every = max(0, int(getattr(args, "save_every", 0) or 0))
        save_numbered = (epoch == num_epochs - 1) or (save_every > 0 and (epoch + 1) % save_every == 0)
        if save_numbered:
            atomic_torch_save(torch, ckpt, join(work_dir, f"model_{epoch}.pth"))

        # ==========================
        # Plot curves (Agg backend, safe)
        # ==========================
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "plot_start",
            epoch=epoch,
        )
        plt.figure()
        plt.plot(loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(lr_log)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.savefig(join(work_dir, "lr.png"))
        plt.close()

        plt.figure()
        plt.plot(dsc_log)
        plt.title("Validation DSC")
        plt.xlabel("Epoch")
        plt.ylabel("DSC")
        plt.savefig(join(work_dir, "val_dsc.png"))
        plt.close()
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "plot_done",
            epoch=epoch,
        )

        logger.info(f"Epoch [{epoch}] - LR: {lr}, Loss: {epoch_loss}, DSC: {epoch_dsc}")
        log_writer.flush()
        write_heartbeat(
            work_dir,
            "pretrain.py",
            "epoch_end",
            epoch=epoch,
            lr=float(lr),
            loss=float(epoch_loss),
            dsc=float(epoch_dsc),
            best_epoch=int(best_epoch),
            best_dsc=float(best_dsc),
        )

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}")
    logger.info(f"Time cost: {total_time_str}")
    write_heartbeat(
        work_dir,
        "pretrain.py",
        "complete",
        best_epoch=int(best_epoch),
        best_dsc=float(best_dsc),
        time_cost=total_time_str,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        log_unhandled_exception(args, exc)
        raise

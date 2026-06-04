"""Train InfraSegFM on a train/val dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrasegfm import InfraSegDataset, build_infrasegfm
from infrasegfm.losses import DiceBCELoss
from infrasegfm.metrics import dice_empty_aware
from infrasegfm.moe_loss import moe_aux_loss
from infrasegfm.moe_preassign import PreassignConfig, build_routing_cache
from infrasegfm.train_utils import ensure_dir, load_torch_checkpoint, move_batch_to_device, select_best_mask, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/ and val/.")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="runs/demo")
    parser.add_argument("--sam_checkpoint", type=str, default="", help="Path to SAM checkpoint. Empty means random init.")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--image_size", type=int, default=256, help="InfraSegFM paper setting: SAM image encoder input size.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=80, help="Paper workflow default.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resume", type=str, default="", help="Optional InfraSegFM checkpoint for fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--mask_decoder_lr_mult", type=float, default=0.1)
    parser.add_argument("--bottleneck_dim", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--moe_topk", type=int, default=2)
    parser.add_argument("--moe_temp", type=float, default=1.0)
    parser.add_argument("--moe_noise_std", type=float, default=0.0)
    parser.add_argument("--moe_style_bn", type=int, default=1)
    parser.add_argument("--moe_style_dropout", type=float, default=0.10)
    parser.add_argument("--moe_lb_coef", type=float, default=0.01)
    parser.add_argument("--moe_ent_coef", type=float, default=0.01)
    parser.add_argument("--moe_warmup_epochs", type=int, default=5)
    parser.add_argument("--moe_preassign", type=int, default=1, choices=[0, 1])
    parser.add_argument("--moe_preassign_force", action="store_true")
    parser.add_argument("--moe_cache_name", type=str, default="_moe_routing_cache.json")
    parser.add_argument("--moe_route_sup_coef", type=float, default=0.05)
    parser.add_argument("--moe_route_warmup", type=int, default=2)
    parser.add_argument("--moe_route_cap_ratio", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def loss_for_multimask(criterion, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
    losses = [criterion(logits[:, idx : idx + 1].float(), label.float()) for idx in range(logits.shape[1])]
    return torch.stack(losses, dim=0).min(dim=0)[0].mean()


def build_optimizer(model, args):
    mask_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mask_decoder" in name:
            mask_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": float(args.lr)})
    if mask_params:
        groups.append({"params": mask_params, "lr": float(args.lr) * float(args.mask_decoder_lr_mult)})
    return torch.optim.AdamW(groups, lr=float(args.lr), weight_decay=float(args.weight_decay))


def routing_supervision_loss(gates, target: torch.Tensor):
    if not gates:
        return None
    try:
        target = target.long()
        nlls = []
        for gate in gates:
            if gate is None:
                continue
            expert_num = int(gate.shape[1])
            valid = (target >= 0) & (target < expert_num)
            if not valid.any():
                continue
            probs = torch.clamp(gate[valid], min=1e-8)
            idx = target[valid].clamp(0, expert_num - 1).view(-1, 1)
            nlls.append((-torch.log(probs.gather(1, idx).squeeze(1))).mean())
        if nlls:
            return torch.stack(nlls).mean()
    except Exception:
        return None
    return None


@torch.no_grad()
def validate(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for data, label in loader:
        data, label = move_batch_to_device(data, label, device)
        logits = model(data)
        pred = select_best_mask(logits, label)
        dsc = dice_empty_aware(pred, label)
        total += float(dsc.sum().item())
        count += int(dsc.numel())
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_root = Path(args.data_root) / args.train_split
    val_root = Path(args.data_root) / args.val_split
    if int(args.moe_preassign) == 1:
        cfg = PreassignConfig(cache_name=args.moe_cache_name)
        cache_path = build_routing_cache(str(train_root), expert_num=args.expert_num, cfg=cfg, force_rebuild=bool(args.moe_preassign_force))
        print(f"[train] MoE route-prior cache: {cache_path}")

    train_set = InfraSegDataset(str(train_root), train=True, image_size=args.image_size, moe_cache_name=args.moe_cache_name)
    val_set = InfraSegDataset(str(val_root), train=False, image_size=args.image_size, moe_cache_name=args.moe_cache_name)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_infrasegfm(
        model_type=args.model_type,
        sam_checkpoint=args.sam_checkpoint,
        image_size=args.image_size,
        bottleneck_dim=args.bottleneck_dim,
        embedding_dim=args.embedding_dim,
        expert_num=args.expert_num,
        gate_topk=args.moe_topk,
        gate_temperature=args.moe_temp,
        gate_noise_std=args.moe_noise_std,
        style_use_std=bool(args.moe_style_bn),
        style_dropout=args.moe_style_dropout,
    ).to(device)

    if args.resume:
        ckpt = load_torch_checkpoint(args.resume, map_location="cpu")
        model.load_parameters(ckpt.get("model", ckpt))
        print(f"[train] initialized from {args.resume}")

    optimizer = build_optimizer(model, args)
    criterion = DiceBCELoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_dsc = -1.0
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_dsc"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            steps = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for data, label in pbar:
                data, label = move_batch_to_device(data, label, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    out = model(data, return_gates=True)
                    seg_loss = loss_for_multimask(criterion, out["masks"], label)
                    warm = min(1.0, float(epoch) / float(max(1, args.moe_warmup_epochs))) if args.moe_warmup_epochs > 0 else 1.0
                    aux = moe_aux_loss(
                        out.get("gates", []),
                        lb_coef=args.moe_lb_coef * warm,
                        ent_coef=args.moe_ent_coef * warm,
                    )
                    route_term = torch.tensor(0.0, device=device)
                    if "moe_target" in data and args.moe_route_sup_coef > 0:
                        route_loss = routing_supervision_loss(out.get("gates", []), data["moe_target"])
                        if route_loss is not None:
                            route_warm = min(1.0, float(max(epoch - 1, 0)) / float(max(1, args.moe_route_warmup)))
                            route_coef = float(args.moe_route_sup_coef) * route_warm
                            route_term = route_loss * route_coef
                            cap = float(args.moe_route_cap_ratio) * seg_loss.detach()
                            route_term = torch.minimum(route_term, cap)
                    loss = seg_loss + aux.total.to(seg_loss.device) + route_term
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                running += float(seg_loss.detach().item())
                steps += 1
                pbar.set_postfix(loss=running / max(steps, 1))

            train_loss = running / max(steps, 1)
            val_dsc = validate(model, val_loader, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_dsc": val_dsc})
            f.flush()

            ckpt = {
                "model": model.save_parameters(),
                "epoch": epoch,
                "args": vars(args),
                "val_dsc": val_dsc,
            }
            torch.save(ckpt, output_dir / "model_last.pth")
            if val_dsc > best_dsc:
                best_dsc = val_dsc
                torch.save(ckpt, output_dir / "model_best.pth")
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_dsc={val_dsc:.4f}")

    print(f"Training finished. Best checkpoint: {output_dir / 'model_best.pth'}")


if __name__ == "__main__":
    main()

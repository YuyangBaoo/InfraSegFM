"""Run InfraSegFM inference on unlabeled images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrasegfm import build_infrasegfm
from infrasegfm.dataset import resolve_task_folder_meta
from infrasegfm.train_utils import ensure_dir, load_torch_checkpoint, move_batch_to_device

IMAGE_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output_dir", type=str, default="runs/predict")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/infrasegfm_vit_b.pth", help="InfraSegFM checkpoint.")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--task_folder", type=str, default="Handheld_ConcreteCrack", help="Task metadata, e.g. Handheld_ConcreteCrack.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"), help="Optional box in original image coordinates. Default: full image.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[predict] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def list_images(path: Path):
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def read_image(path: Path):
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    return np.asarray(Image.open(path).convert("RGB"))


def make_batch(image: np.ndarray, image_size: int, task_folder: str, box):
    h, w = image.shape[:2]
    img_t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    img_t = F.interpolate(img_t, size=(image_size, image_size), mode="bilinear", align_corners=False)

    if box is None:
        box_t = torch.tensor([[0, 0, image_size - 1, image_size - 1]], dtype=torch.float32)
    else:
        x1, y1, x2, y2 = box
        sx = float(image_size) / max(float(w), 1.0)
        sy = float(image_size) / max(float(h), 1.0)
        box_t = torch.tensor([[x1 * sx, y1 * sy, x2 * sx, y2 * sy]], dtype=torch.float32)

    platform, l1, l2, l3, task = resolve_task_folder_meta(task_folder)
    data = {
        "img": img_t,
        "box": box_t,
        "platform": torch.tensor([platform], dtype=torch.long),
        "context": (
            torch.tensor([l1], dtype=torch.long),
            torch.tensor([l2], dtype=torch.long),
            torch.tensor([l3], dtype=torch.long),
            torch.tensor([task], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ),
        "name": [""],
        "task_folder": [task_folder],
    }
    return data, (h, w)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(out_dir / "masks")

    ckpt = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    moe_cfg = ckpt.get("moe_cfg", {}) if isinstance(ckpt, dict) else {}
    model = build_infrasegfm(
        model_type=args.model_type,
        sam_checkpoint=args.sam_checkpoint,
        image_size=args.image_size,
        bottleneck_dim=int(saved_args.get("bottleneck_dim", 16)),
        embedding_dim=int(saved_args.get("embedding_dim", 16)),
        expert_num=int(saved_args.get("expert_num", 4)),
        gate_topk=int(saved_args.get("moe_topk", moe_cfg.get("moe_topk", 2))),
        gate_temperature=float(saved_args.get("moe_temp", moe_cfg.get("moe_temp", 1.0))),
        gate_noise_std=float(saved_args.get("moe_noise_std", moe_cfg.get("moe_noise_std", 0.0))),
        style_use_std=bool(saved_args.get("moe_style_bn", moe_cfg.get("moe_style_bn", 1))),
        style_dropout=float(saved_args.get("moe_style_dropout", moe_cfg.get("moe_style_dropout", 0.10))),
    ).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_parameters(state)
    model.eval()

    images = list_images(Path(args.input))
    if not images:
        raise ValueError(f"No images found: {args.input}")

    with torch.no_grad():
        for path in tqdm(images, desc="Predicting"):
            image = read_image(path)
            data, original_size = make_batch(image, args.image_size, args.task_folder, args.box)
            label_stub = torch.zeros((1, 1, args.image_size, args.image_size), dtype=torch.long)
            data, _ = move_batch_to_device(data, label_stub, device)
            out = model(data, return_iou=True)
            logits = out["masks"]
            iou = out["iou_predictions"]
            best = torch.argmax(iou, dim=1)
            selected = logits[torch.arange(logits.shape[0], device=device), best].unsqueeze(1)
            selected = F.interpolate(selected, size=original_size, mode="bilinear", align_corners=False)
            mask = (torch.sigmoid(selected)[0, 0] > float(args.threshold)).detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(mask).save(mask_dir / f"{path.stem}_mask.png")

    print(f"Predicted masks written to {mask_dir}")


if __name__ == "__main__":
    main()

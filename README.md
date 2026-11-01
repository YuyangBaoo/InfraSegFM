# InfraSegFM

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backbone](https://img.shields.io/badge/backbone-SAM%20ViT--B-lightgrey)](https://github.com/facebookresearch/segment-anything)

Code and pretrained weights for **InfraSegFM**, a SAM-based framework for civil
infrastructure defect segmentation.

**Paper:** *Transferable Civil Infrastructure Defect Segmentation with Source-Aware Benchmarking and Routed SAM Adaptation*

**Authors:** Baoxian Li, Yuyang Bao, Si Chen, Mingwei Fang, Longsheng Bao,
Jiakang Zhao, and Ling Yu

InfraSegFM freezes the SAM image and prompt encoders and trains context-routed
residual adapters together with the mask decoder. Hierarchical task metadata,
acquisition platform, source information, and image-style statistics condition
sparse expert selection. The Shared variant uses a shared residual adapter as
a parameter-matched comparison.

![InfraSegFM overview](assets/overview.png)

## Setup

Clone the repository and download the weights with Git LFS:

```bash
git clone https://github.com/YuyangBaoo/InfraSegFM.git
cd InfraSegFM
git lfs install
git lfs pull
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install PyTorch and torchvision using the [official platform selector](https://pytorch.org/get-started/locally/), then install the project:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the SAM ViT-B backbone:

```bash
curl -L -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"
```

## Pretrained weights

| Model | File |
| --- | --- |
| InfraSegFM Full | `checkpoints/infrasegfm_vit_b.pth` |
| Full v1 | `checkpoints/prompt_protocol/inputs/models/full_v1_model_best.pth` |
| Shared v1 | `checkpoints/prompt_protocol/inputs/models/shared_v1_model_best.pth` |
| Full v1 quality head | `checkpoints/prompt_protocol/inputs/heads/full_v1/quality_head_best.pt` |
| Full v2 quality head | `checkpoints/prompt_protocol/inputs/heads/full_v2/quality_head_best.pt` |
| Shared v1 quality head | `checkpoints/prompt_protocol/inputs/heads/shared_v1/quality_head_best.pt` |
| Shared v2 quality head | `checkpoints/prompt_protocol/inputs/heads/shared_v2/quality_head_best.pt` |

## Input format

Organize paired training images and binary masks by split and task:

```text
your_dataset/
  train/
    Handheld_ConcreteCrack/
      npy_imgs/sample_001.npy
      npy_gts/sample_001.npy
  val/
    Handheld_ConcreteCrack/
      npy_imgs/sample_101.npy
      npy_gts/sample_101.npy
```

For common image formats, use `images/` and `masks/` in each task directory.
Pair images and masks by filename stem. Foreground mask values are greater than
zero. Task folders follow `<Platform>_<TaskName>`; task metadata is defined in
[`infrasegfm/taxonomy.py`](infrasegfm/taxonomy.py).

The synthetic examples in `demo_data/` provide a small input set for trying the
training and evaluation commands.

## Train

```bash
python tools/train.py \
  --data_root /path/to/your_dataset \
  --output_dir runs/train \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --image_size 256 \
  --epochs 30 \
  --batch_size 80 \
  --lr 5e-4 \
  --device cuda:0 \
  --amp
```

Hyperparameters are collected in [`configs/paper_vit_b.json`](configs/paper_vit_b.json).
For fine-tuning, add `--resume checkpoints/infrasegfm_vit_b.pth` and set
`--lr 5e-5`.

## Evaluate

`tools/evaluate.py` performs labeled reference evaluation with GT-Dice Oracle
candidate selection.

```bash
python tools/evaluate.py \
  --data_root /path/to/your_dataset/val \
  --checkpoint checkpoints/infrasegfm_vit_b.pth \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --image_size 256 \
  --batch_size 50 \
  --metrics dsc \
  --device cuda:0 \
  --out_dir runs/eval
```

The command writes `metrics.csv` and `summary.csv` to the output directory.
For the Full/Shared prompt-evaluation commands, inspect the launcher:

```bash
python tools/run_prompt_protocol.py --help
python tools/run_prompt_protocol.py profiles
```

## Predict

```bash
python tools/predict.py \
  --input /path/to/images \
  --output_dir runs/predict \
  --checkpoint checkpoints/infrasegfm_vit_b.pth \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --task_folder Handheld_ConcreteCrack \
  --device cuda:0
```

Prediction uses a full-image box by default and selects a mask using the
predicted quality score. Pass `--box X1 Y1 X2 Y2` to use a box in original
image coordinates. Masks are saved to `runs/predict/masks/` at the original
image size. Use `--device cpu` for CPU execution.

## Citation

```bibtex
@misc{li2026infrasegfm,
  title  = {Transferable Civil Infrastructure Defect Segmentation with Source-Aware Benchmarking and Routed SAM Adaptation},
  author = {Li, Baoxian and Bao, Yuyang and Chen, Si and Fang, Mingwei and Bao, Longsheng and Zhao, Jiakang and Yu, Ling},
  year   = {2026},
  note   = {Manuscript}
}
```

InfraSegFM code uses the [MIT License](LICENSE); SAM-derived source uses [Apache License 2.0](LICENSES/Apache-2.0.txt) and retains its original copyright notices.

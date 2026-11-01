# InfraSegFM

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backbone](https://img.shields.io/badge/backbone-SAM%20ViT--B-lightgrey)](https://github.com/facebookresearch/segment-anything)
[![Model Card](https://img.shields.io/badge/model-card-informational)](MODEL_CARD.md)

Official implementation and release checkpoint for **InfraSegFM**, a
SAM-based framework for open-world civil infrastructure defect segmentation.

**Paper:** *Transferable Civil Infrastructure Defect Segmentation with Source-Aware Benchmarking and Routed SAM Adaptation*

**Authors:** Baoxian Li, Yuyang Bao, Si Chen, Mingwei Fang, Longsheng Bao,
Jiakang Zhao, and Ling Yu

InfraSegFM freezes the SAM image and prompt encoders, then learns
metadata- and style-conditioned routed residual adapters together with the SAM
mask decoder. The repository provides the public release code, the InfraSegFM
ViT-B checkpoint, paper-default configuration, synthetic demo data, and
entry points for training, labeled evaluation, and prediction.

## Highlights

- **Generalist infrastructure segmentation:** one checkpoint for heterogeneous
  asset types, acquisition platforms, materials, and defect tasks.
- **SAM-based parameter-efficient adaptation:** frozen SAM encoders with
  trainable routed residual adapters and mask decoder weights.
- **Metadata/style-conditioned routing:** platform, hierarchy, task, source
  proxy, and online style statistics condition sparse expert selection.
- **Research release utilities:** checkpoint manifest, model card, citation
  metadata, demo data, and train/evaluate/predict entry points.
- **Clear third-party boundary:** SAM source integration is documented and SAM
  weights are downloaded separately from the official Segment Anything release.

## Overview

<p align="center">
  <a href="assets/overview.png">
    <img src="assets/overview.png" alt="Overview of InfraSegDB and InfraSegFM" width="100%">
  </a>
</p>

InfraSegFM is evaluated in the manuscript on InfraSegDB, a 179,094 image-mask
benchmark curated from 40 datasets and six acquisition platforms. The public
repository includes a synthetic mini benchmark for software verification; the
full InfraSegDB benchmark is not bundled in this release.
Click a figure to open the full-resolution image.

**Source provenance footprint**

<p align="center">
  <a href="assets/source_map.png">
    <img src="assets/source_map.png" alt="Evidence-aware source provenance footprint of InfraSegDB" width="100%">
  </a>
</p>

**Benchmark taxonomy and composition**

<p align="center">
  <a href="assets/infrasegdb_taxonomy.png">
    <img src="assets/infrasegdb_taxonomy.png" alt="InfraSegDB taxonomy and benchmark statistics" width="100%">
  </a>
</p>

## Results

Full results from the revised manuscript's main reference evaluation, averaged
across three independent training runs:

| Setting | Target-training data | Image-mean Dice | Task-macro Dice | Protocol |
| --- | --- | ---: | ---: | --- |
| In-distribution | Source training | 0.7134 | 0.7158 | v1 |
| CrossSite | 0% | 0.7755 | 0.5747 | v1 |
| CrossTask | 0% | 0.5995 | 0.6331 | v2 |
| RealWorld | 0% | 0.5418 | 0.5993 | v1 |

Image-mean Dice averages per-image scores. Task-macro Dice first averages within
each recorded task-context group and then gives groups equal weight. Values are
shown to four decimal places.
CrossSite, CrossTask, and RealWorld rows use 0% target-training data. Protocols
v1/v2 identify the source-training/evaluation configuration; the replication
unit for this table is an independent training run.

## Prompt evaluation

The labeled reference evaluator, `tools/evaluate.py`, uses GT-Dice Oracle
selection. For finalized-quality-head selection and controlled spatial inputs,
use [`tools/run_prompt_protocol.py`](tools/run_prompt_protocol.py) with the
profile definitions in
[`configs/prompt_protocol_profiles.json`](configs/prompt_protocol_profiles.json).

The four finalized heads are supplied under
`checkpoints/prompt_protocol/inputs/heads/`. Full v1 and Shared v1 source
checkpoints are supplied under `checkpoints/prompt_protocol/inputs/models/`.
The profile manifest lists each model/head fingerprint and the required
compatibility assets. Follow
[`docs/PROMPT_EVALUATION.md`](docs/PROMPT_EVALUATION.md) to set up the model,
head, SAM backbone, and input directories.

Inspect the profiles and the command for a fixed full-image box:

```bash
python tools/run_prompt_protocol.py profiles
python tools/run_prompt_protocol.py full-image --workdir checkpoints/prompt_protocol --profile full_v1 --eval-root /path/to/eval --sam-checkpoint /path/to/sam_vit_b_01ec64.pth --output-dir runs/full_v1_full_image --dry-run
```

The same entry point exposes `prepare`, `preflight`, `evaluate`, `aggregate`,
and `empty` commands. The protocol guide explains the corresponding inputs and
candidate selectors.

## Release Contents

```text
InfraSegFM/
  assets/                         # selected manuscript figures
  checkpoints/
    infrasegfm_vit_b.pth          # InfraSegFM ViT-B release checkpoint
    manifest.json                 # size and SHA256 hash for the release checkpoint
  configs/
    paper_vit_b.json              # paper-default hyperparameters
  demo_data/
    manifest.csv                  # synthetic demo split/task manifest
    train/, val/                  # synthetic multi-task mini benchmark
  infrasegfm/
    model.py                      # InfraSegFM and routed MoE adapters
    build.py                      # model factories
    dataset.py                    # dataset loaders
    moe_preassign.py              # route-prior cache builder
    losses.py, metrics.py
    segment_anything/             # lightly adapted SAM source code
  tools/
    make_demo_dataset.py
    train.py
    evaluate.py
    predict.py
    run_prompt_protocol.py
  reproduction/prompt_runtime/      # frozen prompt-study implementation
  docs/PROMPT_EVALUATION.md
  CITATION.cff
  MODEL_CARD.md
  requirements.txt
  pyproject.toml
```

## Installation

Create an environment:

```bash
git clone https://github.com/YuyangBaoo/InfraSegFM.git
cd InfraSegFM

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
git clone https://github.com/YuyangBaoo/InfraSegFM.git
cd InfraSegFM

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch for your CUDA/CPU platform first. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

For CPU-only machines, install the official CPU build of PyTorch, then run
`pip install -r requirements.txt`.

## Checkpoints

The reference Full checkpoint and the prompt-protocol assets are supplied
under `checkpoints/`. Exact profile bindings are listed in
[`checkpoints/prompt_protocol/manifest.json`](checkpoints/prompt_protocol/manifest.json).

```text
checkpoints/
  infrasegfm_vit_b.pth
  manifest.json
  prompt_protocol/
    manifest.json
    inputs/models/                # supplied Full v1 and Shared v1 source checkpoints
    inputs/heads/                 # four finalized quality heads
```

Download the required SAM ViT-B checkpoint separately from Meta's official
Segment Anything release:

```bash
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Windows PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile "checkpoints\sam_vit_b_01ec64.pth"
```

Expected hashes:

| File | SHA256 |
| --- | --- |
| `checkpoints/infrasegfm_vit_b.pth` | `A6712C32BC9CB46465A689388DE4E6A611FBB7188ECE45150DB932019751A618` |
| `checkpoints/sam_vit_b_01ec64.pth` | `EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912` |

Verify locally:

```bash
# Linux/macOS
sha256sum checkpoints/*.pth

# Windows PowerShell
Get-FileHash checkpoints\*.pth -Algorithm SHA256
```

## Paper Defaults

The default command-line settings follow the original InfraSegFM Full workflow.

| Setting | Value |
| --- | --- |
| SAM backbone | `vit_b` |
| SAM input size | `256` |
| Training epochs | `30` |
| Fine-tuning epochs | `30` |
| Training batch size | `80` |
| Evaluation batch size | `50` |
| Pre-training LR | `5e-4` |
| Fine-tuning LR | `5e-5` |
| Weight decay | `0.01` |
| Mask decoder LR multiplier | `0.1` |
| Bottleneck dim / embedding dim | `16 / 16` |
| Experts | `4` |
| Gate top-k / temperature | `2 / 1.0` |
| MoE aux loss | load-balance `0.01`, entropy `0.01` |
| Route-prior supervision | enabled, coefficient `0.05` |
| AMP | recommended for full GPU runs |

The same record is saved in [configs/paper_vit_b.json](configs/paper_vit_b.json).

## Quick Smoke Test

The demo is a small synthetic mini benchmark for verifying the software path.
It is not used for paper metrics.

```text
demo_data/manifest.csv
train: 4 tasks x 12 samples
val:   4 tasks x 4 samples
```

Run a CPU smoke test:

```bash
python tools/make_demo_dataset.py --out demo_data --image_size 64 --overwrite
python tools/train.py --data_root demo_data --output_dir runs/demo --image_size 64 --epochs 1 --batch_size 8 --num_workers 0 --device cpu
python tools/evaluate.py --data_root demo_data/val --checkpoint runs/demo/model_best.pth --image_size 64 --batch_size 4 --num_workers 0 --device cpu --out_dir runs/demo_eval
```

Expected outputs:

```text
runs/demo/model_best.pth
runs/demo/history.csv
runs/demo_eval/metrics.csv
runs/demo_eval/summary.csv
```

After downloading the SAM ViT-B checkpoint, verify the release checkpoint on the
synthetic demo validation split:

```bash
python tools/evaluate.py \
  --data_root demo_data/val \
  --checkpoint checkpoints/infrasegfm_vit_b.pth \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --image_size 256 \
  --batch_size 1 \
  --num_workers 0 \
  --device cpu \
  --metrics dsc \
  --out_dir runs/demo_eval_release
```

## Dataset Format

Use one split directory and one task folder per defect type:

```text
your_dataset/
  train/
    Handheld_ConcreteCrack/
      npy_imgs/sample_001.npy
      npy_gts/sample_001.npy
    VehicleProfiler_AsphaltCrack/
      npy_imgs/sample_002.npy
      npy_gts/sample_002.npy
  val/
    Handheld_ConcreteCrack/
      npy_imgs/sample_101.npy
      npy_gts/sample_101.npy
```

Common image files are also supported:

```text
TaskFolder/
  images/sample_001.png
  masks/sample_001.png
```

Images and masks are paired by filename stem. Masks are binary; values greater
than zero are foreground.

Task folders should follow:

```text
<Platform>_<TaskName>
```

Examples:

```text
Handheld_ConcreteCrack
VehicleProfiler_AsphaltCrack
Handheld_ConcretePavementPothole
Aerial_RoadCrack
```

For new tasks or platforms, update [infrasegfm/taxonomy.py](infrasegfm/taxonomy.py).
The taxonomy supplies the metadata indices used by model inputs.

## Train

Paper-style training command:

```bash
python tools/train.py \
  --data_root /path/to/InfraSegDBv1 \
  --train_split train \
  --val_split val \
  --output_dir runs/InfraSegFM_vit_b \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --image_size 256 \
  --epochs 30 \
  --batch_size 80 \
  --num_workers 2 \
  --lr 5e-4 \
  --device cuda:0 \
  --amp
```

The route-prior cache is built automatically:

```text
/path/to/InfraSegDBv1/train/_moe_routing_cache.json
```

The cache is deterministic for a fixed dataset and ignored by git.

## Fine-tune

Fine-tune from the release checkpoint or from a checkpoint produced by training:

```bash
python tools/train.py \
  --data_root /path/to/InfraSegDBCrossTask \
  --train_split train5 \
  --val_split val \
  --resume checkpoints/infrasegfm_vit_b.pth \
  --output_dir runs/InfraSegFM_vit_b_finetune_CrossTask_train5 \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --image_size 256 \
  --epochs 30 \
  --batch_size 80 \
  --num_workers 2 \
  --lr 5e-5 \
  --device cuda:0 \
  --amp
```

## Evaluate Labeled Data

The reference evaluator selects the highest-Dice candidate using the evaluation
labels. Use this path for the labeled reference protocol:

```bash
python tools/evaluate.py \
  --data_root /path/to/your_dataset/val \
  --checkpoint checkpoints/infrasegfm_vit_b.pth \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --image_size 256 \
  --batch_size 50 \
  --num_workers 2 \
  --device cuda:0 \
  --metrics dsc \
  --out_dir runs/eval_val
```

Outputs:

```text
runs/eval_val/metrics.csv
runs/eval_val/summary.csv
```

## Predict Unlabeled Images

Use this path when users only have images and want masks. The script uses the
full image as the SAM box prompt by default. If you have object boxes from a
detector or manual annotation, pass `--box X1 Y1 X2 Y2`.

```bash
python tools/predict.py \
  --input /path/to/images \
  --output_dir runs/predict_my_images \
  --checkpoint checkpoints/infrasegfm_vit_b.pth \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --image_size 256 \
  --task_folder Handheld_ConcreteCrack \
  --device cuda:0
```

Predicted masks are written to:

```text
runs/predict_my_images/masks/
```

If your task is not in the taxonomy, add it to
[infrasegfm/taxonomy.py](infrasegfm/taxonomy.py) first.

## Checkpoint Format

Checkpoints produced by `tools/train.py` save adapter, context embedding, style
projection, gate, and mask decoder parameters using the release model's state
keys. The checkpoint identity is recorded in `checkpoints/manifest.json`.

## Reproducibility Checklist

When reporting results, record:

- Git commit hash.
- `configs/paper_vit_b.json`.
- SAM checkpoint name and SHA256.
- InfraSegFM checkpoint name and SHA256.
- Dataset split and task-folder taxonomy.
- PyTorch, CUDA, GPU model, and random seed.
- Exact training/evaluation command.

Keep these out of git:

```text
runs/
*_moe_routing_cache.json
private or full-scale datasets
```

## Input data

Prepare your image/mask pairs using the [dataset format](#dataset-format)
above. Obtain benchmark images and annotations from their original providers
under the applicable terms. The synthetic `demo_data/` provides a small input
set for checking the training, evaluation, and prediction commands.

## Third-party Code

The files under [infrasegfm/segment_anything](infrasegfm/segment_anything) are
lightly adapted from Meta AI's Segment Anything implementation. SAM is licensed
under Apache License 2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

SAM model weights are not redistributed in this repository. Download
`sam_vit_b_01ec64.pth` from the official Segment Anything release URL before
running the release checkpoint.

## License

InfraSegFM-specific source code, docs, demo generation scripts, and release
metadata are provided under the MIT License. SAM-derived files retain their
Apache-2.0 notice.

## Citation

GitHub renders citation metadata from [CITATION.cff](CITATION.cff).

```bibtex
@misc{li2026infrasegfm,
  title  = {Transferable Civil Infrastructure Defect Segmentation with Source-Aware Benchmarking and Routed SAM Adaptation},
  author = {Li, Baoxian and Bao, Yuyang and Chen, Si and Fang, Mingwei and Bao, Longsheng and Zhao, Jiakang and Yu, Ling},
  year   = {2026},
  note   = {Manuscript}
}
```

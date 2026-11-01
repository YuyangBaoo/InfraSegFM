# Prompt evaluation protocols

`tools/evaluate.py` retains the reference GT-Dice Oracle selector. The revision
prompt experiments use the separate, frozen entry point below.

## Fixed model and head assets

The four profile definitions in `configs/prompt_protocol_profiles.json` bind
each complete model to its finalized six-tensor quality head. Head epochs are
Full v1: 7, Full v2: 11, Shared v1: 2, Shared v2: 6. Place the recorded assets
under an isolated work directory:

```text
WORKDIR/
  inputs/models/full_v1_model_best.pth
  inputs/models/full_v2_model_best.pth
  inputs/models/shared_v1_model_best.pth
  inputs/models/shared_v2_model_best.pth
  inputs/heads/full_v1/quality_head_best.pt
  inputs/heads/full_v1/manifest.json
  inputs/heads/full_v2/quality_head_best.pt
  inputs/heads/shared_v1/quality_head_best.pt
  inputs/heads/shared_v2/quality_head_best.pt
  inputs/provenance/sam_b_256_interpolation_overlay.pt
  inputs/provenance/sam_b_256_interpolation_overlay.manifest.json
```

Checkpoint tensors use Git LFS. After cloning, fetch the supplied files with
`git lfs pull --include="checkpoints/prompt_protocol/**"`.

The release supplies the four finalized heads and the Full v1 companion
manifest in `checkpoints/prompt_protocol/inputs/heads/`. The hash-bound Full v1
and Shared v1 source models are in `checkpoints/prompt_protocol/inputs/models/`.
Point `--workdir` to `checkpoints/prompt_protocol` after placing the remaining
assets at the paths above.

| Profile | Source checkpoint in this repository | Finalized head in this repository |
| --- | --- | --- |
| Full v1 | `inputs/models/full_v1_model_best.pth` | `inputs/heads/full_v1/quality_head_best.pt` |
| Shared v1 | `inputs/models/shared_v1_model_best.pth` | `inputs/heads/shared_v1/quality_head_best.pt` |
| Full v2 | Supply the SHA256-bound file in the profile manifest | `inputs/heads/full_v2/quality_head_best.pt` |
| Shared v2 | Supply the SHA256-bound file in the profile manifest | `inputs/heads/shared_v2/quality_head_best.pt` |

Use the official SAM-B backbone separately. `profiles` prints all locked SHA256
bindings. `assets` checks local files before any model is imported; an absent
or mismatched finalized head fails explicitly. The original head is never used
as a substitute for a finalized head.

```bash
python tools/run_prompt_protocol.py profiles
python tools/run_prompt_protocol.py assets --workdir /path/to/WORKDIR --profile full_v1
```

The frozen runtime is in `reproduction/prompt_runtime`. Its six recorded
source fingerprints are checked by the driver. The formal-test environment
uses PyTorch 2.5.1, torchvision 0.20.1 and FP32. The hash-bound interpolation
compatibility payload restores nine frozen SAM tensors from the original
PyTorch 2.7.1 source construction. It is separate from the finalized heads.

## GT-derived boxes and coordinate perturbations

The evaluation root contains `ID/`, `ODD/CrossSite/`, `ODD/CrossTask/` and
`ODD/RealWorld/`. A task directory contains paired `npy_imgs/` and `npy_gts/`.
The source-project root retains the source-development splits used for the
three-sample migration preflight. The deterministic cache uses seed 26092001
and coordinate perturbations of 0, 5, 10 and 20 percent of the original box
width/height. Target training-data proportions remain a different quantity.

```bash
python tools/run_prompt_protocol.py prepare --workdir /path/to/WORKDIR --eval-root /path/to/eval
python tools/run_prompt_protocol.py preflight --workdir /path/to/WORKDIR --profile full_v1 --source-project-root /path/to/source_project --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
python tools/run_prompt_protocol.py evaluate --workdir /path/to/WORKDIR --profile full_v1 --eval-root /path/to/eval --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
```

Repeat preflight/evaluate for the other three fixed profiles, then run:

```bash
python tools/run_prompt_protocol.py aggregate --workdir /path/to/WORKDIR
```

Full/Shared v1 evaluates ID, CrossSite and RealWorld; v2 evaluates CrossTask.
One forward provides the same three candidates to fixed candidate 0,
original-head argmax, finalized-head argmax and offline same-candidate Oracle.
Only the Oracle uses GT to select a candidate. The finalized selector uses
the quality scores and keeps the saved masks unchanged.

## Full-image box and empty spatial prompt

The annotation-free spatial conditions use the same frozen profile and
finalized head. A fixed full-image box remains a box prompt. The empty
condition passes no points, boxes or masks to the prompt encoder and uses
batch size one. Labels are read after the forward for metric calculation.

```bash
python tools/run_prompt_protocol.py full-image --workdir /path/to/WORKDIR --profile full_v1 --eval-root /path/to/eval --sam-checkpoint /path/to/sam_vit_b_01ec64.pth --output-dir runs/full_v1_full_image
python tools/run_prompt_protocol.py empty --workdir /path/to/WORKDIR --profile full_v1 --eval-root /path/to/eval --sam-checkpoint /path/to/sam_vit_b_01ec64.pth --output-dir runs/full_v1_empty --batch-size 1
```

Choose the visible GPU with `CUDA_VISIBLE_DEVICES`. Add `--dry-run` to inspect
the exact command and current asset status without importing a model.
Image-mean Dice averages per-image values; task-macro Dice averages within
each actual task and then equally across tasks. The driver records all four
selection rules separately and keeps development selection fixed.

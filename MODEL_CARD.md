# InfraSegFM Model Card

## Model Details

InfraSegFM is a SAM-based segmentation framework for civil infrastructure
defects. The release checkpoint `checkpoints/infrasegfm_vit_b.pth` contains the
InfraSegFM adapter, context embedding, style projection, routing gate, and mask
decoder weights for the ViT-B configuration.

The SAM ViT-B backbone checkpoint is required at runtime but is not redistributed
in this repository. Download it from the official Segment Anything release URL
before evaluating or running inference.

## Intended Use

InfraSegFM is intended for research on infrastructure defect segmentation,
benchmarking, and limited deployment prototyping on labeled or box-prompted
inspection imagery. Typical targets include cracks, spalling, efflorescence,
corrosion, leakage, pavement distress, road markings, and related civil
infrastructure surface conditions.

## Out-of-Scope Use

The model should not be used as the sole basis for safety-critical maintenance,
structural capacity decisions, or public-risk decisions without qualified human
review, calibration on the target inspection protocol, and independent validation.

## Training and Evaluation Context

The accompanying manuscript describes InfraSegDB, a 179,094 image-mask benchmark
from 40 source datasets and six acquisition platforms. InfraSegFM is evaluated
under in-distribution, CrossSite, CrossTask, RealWorld, and few-shot settings.

Headline Dice scores reported in the manuscript:

| Setting | Dice |
| --- | ---: |
| In-distribution | 0.7134 |
| CrossSite zero-shot | 0.7771 |
| CrossTask zero-shot | 0.5955 |
| RealWorld zero-shot | 0.5420 |

The synthetic `demo_data/` included in this repository is only for software
smoke testing and is not used for reporting model performance.

## Limitations

- The release requires a box prompt or the default full-image box prompt.
- Performance can degrade under unseen sensors, unusual materials, severe motion
  blur, low contrast, occlusion, or annotation protocols that differ from the
  benchmark taxonomy.
- The release does not include the full InfraSegDB dataset.
- The model can output confident masks for visually similar non-defect patterns;
  downstream use should include inspection-specific quality control.

## Responsible Release Notes

Users should report the dataset split, task taxonomy, SAM checkpoint hash,
InfraSegFM checkpoint hash, PyTorch/CUDA versions, and exact command when
publishing results. See `README.md` for the reproducibility checklist.

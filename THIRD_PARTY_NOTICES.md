# Third-party notices

This repository includes a lightly adapted copy of Meta AI's Segment Anything
Model (SAM) implementation under `infrasegfm/segment_anything/`.

- Original project: https://github.com/facebookresearch/segment-anything
- License: Apache License 2.0
- Bundled license text: `LICENSES/Apache-2.0.txt`

The SAM-derived source files retain the original Meta copyright headers. Local
changes are limited to InfraSegFM integration details, including configurable
input resolution, mask-decoder output handling, and hooks needed by the
metadata/style-conditioned routed adapters.

SAM model weights are not redistributed in this repository. Users should
download the required SAM ViT-B checkpoint from the official Segment Anything
release location:

```text
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The InfraSegFM checkpoint is provided separately by Baoxian Li, Yuyang Bao,
Si Chen, Mingwei Fang, Longsheng Bao, Jiakang Zhao, and Ling Yu.

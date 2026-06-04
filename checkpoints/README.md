# Checkpoints

This directory contains the InfraSegFM release checkpoint:

```text
infrasegfm_vit_b.pth      # InfraSegFM ViT-B release checkpoint
manifest.json             # file size and SHA256 checksum
```

The SAM ViT-B checkpoint is required at runtime but is not redistributed in this
repository. Download it from the official Segment Anything release location:

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

Expected SHA256:

```text
infrasegfm_vit_b.pth    A6712C32BC9CB46465A689388DE4E6A611FBB7188ECE45150DB932019751A618
sam_vit_b_01ec64.pth   EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912
```

The InfraSegFM `.pth` file is larger than 50 MiB. Use Git LFS or a GitHub
Release asset when publishing this repository:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes checkpoints/infrasegfm_vit_b.pth checkpoints/manifest.json
```

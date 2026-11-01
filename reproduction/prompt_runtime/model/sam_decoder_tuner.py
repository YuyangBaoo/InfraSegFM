# -*- coding: utf-8 -*-
"""SAM decoder fine-tuning wrapper.

Lightweight baseline:
- Freeze SAM image encoder + prompt encoder
- Train / load only the mask decoder weights

Used as a simple baseline for infrastructure defect segmentation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SAMDecoderTuner(nn.Module):
    """Freeze SAM encoders and fine-tune the mask decoder."""

    def __init__(self, sam: nn.Module):
        super().__init__()

        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        for p in sam.prompt_encoder.parameters():
            p.requires_grad = False

        self.sam = sam

    def forward(self, data: Dict):
        img, box = data["img"], data["box"]

        if box.ndim == 2:
            box = box[:, None, :]

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )

        input_image = self.sam.preprocess(img)
        enc_out = self.sam.image_encoder(input_image)
        image_embedding = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        return mask_predictions

    def save_parameters(self) -> Dict[str, torch.Tensor]:
        state_dict = self.sam.state_dict()
        return {k: v for k, v in state_dict.items() if "mask_decoder" in k}

    def load_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        cur = self.sam.state_dict()
        load_dict = {}
        for k, v in cur.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                load_dict[k] = state_dict[k]
        cur.update(load_dict)
        self.sam.load_state_dict(cur, strict=False)

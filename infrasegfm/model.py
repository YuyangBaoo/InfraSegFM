"""InfraSegFM model definition."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .segment_anything.modeling import Sam
from .segment_anything.modeling.common import MLPBlock
from .taxonomy import (
    asset_level_1_map,
    asset_level_2_map,
    asset_level_3_map,
    platform_map,
    task_list,
)


class MoEAdapterMLPBlock(nn.Module):
    """Mixture-of-experts adapter injected into a SAM ViT MLP block."""

    def __init__(
        self,
        mlp: MLPBlock,
        embedding_dim: int = 16,
        bottleneck_dim: int = 16,
        expert_num: int = 4,
        gate_topk: int = 2,
        gate_temperature: float = 1.0,
        gate_noise: float = 0.0,
        style_use_std: bool = True,
        style_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.hidden_dim = int(getattr(mlp, "embedding_dim", mlp.lin1.in_features))
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.expert_num = int(expert_num)
        self.gate_topk = int(gate_topk)
        self.gate_temperature = float(gate_temperature)
        self.gate_noise = float(gate_noise)
        self.style_use_std = bool(style_use_std)

        self.adapter_down = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.GELU(),
        )
        self.adapter_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.bottleneck_dim, self.hidden_dim),
                )
                for _ in range(self.expert_num)
            ]
        )
        self.adapter_gate = nn.Linear(self.embedding_dim * 4, self.expert_num, bias=True)
        self.style_proj = nn.Sequential(
            nn.Linear(self.bottleneck_dim * 2, self.embedding_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(p=float(style_dropout)) if style_dropout and style_dropout > 0 else nn.Identity()

        nn.init.zeros_(self.adapter_gate.weight)
        nn.init.zeros_(self.adapter_gate.bias)
        with torch.no_grad():
            e = self.embedding_dim
            self.adapter_gate.weight[:, : 3 * e].normal_(mean=0.0, std=0.02)
            self.adapter_gate.weight[:, : 3 * e].mul_(0.1)

    def _compute_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(1, 2))
        if self.style_use_std:
            spread = x.std(dim=(1, 2), unbiased=False)
        else:
            spread = (x - mu[:, None, None, :]).abs().mean(dim=(1, 2))
        return self.style_proj(torch.cat([mu, spread], dim=-1))

    def _apply_topk(self, probs: torch.Tensor) -> torch.Tensor:
        k = self.gate_topk
        if k <= 0 or k >= probs.shape[1]:
            return probs
        _, top_idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=top_idx, value=1.0)
        probs = probs * mask
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    def forward(
        self,
        x: torch.Tensor,
        platform_embed: torch.Tensor,
        context_embed: Tuple[torch.Tensor, ...],
        force_expert: Optional[torch.Tensor] = None,
    ):
        del force_expert
        base = self.mlp(x)
        x_bn = self.adapter_down(x)

        task_e = context_embed[4]
        source_e = context_embed[5]
        style_e = self.dropout(self._compute_style_embedding(x_bn))
        gate_input = torch.cat([task_e, source_e, platform_embed, style_e], dim=-1)

        logits = self.adapter_gate(gate_input / max(self.gate_temperature, 1e-6))
        if self.training and self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise
        probs = self._apply_topk(torch.softmax(logits, dim=-1))

        adapter = 0.0
        for expert_idx, expert in enumerate(self.adapter_up):
            adapter = adapter + probs[:, expert_idx].view(-1, 1, 1, 1) * expert(x_bn)
        return base + adapter, probs


class InfraSegFM(nn.Module):
    """SAM with hierarchical task/source/style conditioned MoE adapters."""

    def __init__(
        self,
        sam: Sam,
        bottleneck_dim: int,
        embedding_dim: int,
        expert_num: int,
        pos: Optional[Sequence[int]] = None,
        gate_topk: int = 2,
        gate_temperature: float = 1.0,
        gate_noise_std: float = 0.0,
        style_use_std: bool = True,
        style_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0 or embedding_dim <= 0 or expert_num <= 0:
            raise ValueError("bottleneck_dim, embedding_dim, and expert_num must be positive.")

        self.pos = list(pos) if pos is not None else list(range(len(sam.image_encoder.blocks)))
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

        dataset_buckets = int(os.getenv("MOE_DATASET_BUCKETS", "1024"))
        sam.image_encoder.platform_embed = nn.Embedding(len(platform_map), embedding_dim)
        sam.image_encoder.context_embed = nn.ModuleList(
            [
                nn.Embedding(1, embedding_dim),
                nn.Embedding(len(asset_level_1_map), embedding_dim),
                nn.Embedding(len(asset_level_2_map), embedding_dim),
                nn.Embedding(len(asset_level_3_map), embedding_dim),
                nn.Embedding(len(task_list) + 1, embedding_dim),
                nn.Embedding(dataset_buckets, embedding_dim),
            ]
        )
        nn.init.normal_(sam.image_encoder.platform_embed.weight, mean=0.0, std=0.02)
        for emb in sam.image_encoder.context_embed:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        for idx, block in enumerate(sam.image_encoder.blocks):
            if idx in self.pos:
                block.mlp = MoEAdapterMLPBlock(
                    block.mlp,
                    embedding_dim=embedding_dim,
                    bottleneck_dim=bottleneck_dim,
                    expert_num=expert_num,
                    gate_topk=gate_topk,
                    gate_temperature=gate_temperature,
                    gate_noise=gate_noise_std,
                    style_use_std=style_use_std,
                    style_dropout=style_dropout,
                )

        self.sam = sam

    def save_parameters(self) -> Dict[str, torch.Tensor]:
        keep: Dict[str, torch.Tensor] = {}
        for key, value in self.sam.state_dict().items():
            if (
                "adapter_" in key
                or "adapter_gate" in key
                or "style_proj" in key
                or "platform_embed" in key
                or "context_embed" in key
                or "mask_decoder" in key
            ):
                keep[key] = value.detach().cpu()
        return keep

    def load_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        current = self.sam.state_dict()
        loadable = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        current.update(loadable)
        self.sam.load_state_dict(current, strict=False)

    def _context_embeddings(self, data: Dict, img: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size = img.shape[0]

        def clamp_ids(x: torch.Tensor, max_n: int) -> torch.Tensor:
            return x.to(device=img.device, dtype=torch.long).clamp(min=0, max=max_n - 1)

        platform_ids = clamp_ids(data["platform"], self.sam.image_encoder.platform_embed.num_embeddings)
        platform_e = self.sam.image_encoder.platform_embed(platform_ids)

        context = data["context"]
        if len(context) == 4:
            l1, l2, l3, task = context
            source_bucket = torch.zeros(batch_size, dtype=torch.long, device=img.device)
        elif len(context) == 5:
            l1, l2, l3, task, source_bucket = context
        else:
            raise ValueError(f"Expected context length 4 or 5, got {len(context)}")

        root = torch.zeros(batch_size, dtype=torch.long, device=img.device)
        ctx0 = self.sam.image_encoder.context_embed[0](root)
        ctx1 = self.sam.image_encoder.context_embed[1](clamp_ids(l1, self.sam.image_encoder.context_embed[1].num_embeddings))
        ctx2 = self.sam.image_encoder.context_embed[2](clamp_ids(l2, self.sam.image_encoder.context_embed[2].num_embeddings))
        ctx3 = self.sam.image_encoder.context_embed[3](clamp_ids(l3, self.sam.image_encoder.context_embed[3].num_embeddings))

        task = task.to(device=img.device, dtype=torch.long)
        task_emb_n = self.sam.image_encoder.context_embed[4].num_embeddings
        unknown = torch.full_like(task, task_emb_n - 1)
        task = torch.where((task >= 0) & (task < task_emb_n - 1), task, unknown)
        ctx4 = self.sam.image_encoder.context_embed[4](task)

        source_bucket = clamp_ids(source_bucket, self.sam.image_encoder.context_embed[5].num_embeddings)
        ctx5 = self.sam.image_encoder.context_embed[5](source_bucket)
        return platform_e, (ctx0, ctx1, ctx2, ctx3, ctx4, ctx5)

    def forward(self, data: Dict, return_gates: bool = False, return_iou: bool = False):
        img = data["img"]
        box = data["box"]
        platform_e, context_e = self._context_embeddings(data, img)

        if box.ndim == 2:
            box = box[:, None, :]
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=box, masks=None)

        input_image = self.sam.preprocess(img)
        enc_out = self.sam.image_encoder(input_image, platform_e, context_e)
        if isinstance(enc_out, tuple):
            image_embedding, gates = enc_out
        else:
            image_embedding, gates = enc_out, []

        masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        if return_gates or return_iou:
            out = {"masks": masks}
            if return_gates:
                out["gates"] = gates
            if return_iou:
                out["iou_predictions"] = iou_predictions
            return out
        return masks

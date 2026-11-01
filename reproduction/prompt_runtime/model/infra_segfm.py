# -*- coding: utf-8 -*-
"""InfraSegFM: hierarchical context + platform conditioned MoE adapter for SAM."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.common import MLPBlock

from data.datainfo import (
    platform_map,
    asset_level_1_map,
    asset_level_2_map,
    asset_level_3_map,
    task_list,
)


class MoEAdapterMLPBlock(nn.Module):
    """MoE adapter injected into each ViT MLP block."""

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
        disable_style: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = mlp

        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.expert_num = int(expert_num)

        self.gate_topk = int(gate_topk) if gate_topk is not None else 0
        self.gate_temperature = float(gate_temperature) if gate_temperature is not None else 1.0
        self.gate_noise = float(gate_noise) if gate_noise is not None else 0.0
        self.style_use_std = bool(style_use_std)
        self.style_dropout = float(style_dropout) if style_dropout is not None else 0.0
        self.disable_style = bool(disable_style)

        self.adapter_down = nn.Sequential(
            nn.Linear(768, self.bottleneck_dim),
            nn.GELU(),
        )

        self.adapter_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.bottleneck_dim, 768),
                )
                for _ in range(self.expert_num)
            ]
        )

        self.adapter_gate = nn.Linear(self.embedding_dim * 4, self.expert_num, bias=True)

        self.style_proj = nn.Sequential(
            nn.Linear(self.bottleneck_dim * 2, self.embedding_dim),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(p=self.style_dropout) if self.style_dropout > 0 else nn.Identity()

        nn.init.zeros_(self.adapter_gate.weight)
        nn.init.zeros_(self.adapter_gate.bias)
        with torch.no_grad():
            e = self.embedding_dim
            # input order: [task | source | platform | style]
            self.adapter_gate.weight[:, : 3 * e].normal_(mean=0.0, std=0.02)
            self.adapter_gate.weight[:, : 3 * e].mul_(0.1)
            self.adapter_gate.weight[:, 3 * e :].zero_()

    @staticmethod
    def _zeros_like(ref: torch.Tensor, d: int) -> torch.Tensor:
        return torch.zeros((ref.shape[0], d), device=ref.device, dtype=ref.dtype)

    def _compute_style_embed(self, x_bn: torch.Tensor) -> torch.Tensor:
        mu = x_bn.mean(dim=(1, 2))
        if self.style_use_std:
            sig = x_bn.std(dim=(1, 2), unbiased=False)
        else:
            sig = (x_bn - mu[:, None, None, :]).abs().mean(dim=(1, 2))
        feat = torch.cat([mu, sig], dim=-1)
        return self.style_proj(feat)

    def _apply_topk(self, probs: torch.Tensor) -> torch.Tensor:
        k = self.gate_topk
        if k is None or k <= 0 or k >= probs.shape[1]:
            return probs
        _, topk_idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
        return probs

    def forward(
        self,
        x: torch.Tensor,
        platform_embed: torch.Tensor,
        context_embed: Tuple[torch.Tensor, ...],
        force_expert: Optional[torch.Tensor] = None,
    ):
        x_res = self.mlp(x)
        x_bn = self.adapter_down(x)

        task_e = context_embed[4] if len(context_embed) > 4 else self._zeros_like(platform_embed, self.embedding_dim)
        src_e = context_embed[5] if len(context_embed) > 5 else self._zeros_like(platform_embed, self.embedding_dim)

        style_e = self._compute_style_embed(x_bn)
        style_e = self.dropout(style_e)
        if self.disable_style:
            style_e = torch.zeros_like(style_e)

        gate_in = torch.cat([task_e, src_e, platform_embed, style_e], dim=-1)
        temp = max(1e-6, self.gate_temperature)
        logits = self.adapter_gate(gate_in / temp)
        if self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise

        probs = torch.softmax(logits, dim=-1)
        probs = self._apply_topk(probs)

        probs_raw = probs
        probs_used = probs_raw
        if force_expert is not None:
            try:
                fe = force_expert
                if not torch.is_tensor(fe):
                    fe = torch.as_tensor(fe, device=probs_raw.device)
                fe = fe.to(device=probs_raw.device, dtype=torch.long).view(-1)
                if fe.numel() == probs_raw.shape[0]:
                    m = fe >= 0
                    if m.any():
                        forced = torch.zeros_like(probs_raw[m])
                        forced.scatter_(1, fe[m].unsqueeze(1).clamp(0, self.expert_num - 1), 1.0)
                        probs_used = probs_raw.clone()
                        probs_used[m] = forced
            except Exception:
                probs_used = probs_raw

        out = 0.0
        for i in range(self.expert_num):
            out = out + probs_used[:, i].view(-1, 1, 1, 1) * self.adapter_up[i](x_bn)

        x_out = x_res + out
        return x_out, probs_raw


def matched_shared_hidden_dim(bottleneck_dim: int, embedding_dim: int, expert_num: int) -> int:
    """Hidden width for a single shared adapter whose up-path parameters match
    the routed MoE's extra parameters (experts + gate + style_proj).

    MoE extra params per block:
      experts: E * (bn*bn + bn + bn*768 + 768)
      gate:    (4*emb) * E + E
      style:   (2*bn) * emb + emb
    Shared up-path with hidden width W: W*(bn + 1 + 768) + 768
    """
    moe_up = expert_num * (
        bottleneck_dim * bottleneck_dim + bottleneck_dim
        + bottleneck_dim * 768 + 768
    )
    gate = (embedding_dim * 4) * expert_num + expert_num
    style = (bottleneck_dim * 2) * embedding_dim + embedding_dim
    target = moe_up + gate + style
    return max(1, round((target - 768) / (bottleneck_dim + 1 + 768)))


class SharedAdapterMLPBlock(nn.Module):
    """Parameter-matched shared-adapter control: one wide bottleneck adapter.

    No gate, no style branch, no routing. Interface-compatible with
    MoEAdapterMLPBlock; returns gate=None so the encoder collects no gates.
    """

    def __init__(
        self,
        mlp: MLPBlock,
        bottleneck_dim: int = 16,
        hidden_dim: int = 68,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dim = int(hidden_dim)

        self.adapter_down = nn.Sequential(
            nn.Linear(768, self.bottleneck_dim),
            nn.GELU(),
        )
        self.adapter_up = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 768),
        )

    def forward(
        self,
        x: torch.Tensor,
        platform_embed: Optional[torch.Tensor] = None,
        context_embed: Optional[Tuple[torch.Tensor, ...]] = None,
        force_expert: Optional[torch.Tensor] = None,
    ):
        x_res = self.mlp(x)
        out = self.adapter_up(self.adapter_down(x))
        return x_res + out, None


class MedSAMLPBlock(nn.Module):
    """Med-SA style MLP-side adapter (Wu et al., 2023): out = mlp(x) + scale * adapter(x).

    The adapter has no inner skip connection (skip_connect=False in the official
    implementation); Block adds the residual outside. Returns gate=None.
    """

    def __init__(self, mlp: MLPBlock, dim: int = 768, mlp_ratio: float = 0.25, scale: float = 0.5) -> None:
        super().__init__()
        self.mlp = mlp
        self.scale = float(scale)
        hidden = max(1, int(dim * mlp_ratio))
        self.adapter_mlp_medsa = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init the up-projection so the adapter starts as a no-op (identity),
        # letting training depart smoothly from the pretrained frozen SAM instead
        # of fighting injected noise at every block.
        nn.init.zeros_(self.adapter_mlp_medsa[-1].weight)
        nn.init.zeros_(self.adapter_mlp_medsa[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        platform_embed: Optional[torch.Tensor] = None,
        context_embed: Optional[Tuple[torch.Tensor, ...]] = None,
        force_expert: Optional[torch.Tensor] = None,
    ):
        return self.mlp(x) + self.scale * self.adapter_mlp_medsa(x), None


class MedSAAttnWrapper(nn.Module):
    """Med-SA space adapter applied right after attention (with skip connect)."""

    def __init__(self, attn: nn.Module, dim: int = 768, mlp_ratio: float = 0.25) -> None:
        super().__init__()
        self.attn = attn
        hidden = max(1, int(dim * mlp_ratio))
        self.adapter_space_medsa = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init up-projection: adapter starts as identity (see MedSAMLPBlock).
        nn.init.zeros_(self.adapter_space_medsa[-1].weight)
        nn.init.zeros_(self.adapter_space_medsa[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return x + self.adapter_space_medsa(x)


class LoRAQKVWrapper(nn.Module):
    """SAMed-style LoRA (Zhang & Liu, 2023) on the q and v projections of SAM's
    fused qkv linear. A matrices kaiming-init, B matrices zero-init (identity at
    start). k projection is left untouched, matching the official SAMed code.
    """

    def __init__(self, qkv: nn.Linear, r: int = 4) -> None:
        super().__init__()
        self.qkv = qkv
        self.dim = int(qkv.in_features)
        self.r = int(r)
        self.adapter_lora_a_q = nn.Linear(self.dim, self.r, bias=False)
        self.adapter_lora_b_q = nn.Linear(self.r, self.dim, bias=False)
        self.adapter_lora_a_v = nn.Linear(self.dim, self.r, bias=False)
        self.adapter_lora_b_v = nn.Linear(self.r, self.dim, bias=False)
        import math as _math
        nn.init.kaiming_uniform_(self.adapter_lora_a_q.weight, a=_math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter_lora_a_v.weight, a=_math.sqrt(5))
        nn.init.zeros_(self.adapter_lora_b_q.weight)
        nn.init.zeros_(self.adapter_lora_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x)
        dq = self.adapter_lora_b_q(self.adapter_lora_a_q(x))
        dv = self.adapter_lora_b_v(self.adapter_lora_a_v(x))
        return out + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)


class InfraSegFM(nn.Module):
    """Apply a hierarchical-context MoE adapter to SAM's ViT image encoder."""

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
        disable_style: bool = False,
        adapter_mode: str = "moe",
        task_drop_p: float = 0.0,
    ):
        super().__init__()

        if bottleneck_dim <= 0 or embedding_dim <= 0 or expert_num <= 0:
            raise ValueError("bottleneck_dim/embedding_dim/expert_num must be > 0")

        self.adapter_mode = str(adapter_mode or "moe").lower().strip()
        if self.adapter_mode not in ("moe", "shared", "lora", "medsa"):
            raise ValueError(f"Unknown adapter_mode: {self.adapter_mode}")
        # Task-identity dropout: during training, replace the task id with the
        # unknown bucket for a random subset of samples so the gate learns to
        # route from style/platform alone (open-set unseen-task transfer).
        self.task_drop_p = float(task_drop_p or 0.0)

        self.pos = list(pos) if pos else list(range(len(sam.image_encoder.blocks)))

        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        for p in sam.prompt_encoder.parameters():
            p.requires_grad = False

        num_platforms = len(platform_map)
        num_l1 = len(asset_level_1_map)
        num_l2 = len(asset_level_2_map)
        num_l3 = len(asset_level_3_map)
        num_tasks = len(task_list)
        dataset_buckets = int(os.getenv("MOE_DATASET_BUCKETS", "1024"))

        sam.image_encoder.platform_embed = nn.Embedding(num_platforms, embedding_dim)
        sam.image_encoder.context_embed = nn.ModuleList(
            [
                nn.Embedding(1, embedding_dim),
                nn.Embedding(num_l1, embedding_dim),
                nn.Embedding(num_l2, embedding_dim),
                nn.Embedding(num_l3, embedding_dim),
                nn.Embedding(num_tasks + 1, embedding_dim),
                nn.Embedding(dataset_buckets, embedding_dim),
            ]
        )

        nn.init.normal_(sam.image_encoder.platform_embed.weight, mean=0.0, std=0.02)
        for emb in sam.image_encoder.context_embed:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        shared_hidden = matched_shared_hidden_dim(bottleneck_dim, embedding_dim, expert_num)
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue
            if self.adapter_mode == "shared":
                blk.mlp = SharedAdapterMLPBlock(
                    blk.mlp,
                    bottleneck_dim=bottleneck_dim,
                    hidden_dim=shared_hidden,
                )
            elif self.adapter_mode == "medsa":
                blk.attn = MedSAAttnWrapper(blk.attn, dim=768, mlp_ratio=0.25)
                blk.mlp = MedSAMLPBlock(blk.mlp, dim=768, mlp_ratio=0.25, scale=0.5)
            elif self.adapter_mode == "lora":
                blk.attn.qkv = LoRAQKVWrapper(blk.attn.qkv, r=4)
            else:
                blk.mlp = MoEAdapterMLPBlock(
                    blk.mlp,
                    embedding_dim=embedding_dim,
                    bottleneck_dim=bottleneck_dim,
                    expert_num=expert_num,
                    gate_topk=gate_topk,
                    gate_temperature=gate_temperature,
                    gate_noise=gate_noise_std,
                    style_use_std=style_use_std,
                    style_dropout=style_dropout,
                    disable_style=disable_style,
                )

        if self.adapter_mode != "moe":
            # Conditioning embeddings are unused without a gate: freeze them so
            # trainable-parameter accounting stays clean.
            sam.image_encoder.platform_embed.weight.requires_grad = False
            for emb in sam.image_encoder.context_embed:
                emb.weight.requires_grad = False

        self.sam = sam

    def save_parameters(self) -> Dict[str, torch.Tensor]:
        state_dict = self.sam.state_dict()
        keep: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if (
                ("adapter_" in k)
                or ("adapter_gate" in k)
                or ("style_proj" in k)
                or ("platform_embed" in k)
                or ("context_embed" in k)
                or ("mask_decoder" in k)
            ):
                keep[k] = v
        return keep

    def load_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load adapter + embeddings + mask decoder (shape-safe, partial allowed).

        The loader is tolerant to older checkpoints whose embedding parameter
        names differ, by matching on tensor shapes under the image encoder.
        """

        cur = self.sam.state_dict()
        # Shape-based fallback for encoder conditioning embeddings
        def _find_emb_key(target_shape, prefer_suffix=None):
            cands = []
            for k, v in state_dict.items():
                if (not isinstance(v, torch.Tensor)) or v.shape != target_shape:
                    continue
                if ("image_encoder" not in k) or ("embed" not in k) or (not k.endswith(".weight")):
                    continue
                cands.append(k)
            if not cands:
                return None
            if prefer_suffix:
                for k in cands:
                    if k.endswith(prefer_suffix):
                        return k
            return cands[0]

        # Fill missing new keys from any compatible legacy keys (if present)
        plat_k = "image_encoder.platform_embed.weight"
        if plat_k in cur and plat_k not in state_dict:
            cand = _find_emb_key(cur[plat_k].shape)
            if cand is not None:
                state_dict = dict(state_dict)
                state_dict[plat_k] = state_dict[cand]

        for i in range(6):
            k = f"image_encoder.context_embed.{i}.weight"
            if k in cur and k not in state_dict:
                cand = _find_emb_key(cur[k].shape, prefer_suffix=f".{i}.weight")
                if cand is not None:
                    state_dict[k] = state_dict[cand]

        load_dict = {}
        for k, v in cur.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                load_dict[k] = state_dict[k]
        cur.update(load_dict)
        self.sam.load_state_dict(cur, strict=False)

    def forward(self, data: Dict, return_gates: bool = False):
        img, box = data["img"], data["box"]
        platform_ids = data["platform"]
        context_ids = data["context"]

        b = img.shape[0]

        def _clamp_ids(x: torch.Tensor, max_n: int) -> torch.Tensor:
            x = x.to(device=img.device, dtype=torch.long)
            return x.clamp(min=0, max=max_n - 1)

        platform_ids = _clamp_ids(platform_ids, self.sam.image_encoder.platform_embed.num_embeddings)
        platform_e = self.sam.image_encoder.platform_embed(platform_ids)

        if len(context_ids) == 4:
            l1, l2, l3, task_id = context_ids
            src_bucket = None
        elif len(context_ids) == 5:
            l1, l2, l3, task_id, src_bucket = context_ids
        else:
            raise ValueError(f"Invalid context tuple length: {len(context_ids)}")

        root_idx = torch.zeros(b, dtype=torch.long, device=img.device)
        ctx0 = self.sam.image_encoder.context_embed[0](root_idx)

        l1 = _clamp_ids(l1, self.sam.image_encoder.context_embed[1].num_embeddings)
        l2 = _clamp_ids(l2, self.sam.image_encoder.context_embed[2].num_embeddings)
        l3 = _clamp_ids(l3, self.sam.image_encoder.context_embed[3].num_embeddings)
        ctx1 = self.sam.image_encoder.context_embed[1](l1)
        ctx2 = self.sam.image_encoder.context_embed[2](l2)
        ctx3 = self.sam.image_encoder.context_embed[3](l3)

        task_emb_n = self.sam.image_encoder.context_embed[4].num_embeddings
        task_id = task_id.to(device=img.device, dtype=torch.long)
        task_id = torch.where(
            (task_id >= 0) & (task_id < task_emb_n - 1),
            task_id,
            torch.full_like(task_id, task_emb_n - 1),
        )
        if self.training and self.task_drop_p > 0:
            drop = torch.rand(task_id.shape[0], device=task_id.device) < self.task_drop_p
            task_id = torch.where(drop, torch.full_like(task_id, task_emb_n - 1), task_id)
        ctx4 = self.sam.image_encoder.context_embed[4](task_id)

        if src_bucket is None:
            ctx5 = torch.zeros_like(ctx4)
        else:
            src_bucket = _clamp_ids(src_bucket, self.sam.image_encoder.context_embed[5].num_embeddings)
            ctx5 = self.sam.image_encoder.context_embed[5](src_bucket)

        context_e = (ctx0, ctx1, ctx2, ctx3, ctx4, ctx5)

        if box.ndim == 2:
            box = box[:, None, :]
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=box, masks=None)

        input_image = self.sam.preprocess(img)
        enc_out = self.sam.image_encoder(
            input_image,
            platform_e,
            context_e,
            force_expert=data.get("force_expert", None),
        )
        if isinstance(enc_out, tuple):
            image_embedding, gate_list = enc_out
        else:
            image_embedding, gate_list = enc_out, []

        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        if return_gates:
            return {"masks": mask_predictions, "gates": gate_list}
        return mask_predictions

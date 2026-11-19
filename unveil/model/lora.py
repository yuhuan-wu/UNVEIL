"""The shared LoRA branch: trained copies of SAM 3's last ViT blocks.

ViT blocks 0..first-1 stay shared and frozen. Blocks first..31 are deep-copied, and LoRA is added to the copies'
attention (``qkv``, ``proj``) and MLP (``fc1``, ``fc2``) layers. SAM 3 keeps its own blocks, so its native outputs do not
change. After training the update is merged into plain copies (``merged_blocks``), so a forward pass through the branch
costs what SAM 3's own blocks cost.

LoRA: y = W x + b + (alpha / r) B A x, A Kaiming-uniform (a = sqrt(5)), B zero (the branch starts at SAM 3's function),
no dropout. The copies always run in eval mode (no drop path).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

FIRST, DEPTH = 24, 32
TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def vit_trunk(sam3) -> nn.Module:
    return sam3.backbone.vision_backbone.trunk


def _get(mod: nn.Module, path: str) -> nn.Module:
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set(mod: nn.Module, path: str, new: nn.Module) -> None:
    *head, last = path.split(".")
    for p in head:
        mod = getattr(mod, p)
    setattr(mod, last, new)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t().to(x.dtype)) @ (self.lora_B.t() * self.scale).to(x.dtype)


def merge_weight(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> torch.Tensor:
    """W + scale B A, computed in fp32 and cast back to W's dtype."""
    return (weight.float() + (lora_b.float() @ lora_a.float()) * scale).to(weight.dtype)


class LoRABranch(nn.Module):
    """Copies of ViT blocks first..31 with LoRA, for training. ``lora_state`` / ``load_lora`` use the keys
    ``blocks.<k>.<target>.lora_A|lora_B`` (k = 0 for block ``first``)."""

    def __init__(self, sam3, r: int = 16, alpha: float = 32.0, first: int = FIRST):
        super().__init__()
        self.first, self.r, self.alpha = first, r, alpha
        self.blocks = nn.ModuleList(copy.deepcopy(b) for b in vit_trunk(sam3).blocks[first:])
        self.blocks.requires_grad_(False)
        for b in self.blocks:
            for t in TARGETS:
                _set(b, t, LoRALinear(_get(b, t), r, alpha))
        self.checkpointing = True
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)                 # the copied blocks never use drop path

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def lora_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora(self, state: dict[str, torch.Tensor]) -> None:
        own = {k for k in self.state_dict() if "lora_" in k}
        if own != set(state):
            raise KeyError(f"LoRA keys differ: {sorted(own ^ set(state))[:4]}")
        self.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = ckpt.checkpoint(b, x, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else b(x)
        return x


def merged_blocks(sam3, lora: dict[str, torch.Tensor], r: int, alpha: float, first: int = FIRST) -> nn.Sequential:
    """Inference copies of blocks first..31 with W + (alpha / r) B A, merged on the model's device in fp32.

    lora: ``blocks.<k>.<target>.lora_A|lora_B`` tensors (``LoRABranch.lora_state``).
    """
    out = nn.Sequential(*(copy.deepcopy(b) for b in vit_trunk(sam3).blocks[first:]))
    used = set()
    with torch.no_grad():
        for k, blk in enumerate(out):
            for t in TARGETS:
                lin = _get(blk, t)
                ka, kb = f"blocks.{k}.{t}.lora_A", f"blocks.{k}.{t}.lora_B"
                dev = lin.weight.device
                lin.weight.copy_(merge_weight(lin.weight, lora[ka].to(dev), lora[kb].to(dev), alpha / r))
                used.update((ka, kb))
    if set(lora) != used:
        raise KeyError(f"unexpected LoRA tensors: {sorted(set(lora) - used)[:4]}")
    return out.eval().requires_grad_(False)


def n_lora_params(r: int = 16, dim: int = 1024, mlp: int = 4736, n_blocks: int = DEPTH - FIRST) -> int:
    """A + B over qkv (dim -> 3 dim), proj (dim -> dim), fc1 (dim -> mlp) and fc2 (mlp -> dim)."""
    return n_blocks * r * ((dim + 3 * dim) + 2 * dim + 2 * (dim + mlp))

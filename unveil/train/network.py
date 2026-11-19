"""The network UNVEIL trains, on top of a frozen SAM 3.

    prefix tokens (frozen ViT blocks 0..23, computed outside, no gradients)
      -> LoRA branch (copies of blocks 24..31) -> ln_post -> SAM 3's neck -> prompt assembly
      -> ConceptHeads: fusion encoder -> semantic path (map) ; presence decoder (stop-gradient inputs)

The heads are SAM 3's own modules, updated in place; the checkpoint stores them as a separate copy, so the deployed
model runs them next to an unchanged SAM 3. The convention vectors (one 256-d vector per member of the joint pair) are
added to the text tokens of the USC12K pairs only. Those pairs run the fusion encoder twice: their maps read the memory
computed with the vectors, their presence logits the memory without them, so presence training sees the plain prompt
for every pair and the vectors get gradients only from the dense loss of the USC12K pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from unveil.model import backbone as B
from unveil.model.heads import ConceptHeads, coarse_to
from unveil.model.lora import LoRABranch
from unveil.model.sam3 import enable_vit_training

LORA_INIT_SEED = 0


@dataclass
class LoraConfig:
    first_block: int = 24
    r: int = 16
    alpha: float = 32.0


def make_branch(sam3, cfg: LoraConfig, device: str) -> LoRABranch:
    """The LoRA branch, with A drawn from LORA_INIT_SEED (the same on every GPU); the global RNG state is restored."""
    dev = torch.device(device)
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.random.default_generator.manual_seed(LORA_INIT_SEED)
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.manual_seed(LORA_INIT_SEED)
        return LoRABranch(sam3, cfg.r, cfg.alpha, cfg.first_block).to(device)


class TrainNet(nn.Module):
    def __init__(self, sam3, text: dict, lora: LoraConfig, device: str):
        """text: ``backbone.forward_text`` output of every training prompt (``prompts`` in the batch index it)."""
        super().__init__()
        enable_vit_training()
        branch = make_branch(sam3, lora, device)
        self.heads = ConceptHeads.from_sam3(sam3, copy=False).set_trainable()
        self.branch = branch
        self.convention = nn.Parameter(torch.zeros(2, self.heads.semantic_seg_head.in_channels, device=device))
        self._sam3 = (sam3,)                         # not a registered child: SAM 3 itself stays frozen
        self.text = text
        self.first = lora.first_block

    @property
    def sam3(self):
        return self._sam3[0]

    def pair_inputs(self, prefix: torch.Tensor, img_ids: torch.Tensor, text_ids: torch.Tensor) -> dict:
        """Prefix tokens -> per-pair inputs of the heads, through the LoRA branch (differentiable)."""
        bb = B.image_features(self.sam3, prefix, self.branch)
        return B.pair_features(self.sam3, bb, self.text, img_ids, text_ids)

    def forward(self, prefix: torch.Tensor, img_ids: torch.Tensor, text_ids: torch.Tensor, convention: torch.Tensor,
                sizes: list) -> tuple[list, torch.Tensor]:
        """-> per-pair fp32 logit maps at ``sizes`` (None where sizes[i] is None) and presence logits (P,).
        convention: (P,) bool, the pairs whose maps read the convention vectors (text ids 0 / 1 only)."""
        h = self.heads
        feats = self.pair_inputs(prefix, img_ids, text_ids)
        prompt, pmask = h.prompt_of(feats)
        enc = h.encode(feats["src"], feats["pos"], feats["size"], prompt, pmask)
        mem, pr = enc["memory"], prompt
        u = torch.nonzero(convention.cpu()).flatten()
        if u.numel():
            rows = text_ids.cpu()[u]
            assert bool(((rows == 0) | (rows == 1)).all()), rows
            u, rows = u.to(prompt.device), rows.to(prompt.device)
            sub = {"txt": feats["txt"][:, u], "geo": feats["geo"][:, u], "txt_mask": feats["txt_mask"][u],
                   "geo_mask": feats["geo_mask"][u]}
            pr_u, pm_u = h.prompt_of(sub, self.convention[rows])
            enc_u = h.encode(feats["src"][:, u], feats["pos"][:, u], feats["size"], pr_u, pm_u)
            mem = mem.index_copy(1, u, enc_u["memory"].to(mem.dtype))
            pr = prompt.index_copy(1, u, pr_u.to(prompt.dtype))
        logits = [None] * len(sizes)
        sel = [i for i, s in enumerate(sizes) if s is not None]
        if sel:
            idx = torch.tensor(sel, device=prompt.device)
            coarse = h.semantic([f[idx] for f in feats["fpn"][:-1]], feats["size"], mem[:, idx], pr[:, idx], pmask[idx])
            for k, i in enumerate(sel):
                logits[i] = coarse_to(coarse[k:k + 1], sizes[i])[0, 0]
        pres = h.presence(enc, prompt, pmask)
        if not u.numel() and self.training:
            pres = pres + 0.0 * self.convention.sum()          # keeps the vectors in the graph for DDP
        return logits, pres

    # ------------------------------------------------------------------------------------------------ state

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state(self) -> dict:
        """Trained tensors, grouped as in the checkpoint: heads, LoRA, convention vectors."""
        keep = self.heads.trained_keys()
        return {"heads": {k: v.detach().cpu() for k, v in self.heads.state_dict().items() if k in keep},
                "lora": self.branch.lora_state(), "convention": self.convention.detach().cpu()}

    def load(self, state: dict) -> None:
        self.heads.load_trained(state["heads"])
        self.branch.load_lora(state["lora"])
        with torch.no_grad():
            self.convention.copy_(state["convention"])


def param_groups(net: TrainNet, lr: float, weight_decay: float, lr_lora: float, wd_lora: float,
                 lr_convention: float) -> list[dict]:
    """Heads at lr, without weight decay on biases and LayerNorm / GroupNorm weights (SAM 3's rule); the LoRA
    factors at lr_lora; the convention vectors at lr_convention without weight decay."""
    decay, no_decay = [], []
    for _, mod in net.heads.named_modules():
        for name, p in mod.named_parameters(recurse=False):
            if p.requires_grad:
                plain = name.endswith("bias") or isinstance(mod, (nn.LayerNorm, nn.GroupNorm))
                (no_decay if plain else decay).append(p)
    groups = [(decay, lr, weight_decay), (no_decay, lr, 0.0), (net.branch.lora_parameters(), lr_lora, wd_lora),
              ([net.convention], lr_convention, 0.0)]
    return [{"params": ps, "lr": r, "base_lr": r, "weight_decay": wd} for ps, r, wd in groups if ps]


def decoder_param_groups(decoder: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    """The joint presence decoder's groups (the same weight-decay rule)."""
    decay, no_decay = [], []
    for _, mod in decoder.named_modules():
        for name, p in mod.named_parameters(recurse=False):
            if p.requires_grad:
                plain = name.endswith("bias") or isinstance(mod, (nn.LayerNorm, nn.GroupNorm))
                (no_decay if plain else decay).append(p)
    return [{"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd}
            for ps, wd in ((decay, weight_decay), (no_decay, 0.0)) if ps]

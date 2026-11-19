"""The heads UNVEIL trains: SAM 3's fusion encoder, semantic path and presence decoder.

Per (image, prompt) pair the fusion encoder gives a prompt-conditioned memory M on the 72 x 72 grid.
- Semantic path: prompt cross-attention + LayerNorm on M, the pixel decoder (72 -> 144 -> 288) and a 1 x 1 conv give the
  logit map S (288 x 288). SAM 3 builds a third pixel-decoder stage that is never run; it stays frozen.
- Presence decoder: SAM 3's DETR decoder (200 queries + presence token) on M and the prompt; its last layer's
  presence token gives one logit p per pair. It reads M and the prompt through a stop-gradient, so the presence loss
  never reaches the encoder or the LoRA branch. The box head and the final query norm do not affect p and stay frozen.

UNVEIL trains a copy of these modules (``ConceptHeads.from_sam3(sam3, copy=True)`` at inference). With
``copy=False`` the class wraps SAM 3's own modules: the frozen semantic path of native SAM 3, or the modules that
training updates in place before they are exported as the copy.
"""

from __future__ import annotations

import copy as _copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

FROZEN_DECODER = ("bbox_embed", "norm", "instance_")     # decoder children that do not feed the presence logit
USED_PIXEL_STAGES = 2


def coarse_to(logit: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of (P, 1, 288, 288) logits to hw, in fp32."""
    return F.interpolate(logit.float(), size=hw, mode="bilinear", align_corners=False)


def presence_logits(decoder: nn.Module, enc: dict, prompt: torch.Tensor, pmask: torch.Tensor) -> torch.Tensor:
    """Presence logits (P,) of a DETR decoder with presence token, reading the memory and the prompt detached."""
    mem = enc["memory"].detach()
    tgt = decoder.query_embed.weight.unsqueeze(1).repeat(1, mem.shape[1], 1)
    _, _, pres, _ = decoder(
        tgt=tgt, memory=mem, memory_key_padding_mask=enc["padding_mask"], pos=enc["pos_embed"],
        reference_boxes=None, level_start_index=enc["level_start_index"], spatial_shapes=enc["spatial_shapes"],
        valid_ratios=enc["valid_ratios"], tgt_mask=None, memory_text=prompt.detach(), text_attention_mask=pmask,
        apply_dac=False)
    return pres[-1].reshape(-1).float()


def make_trainable_decoder(decoder: nn.Module) -> nn.Module:
    """Mark the presence-relevant parts of a decoder trainable (in place)."""
    decoder.requires_grad_(True)
    for name, mod in decoder.named_children():
        if name.startswith(FROZEN_DECODER):
            mod.requires_grad_(False)
    return decoder


def trained_decoder_keys(decoder: nn.Module) -> set[str]:
    return {k for k, _ in decoder.named_parameters() if not k.split(".")[0].startswith(FROZEN_DECODER)}


class ConceptHeads(nn.Module):
    def __init__(self, encoder, cross_attn_norm, cross_attend_prompt, pixel_decoder, semantic_seg_head, decoder):
        super().__init__()
        self.encoder = encoder
        self.cross_attn_norm = cross_attn_norm
        self.cross_attend_prompt = cross_attend_prompt
        self.pixel_decoder = pixel_decoder
        self.semantic_seg_head = semantic_seg_head
        self.decoder = decoder

    @classmethod
    def from_sam3(cls, sam3, copy: bool) -> "ConceptHeads":
        seg = sam3.segmentation_head
        mods = (sam3.transformer.encoder, seg.cross_attn_norm, seg.cross_attend_prompt, seg.pixel_decoder,
                seg.semantic_seg_head, sam3.transformer.decoder)
        if copy:
            mods = tuple(_copy.deepcopy(m) for m in mods)
        return cls(*mods).eval().requires_grad_(False)

    # ------------------------------------------------------------------------------------------------ parameters

    def set_trainable(self) -> "ConceptHeads":
        """Everything trains except the unused third pixel-decoder stage and the decoder parts outside presence."""
        self.requires_grad_(True)
        for i in range(USED_PIXEL_STAGES, len(self.pixel_decoder.conv_layers)):
            self.pixel_decoder.conv_layers[i].requires_grad_(False)
            self.pixel_decoder.norms[i].requires_grad_(False)
        make_trainable_decoder(self.decoder)
        return self

    def trained_keys(self) -> set[str]:
        keep = set()
        for k, _ in self.named_parameters():
            parts = k.split(".")
            if parts[0] == "decoder" and parts[1].startswith(FROZEN_DECODER):
                continue
            if parts[0] == "pixel_decoder" and parts[1] in ("conv_layers", "norms") and int(parts[2]) >= USED_PIXEL_STAGES:
                continue
            keep.add(k)
        return keep

    def load_trained(self, state: dict) -> None:
        missing, extra = self.trained_keys() - set(state), set(state) - set(self.state_dict())
        if missing or extra:
            raise KeyError(f"head tensors: missing {sorted(missing)[:4]}, unexpected {sorted(extra)[:4]}")
        self.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------------------------------------ forward

    def encode(self, src: torch.Tensor, pos: torch.Tensor, size, prompt: torch.Tensor, pmask: torch.Tensor) -> dict:
        """Fusion encoder: image tokens (HW, P, C) attend to the prompt (L, P, C) -> {"memory", ...}."""
        return self.encoder(src=[src], src_key_padding_mask=None, src_pos=[pos], prompt=prompt,
                            prompt_pos=torch.zeros_like(prompt), prompt_key_padding_mask=pmask, feat_sizes=[size])

    def semantic(self, hires: list, size, mem: torch.Tensor, prompt: torch.Tensor, pmask: torch.Tensor) -> torch.Tensor:
        """Semantic logits (P, 1, 288, 288) from the memory. hires: the pairs' 288 and 144 FPN levels
        (``feats["fpn"][:-1]``); the 72 level is replaced by the prompt-attended memory."""
        x = self.cross_attend_prompt(query=self.cross_attn_norm(mem), key=prompt, value=prompt,
                                     key_padding_mask=pmask)[0] + mem
        b, c = x.shape[1], x.shape[2]
        h, w = size
        levels = list(hires) + [x.permute(1, 2, 0).reshape(b, c, h, w)]
        if self.training:
            pixel_embed = ckpt.checkpoint(self.pixel_decoder, levels, use_reentrant=False)
        else:
            pixel_embed = self.pixel_decoder(levels)
        return self.semantic_seg_head(pixel_embed)

    def presence(self, enc: dict, prompt: torch.Tensor, pmask: torch.Tensor) -> torch.Tensor:
        return presence_logits(self.decoder, enc, prompt, pmask)

    @staticmethod
    def prompt_of(feats: dict, convention: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt tokens (text + geometry) and mask; convention (P, C) is added to every text token if given."""
        txt = feats["txt"]
        if convention is not None:
            txt = (txt.float() + convention.float()[None]).to(txt.dtype)
        return torch.cat([txt, feats["geo"]]), torch.cat([feats["txt_mask"], feats["geo_mask"]], 1)

    @torch.no_grad()
    def single(self, feats: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Single queries -> (semantic logits (P, 1, 288, 288), presence logits (P,))."""
        prompt, pmask = self.prompt_of(feats)
        enc = self.encode(feats["src"], feats["pos"], feats["size"], prompt, pmask)
        sem = self.semantic(feats["fpn"][:-1], feats["size"], enc["memory"], prompt, pmask)
        return sem, self.presence(enc, prompt, pmask)

    @torch.no_grad()
    def joint(self, feats: dict, convention: torch.Tensor, joint_decoder: nn.Module
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Members of a joint query, each with its convention vector (P, C) -> (semantic logits, joint presence)."""
        prompt, pmask = self.prompt_of(feats, convention)
        enc = self.encode(feats["src"], feats["pos"], feats["size"], prompt, pmask)
        return (self.semantic(feats["fpn"][:-1], feats["size"], enc["memory"], prompt, pmask),
                presence_logits(joint_decoder, enc, prompt, pmask))


def joint_decoder_from(heads: ConceptHeads, state: dict | None = None) -> nn.Module:
    """A copy of the heads' presence decoder, with the stored trained parts if ``state`` is given."""
    dec = _copy.deepcopy(heads.decoder)
    if state is not None:
        missing, extra = trained_decoder_keys(dec) - set(state), set(state) - set(dec.state_dict())
        if missing or extra:
            raise KeyError(f"joint decoder tensors: missing {sorted(missing)[:4]}, unexpected {sorted(extra)[:4]}")
        dec.load_state_dict(state, strict=False)
    return dec.eval().requires_grad_(False)

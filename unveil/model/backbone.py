"""SAM 3 image and prompt features with an exchangeable ViT tail.

``vit_prefix`` runs the shared part of the ViT once (patch embedding, absolute positions, ``ln_pre``, blocks
0..first-1). ``image_features`` continues through a tail (SAM 3's own blocks, the merged UNVEIL copies, or the LoRA
branch in training), ``ln_post`` and SAM 3's neck. ``pair_features`` assembles the per-(image, prompt) inputs of the
heads as ``Sam3Image._encode_prompt`` does: text tokens plus the geometry encoder's empty-box token.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unveil.model.lora import vit_trunk


def vit_prefix(sam3, images: torch.Tensor, first: int) -> torch.Tensor:
    """Normalized images (B, 3, 1008, 1008) -> tokens after block first-1 (B, 72, 72, 1024). No gradients."""
    from sam3.model.vitdet import get_abs_pos

    t = vit_trunk(sam3)
    with torch.no_grad():
        x = t.patch_embed(images)
        h, w = x.shape[1], x.shape[2]
        x = x + get_abs_pos(t.pos_embed, t.pretrain_use_cls_token, (h, w), t.retain_cls_token, tiling=t.tile_abs_pos)
        x = t.ln_pre(x)
        for blk in t.blocks[:first]:
            x = blk(x)
    return x


def native_tail(sam3, first: int) -> nn.Module:
    """SAM 3's own blocks first..31."""
    return nn.Sequential(*vit_trunk(sam3).blocks[first:])


def image_features(sam3, x: torch.Tensor, tail: nn.Module) -> dict:
    """``SAM3VLBackbone.forward_image`` (SAM 3 outputs only) from the prefix tokens through ``tail``. Gradients flow
    when the caller enables them (training of the LoRA branch)."""
    bb = sam3.backbone
    neck = bb.vision_backbone
    x = vit_trunk(sam3).ln_post(tail(x)).permute(0, 3, 1, 2)
    n = len(neck.convs) - bb.scalp
    fpn = [neck.convs[i](x) for i in range(n)]
    pos = [neck.position_encoding(f).to(f.dtype) for f in fpn]
    return {"vision_features": fpn[-1], "vision_pos_enc": pos, "backbone_fpn": fpn, "sam2_backbone_out": None}


def pair_features(sam3, bb: dict, text: dict, img_ids: torch.Tensor, text_ids: torch.Tensor) -> dict:
    """Inputs of the heads for P (image, prompt) pairs.

    bb: ``image_features`` output; text: ``backbone.forward_text`` output; img_ids, text_ids: (P,) indices into the
    images of ``bb`` and the prompts of ``text``. Returns the flattened image tokens (``src``, ``pos``, ``size``), the
    FPN levels per pair, the text tokens (``txt``, ``txt_mask``) and the geometry token (``geo``, ``geo_mask``).
    """
    from sam3.model.data_misc import FindStage

    dev = bb["backbone_fpn"][0].device
    p = int(img_ids.numel())
    bb = {**bb, **text}
    find = FindStage(img_ids=img_ids.to(dev), text_ids=text_ids.to(dev), input_boxes=None, input_boxes_mask=None,
                     input_boxes_label=None, input_points=None, input_points_mask=None)
    prompt, prompt_mask, bb = sam3._encode_prompt(bb, find, sam3._get_dummy_prompt(p))
    _, src, pos, sizes = sam3._get_img_feats(bb, find.img_ids)
    fpn = [x[find.img_ids] for x in bb["backbone_fpn"]]
    n = text["language_features"].shape[0]
    return {"fpn": fpn, "src": src[0], "pos": pos[0], "size": tuple(sizes[0]),
            "txt": prompt[:n], "txt_mask": prompt_mask[:, :n], "geo": prompt[n:], "geo_mask": prompt_mask[:, n:]}


def encode_text(sam3, phrases: list[str], device) -> dict:
    """SAM 3's text encoder on a list of phrases (bf16 autocast, as in training and evaluation)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return sam3.backbone.forward_text(list(phrases), device=device)

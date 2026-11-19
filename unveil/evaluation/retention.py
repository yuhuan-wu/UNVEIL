"""Identity retention: LVIS categories never seen in training, answered through the same text interface.

Pairs (``catalog.retention_records``): held-out LVIS images x categories left out of concept replay. Per pair, the
ungated map of the category name at the image size, quantized to 8 bits, binarized at 128, against the union mask of
the category (> 127); IoU(empty, empty) = 1. The value is the mean IoU over the pairs (paper: UNVEIL 0.841, frozen
SAM 3's semantic path 0.831).

Batches as in the paper's run: the category names encoded in chunks of 128 (sorted), the images sorted by id and
split into ``shards`` (image j to shard j % shards), eight images per batch with all their pairs.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from unveil.model.sam3 import to_input

TEXT_CHUNK = 128


def _encode_all(model, phrases: list[str]) -> None:
    """Encode the names in chunks into the model's text cache (the task prompts keep their own encoding)."""
    from unveil.prompts import TASK_PROMPTS, normalize

    phrases = [normalize(p) for p in phrases]
    for i in range(0, len(phrases), TEXT_CHUNK):
        chunk = phrases[i:i + TEXT_CHUNK]
        model._text.update((p, t) for p, t in zip(chunk, model._encode(chunk)) if p not in TASK_PROMPTS)


@torch.no_grad()
def retention(model, records: list[dict], path: str = "unveil", limit: int | None = None, all_names=None,
              shards: int = 8, batch: int = 8) -> dict:
    """Mean IoU over the pairs (path "sam3": SAM 3's own semantic path). all_names: the category names the paper's
    run encoded together (both retention splits), for its text-encoder chunks."""
    _encode_all(model, sorted(set(all_names or (r["np"] for r in records))))
    by_img: dict[int, list[dict]] = {}
    for r in records:
        by_img.setdefault(r["image_id"], []).append(r)
    groups = [g for _, g in sorted(by_img.items())][:limit]
    ious = []
    for s in range(shards):
        mine = groups[s::shards]
        for i in range(0, len(mine), batch):
            part = mine[i:i + batch]
            imgs = [Image.open(g[0]["image"]).convert("RGB") for g in part]
            img_ids = [k for k, g in enumerate(part) for _ in g]
            pairs = [r for g in part for r in g]
            coarse, _ = model.single(torch.stack([to_input(im) for im in imgs]), torch.tensor(img_ids),
                                     [r["np"] for r in pairs], path)
            for j, (k, r) in enumerate(zip(img_ids, pairs)):
                hw = (imgs[k].height, imgs[k].width)
                m = torch.nn.functional.interpolate(coarse[j:j + 1].float(), size=hw, mode="bilinear",
                                                    align_corners=False)[0, 0]
                pred = ((m.sigmoid() * 255).round() >= 128).cpu().numpy()
                gt = np.asarray(Image.open(r["mask"]).convert("L")) > 127
                union = np.count_nonzero(pred | gt)
                ious.append(np.count_nonzero(pred & gt) / union if union else 1.0)
    return {"pairs": len(ious), "images": len(groups), "mean_iou": float(np.mean(ious))}

"""Training losses. All in fp32 at the working (cropped) image size; labels are uint8 maps.

- Region pairs (salient, camouflaged, transparent, mirror, USC12K pairs, LVIS replay): BCE + IoU loss,
  IoU = 1 - sum(p g) / sum(p + g - p g) with p = sigmoid(logit), g = label >= 128.
- Shadow pairs: class-balanced BCE (mean of the per-class mean BCE over the classes present) + the IoU loss.
- Edge pairs: the class-balanced BCE of RCF. Pixels >= 127.5 are edges (weight |Y-| / |Y|), pixels at 0 non-edges
  (weight 1.1 |Y+| / |Y|), the others are ignored; the sum is divided by the weight sum.
- Skeleton pairs: HED's class-balanced BCE on the binary skeleton dilated by one pixel (3 x 3 max-pool).
- Dense loss: the mean over the loss groups present of each group's mean pair loss.
- Presence: BCE on the presence logits of the pairs with a target (1 present, 0 verified absent; others skipped),
  class-balanced as 0.5 x mean over positives + 0.5 x mean over negatives (a class absent from the batch drops out).
- Total = dense + lambda_presence x presence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EDGE_THR, RCF_LAMBDA = 127.5, 1.1


def iou_loss(logit: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    p = logit.sigmoid()
    inter = (p * g).sum()
    union = (p + g - p * g).sum()
    return 1.0 - inter / union.clamp_min(1e-6)


def region_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    g = (label >= 128).float()
    return F.binary_cross_entropy_with_logits(logit, g) + iou_loss(logit, g)


def balanced_bce(logit: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, g, reduction="none")
    n_pos = g.sum()
    n_neg = g.numel() - n_pos
    pos = (bce * g).sum() / n_pos.clamp_min(1.0)
    neg = (bce * (1.0 - g)).sum() / n_neg.clamp_min(1.0)
    has_pos, has_neg = (n_pos > 0).float(), (n_neg > 0).float()
    return (has_pos * pos + has_neg * neg) / (has_pos + has_neg).clamp_min(1.0)


def shadow_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    g = (label >= 128).float()
    return balanced_bce(logit, g) + iou_loss(logit, g)


def edge_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    v = label.float()
    pos, neg = v >= EDGE_THR, v == 0
    n_pos, n_neg = pos.sum().float(), neg.sum().float()
    tot = (n_pos + n_neg).clamp_min(1.0)
    zero = torch.zeros((), device=v.device)
    w = torch.where(pos, n_neg / tot, torch.where(neg, RCF_LAMBDA * n_pos / tot, zero))
    return F.binary_cross_entropy_with_logits(logit, pos.float(), weight=w, reduction="sum") / w.sum().clamp_min(1e-6)


def dilate(label: torch.Tensor) -> torch.Tensor:
    """Binary skeleton (label >= 128) dilated by a 3 x 3 square -> uint8 in {0, 255}."""
    pos = (label >= 128).float()[None, None]
    return (F.max_pool2d(pos, 3, stride=1, padding=1)[0, 0] * 255).to(torch.uint8)


def skeleton_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    pos = dilate(label) >= 128
    n_pos = pos.sum().float()
    tot = float(pos.numel())
    w = torch.where(pos, (tot - n_pos) / tot, n_pos / tot)
    return F.binary_cross_entropy_with_logits(logit, pos.float(), weight=w, reduction="sum") / w.sum().clamp_min(1e-6)


def pair_loss(kind: str, group: str):
    if group == "shadow":
        return shadow_loss
    return {"region": region_loss, "edge": edge_loss, "skeleton": skeleton_loss}[kind]


def presence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terms = []
    for v in (1.0, 0.0):
        m = target == v
        if bool(m.any()):
            terms.append(F.binary_cross_entropy_with_logits(logits[m], target[m]))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def total_loss(logits: list, labels: list, kinds: list[str], groups: list[str], pres_logits: torch.Tensor,
               pres_target: torch.Tensor, lam_presence: float = 1.0) -> tuple[torch.Tensor, dict]:
    """-> (total loss, detached per-group means and the presence term). Pairs without a label are presence-only."""
    per_group: dict[str, list[torch.Tensor]] = {}
    for lg, lb, k, g in zip(logits, labels, kinds, groups):
        if lb is not None:
            per_group.setdefault(g, []).append(pair_loss(k, g)(lg, lb))
    means = {g: torch.stack(v).mean() for g, v in per_group.items()}
    dense = torch.stack(list(means.values())).mean()
    pres = presence_loss(pres_logits, pres_target.to(pres_logits.device))
    parts = {g: v.detach() for g, v in means.items()}
    parts["presence"] = pres.detach()
    return dense + lam_presence * pres, parts

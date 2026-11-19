"""Per-sample loading and augmentation of the training streams.

Every sample is a deterministic function of (item, sample seed): ``numpy.random.default_rng(seed)`` draws all of its
random choices, so the data do not depend on worker processes or on the order in which samples are loaded.

- Region items (salient, camouflaged, shadow, transparent, mirror, the USC12K pairs, LVIS replay) and presence
  negatives: horizontal flip with p = 0.5, then a crop whose height and width are each U(0.75, 1) of the image side,
  at a random position. Masks are binarized at 128.
- Edge items: no augmentation; the label is the 8-bit mean over annotators (HED-BSDS).
- Skeleton items: one of the 8 dihedral transforms (flip with p = 0.5, rotation by k x 90 degrees, k uniform in 0..3),
  no crop; the label is the binary skeleton.
- Network input: PIL bilinear stretch to 1008 x 1008. Labels stay at the working (cropped) size.
- Items marked ``downscale`` (longer side > 1008) are first resized to a longer side of 1008 (PIL bilinear; masks
  resized bilinearly from 8-bit gray and binarized at 128).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from unveil.model.sam3 import to_input

MAX_SIDE = 1008
CROP_MIN = 0.75


def fit_size(w: int, h: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    s = max(w, h)
    if s <= max_side:
        return w, h
    return max(1, round(w * max_side / s)), max(1, round(h * max_side / s))


def _binary_resized(path: str, size: tuple[int, int]) -> Image.Image:
    g = np.asarray(Image.open(path).convert("L").resize(size, Image.BILINEAR)) >= 128
    return Image.fromarray(g.astype(np.uint8) * 255)


def open_item(item: dict) -> tuple[Image.Image, list[Image.Image]]:
    """-> (RGB image, its masks in the order gt, gt2). Downscaled items are resized as described above."""
    img = Image.open(item["image"]).convert("RGB")
    paths = [item[k] for k in ("gt", "gt2") if item.get(k)]
    if item.get("downscale"):
        size = fit_size(*img.size)
        return img.resize(size, Image.BILINEAR), [_binary_resized(p, size) for p in paths]
    return img, [Image.open(p) for p in paths]


def fit_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.resize(size, Image.BILINEAR) if mask.size != size else mask


def region_label(mask: Image.Image) -> np.ndarray:
    return ((np.asarray(mask.convert("L")) >= 128) * 255).astype(np.uint8)


def random_crop_box(w: int, h: int, rng: np.random.Generator, lo: float = CROP_MIN) -> tuple[int, int, int, int]:
    cw, ch = max(1, round(w * rng.uniform(lo, 1.0))), max(1, round(h * rng.uniform(lo, 1.0)))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return x0, y0, x0 + cw, y0 + ch


def flip_crop(img: Image.Image, labels: list[np.ndarray], rng: np.random.Generator):
    """The region augmentation: flip with p = 0.5, then the random crop."""
    if rng.random() < 0.5:
        img, labels = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), [x[:, ::-1] for x in labels]
    box = random_crop_box(img.width, img.height, rng)
    return img.crop(box), [x[box[1]:box[3], box[0]:box[2]] for x in labels]


def dihedral(img: Image.Image, lab: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        img, lab = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), lab[:, ::-1]
    k = int(rng.integers(0, 4))
    if k:
        rot = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)[k - 1]
        img, lab = img.transpose(rot), np.rot90(lab, k)       # both counter-clockwise
    return img, lab


def has_foreground(lab: np.ndarray, kind: str) -> bool:
    return bool((lab >= 127.5).any()) if kind == "edge" else bool((lab >= 128).any())


def pair(prompt: int, label: np.ndarray | None, kind: str, presence: int, group: str, convention: bool = False) -> dict:
    """One (image, prompt) pair of a sample. label None: a presence-only pair. presence: 1 present, 0 verified
    absent, -1 no presence target (e.g. a crop that removed the object)."""
    lab = None if label is None else torch.from_numpy(np.array(label, dtype=np.uint8, order="C"))
    return {"prompt": int(prompt), "label": lab, "kind": kind, "presence": int(presence), "group": group,
            "convention": bool(convention)}


def dense_sample(item: dict, kind: str, prompts: tuple[int, ...], group: str, seed: int, augment: bool = True,
                 convention: bool = False) -> dict:
    """A dense item (kind region / edge / skeleton) or a presence negative (kind "negative")."""
    img, masks = open_item(item)
    masks = [fit_mask(m, img.size) for m in masks]
    rng = np.random.default_rng(seed)
    if kind == "edge":
        labs = [np.asarray(masks[0].convert("L"), dtype=np.uint8)]
    elif kind == "negative":
        labs = []
    else:
        labs = [region_label(m) for m in masks]
    if augment and kind == "skeleton":
        img, labs[0] = dihedral(img, labs[0], rng)
    elif augment and kind in ("region", "negative"):
        img, labs = flip_crop(img, labs, rng)
    if kind == "negative":
        pairs = [pair(prompts[0], None, "negative", 0, group)]
    else:
        assert len(labs) == len(prompts), (group, item.get("id", item["image"]))
        pairs = [pair(p, lab, kind, 1 if has_foreground(lab, kind) else -1, group, convention)
                 for p, lab in zip(prompts, labs)]
    return {"input": to_input(img), "pairs": pairs}


def replay_sample(item: dict, prompt_base: int, seed: int, augment: bool = True) -> dict:
    """LVIS concept replay: one of the image's categories, its union mask as the region target."""
    rng = np.random.default_rng(seed)
    k, mask_path = item["cats"][int(rng.integers(len(item["cats"])))]
    img = Image.open(item["image"]).convert("RGB")
    with Image.open(mask_path) as m:
        lab = region_label(fit_mask(m, img.size))
    if augment:
        img, (lab,) = flip_crop(img, [lab], rng)
    return {"input": to_input(img),
            "pairs": [pair(prompt_base + int(k), lab, "region", 1 if has_foreground(lab, "region") else -1, "replay")]}


def replay_negative_sample(item: dict, seed: int, augment: bool = True) -> dict:
    """An LVIS image with a category verified absent: presence target 0, no map. "mirror" is drawn whenever the image
    lists it (``item["task"]``), else one of its verified-absent categories (``item["negs"]``)."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(len(item["negs"])))
    prompt = item["task"][0] if item["task"] else item["negs"][k]
    img = Image.open(item["image"]).convert("RGB")
    if augment:
        if rng.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = img.crop(random_crop_box(img.width, img.height, rng))
    return {"input": to_input(img), "pairs": [pair(prompt, None, "negative", 0, "replay_neg")]}

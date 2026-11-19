"""Drawing maps on images (numpy and PIL only), in the style of the paper's figures.

Region maps: the binary map (p >= 0.5) as a translucent fill with an outline, on the image desaturated so that the
overlay reads on any scene. Thin maps (edges, skeletons): the map as dark lines on a lightened grey image, the line
opacity following the probability.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

COLORS = {
    "salient object": (230, 120, 20),
    "camouflaged object": (20, 150, 220),
    "shadow": (120, 60, 220),
    "transparent object": (0, 170, 120),
    "mirror": (215, 40, 120),
}
DEFAULT_COLOR, LINE_COLOR = (230, 120, 20), (20, 20, 20)


def to_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    a = np.asarray(image)
    return np.stack([a] * 3, -1) if a.ndim == 2 else a[..., :3].astype(np.uint8)


def _desaturate(rgb: np.ndarray, keep: float) -> np.ndarray:
    grey = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return keep * rgb.astype(np.float32) + (1 - keep) * grey[..., None]


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    import cv2

    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


def _paint(out: np.ndarray, mask: np.ndarray, color, alpha: float, width: int) -> None:
    c = np.asarray(color, np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    out[_outline(mask, width)] = c


def region_overlay(image, prob: np.ndarray, color=DEFAULT_COLOR, alpha: float = 0.45, threshold: float = 0.5
                   ) -> Image.Image:
    rgb = to_rgb(image)
    out = _desaturate(rgb, 0.45)
    _paint(out, prob >= threshold, color, alpha, max(1, round(max(rgb.shape[:2]) / 400)))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def line_overlay(image, prob: np.ndarray, color=LINE_COLOR, lighten: float = 0.6) -> Image.Image:
    rgb = to_rgb(image)
    grey = _desaturate(rgb, 0.0)
    base = grey + lighten * (255.0 - grey)
    a = np.clip(prob.astype(np.float32), 0, 1)[..., None]
    out = (1 - a) * base + a * np.asarray(color, np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def overlay(image, prob: np.ndarray, prompt: str, thin: bool | None = None) -> Image.Image:
    """Region or line overlay of a [0, 1] map, chosen by the prompt."""
    from unveil.prompts import is_thin, normalize

    thin = is_thin(prompt) if thin is None else thin
    if thin:
        return line_overlay(image, prob)
    return region_overlay(image, prob, COLORS.get(normalize(prompt), DEFAULT_COLOR))


def joint_overlay(image, probs: list[np.ndarray], prompts: list[str], alpha: float = 0.45) -> Image.Image:
    """Both members of a joint query in their own colors on one image."""
    from unveil.prompts import normalize

    rgb = to_rgb(image)
    out, width = _desaturate(rgb, 0.45), max(1, round(max(rgb.shape[:2]) / 400))
    for p, prompt in zip(probs, prompts):
        _paint(out, p >= 0.5, COLORS.get(normalize(prompt), DEFAULT_COLOR), alpha, width)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def mask_image(prob: np.ndarray) -> Image.Image:
    """A [0, 1] map as an 8-bit grey image (white = 1)."""
    return Image.fromarray(np.round(np.clip(prob, 0, 1) * 255).astype(np.uint8))

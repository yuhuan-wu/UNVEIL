"""SAM 3's presence veto, and SAM 3 with its presence score set to one (no UNVEIL weights needed).

SAM 3 scores detection i as s_i = sigmoid(d_i) * sigmoid(p), with one image-level presence logit p, and keeps the
detections with s_i >= 0.5. For concepts such as "camouflaged object" or "edges", sigmoid(p) is low on most images, so
every detection is dropped although the masks often hold the concept (paper, Sec. 3.3). Setting sigmoid(p) = 1 keeps
s_i = sigmoid(d_i): the image-level veto is gone, and only the detection scores decide, so "none" is answered only
when no detection is confident.

    import torch
    from unveil.model.sam3 import build_sam3
    from unveil.model.veto import set_presence_one, instance_map
    processor = set_presence_one(build_sam3("cuda"), confidence=0.05)     # an official Sam3Processor
    with torch.autocast("cuda", dtype=torch.bfloat16):                    # as SAM 3's own examples
        state = processor.set_text_prompt("camouflaged object", processor.set_image(image))
    soft = instance_map(state)                                            # the paper's instance map
    state["presence"]                                                     # SAM 3's own presence probability
"""

from __future__ import annotations

import numpy as np
import torch

MAP_FLOOR = 0.05            # the instance map keeps detections with s_i >= MAP_FLOOR


def _processor_class():
    from sam3.model.sam3_image_processor import Sam3Processor

    class PresenceOneProcessor(Sam3Processor):
        """``Sam3Processor`` that records SAM 3's presence probability in ``state["presence"]`` and, while ``lift``
        is True, scores the detections with sigmoid(p) = 1."""

        lift = True

        @torch.inference_mode()
        def _forward_grounding(self, state):
            model = self.model
            original = model.forward_grounding

            def forward_grounding(*args, **kwargs):
                out = original(*args, **kwargs)
                pres = out["presence_logit_dec"]
                state["presence"] = float(pres.float().sigmoid().reshape(-1)[0])
                if self.lift:
                    out = {**out, "presence_logit_dec": torch.full_like(pres, float("inf"))}   # sigmoid = 1
                return out

            model.forward_grounding = forward_grounding          # instance attribute, removed below
            try:
                return super()._forward_grounding(state)
            finally:
                del model.forward_grounding

    return PresenceOneProcessor


def processor_for(sam3_model, device: str | None = None, lift: bool = True, confidence: float = 0.5):
    """An official-style processor on ``sam3_model``; lift=False keeps SAM 3's own scores (presence still recorded)."""
    device = device or str(next(sam3_model.parameters()).device)
    proc = _processor_class()(sam3_model, device=device, confidence_threshold=confidence)
    proc.lift = lift
    return proc


def set_presence_one(sam3_model, confidence: float = 0.5):
    """A processor on SAM 3 with sigmoid(p) set to one; nothing else changes."""
    return processor_for(sam3_model, None, True, confidence)


@torch.no_grad()
def instance_map(state: dict, floor: float = MAP_FLOOR) -> np.ndarray:
    """The paper's instance map: per pixel, the highest s_i times the mask over the detections with s_i >= floor.
    The state must come from a processor whose confidence threshold is at most ``floor``. -> float32 (H, W)."""
    masks, scores = state["masks"], state["scores"]
    h, w = state["original_height"], state["original_width"]
    keep = scores >= floor
    if not bool(keep.any()):
        return np.zeros((h, w), np.float32)
    m = masks[keep].float().reshape(-1, h, w) * scores[keep].float()[:, None, None]
    return m.amax(0).cpu().numpy().astype(np.float32)

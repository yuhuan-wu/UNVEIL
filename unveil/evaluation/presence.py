"""Presence decisions (paper, Tab. 6): answering when the concept is present, saying "none" when it is absent.

- Answers when present: per task, the share of its test images whose presence passes the prompt's threshold, pooled
  over the task's test sets; the value is the mean over the seven tasks. The joint row takes, on the salient and
  camouflaged tasks, the joint query's decision for the queried member, and the single query's elsewhere.
- Says "none" when absent: the rejection split (``catalog.rejection_records``: COD10K background images, USC12K scenes
  without the prompted concept, LVIS categories verified absent). A query is answered "none" when the deployed output
  is empty: the presence is below the threshold, or no pixel of the map (at the image size) reaches 0.5. The value is
  the mean over the three sources. The joint row answers the COD10K and USC12K queries with the joint query (its
  queried member) and the LVIS queries with single queries.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from unveil import prompts as P
from unveil.model.sam3 import to_input

SOURCES = ("cod10k_bg", "usc12k", "lvis")
JOINT_SOURCES = ("cod10k_bg", "usc12k")
JOINT_TASKS = ("salient", "camouflaged")


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@torch.no_grad()
def answer(model, image: Image.Image, prompt: str, joint: bool = False) -> dict:
    """One query at the image size -> {"presence", "on", "empty"}; joint: the joint query's queried member."""
    x = to_input(image)[None]
    zero = torch.zeros(1, dtype=torch.long)
    if joint:
        coarse, pres = model.joint(x, zero, torch.tensor([P.JOINT_PAIR.index(P.normalize(prompt))]))
    else:
        coarse, pres = model.single(x, zero, [prompt])
    logit = torch.nn.functional.interpolate(coarse[:1].float(), size=(image.height, image.width), mode="bilinear",
                                            align_corners=False)[0, 0]
    p = float(torch.sigmoid(pres[0]))
    on = p >= model.threshold(prompt, joint)
    if joint:            # the joint path stores 8-bit maps: a pixel counts at round(255 p) >= 128
        hit = bool(((logit.sigmoid() * 255).round() >= 128).any())
    else:
        hit = bool((logit >= 0).any())
    return {"presence": round(p, 5), "on": on, "empty": not (on and hit)}


def rejection(model, records: list[dict], joint: bool = False) -> dict:
    """Share answered "none" per source and their mean (x 100)."""
    per = {s: [] for s in SOURCES}
    for r in records:
        use_joint = joint and r["source"] in JOINT_SOURCES
        with Image.open(r["image"]) as im:
            per[r["source"]].append(answer(model, im.convert("RGB"), r["prompt"], use_joint)["empty"])
    out = {s: {"n": len(v), "none": 100.0 * float(np.mean(v))} for s, v in per.items() if v}
    out["mean"] = float(np.mean([out[s]["none"] for s in SOURCES if s in out]))
    return out


def answered(presence: dict[str, dict[str, float]], tasks, sets_of_task, prompt_of: dict[str, str], tau) -> dict:
    """presence: set -> {id: presence logit}; tau(prompt) -> threshold. Share x 100 accepted per task and the mean."""
    out = {}
    for task in tasks:
        k = n = 0
        for s in sets_of_task[task]:
            t = tau(prompt_of[s])
            k += sum(_sig(v) >= t for v in presence[s].values())
            n += len(presence[s])
        out[task] = 100.0 * k / n
    out["mean"] = float(np.mean([out[t] for t in tasks]))
    return out

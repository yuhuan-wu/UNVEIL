"""Presence thresholds, fitted on validation pairs from the training splits before any test image is seen.

For a prompt with validation pairs of both classes, tau maximizes the balanced accuracy (mean of the true positive
and true negative rates) over the grid 0.001 .. 0.999 (step 0.001); ties go to the value closest to 0.5. Prompts
without validation pairs of both classes (edges, shadow, transparent object, mirror, skeleton, symmetry axis) and free
text use the pooled tau, the same rule over all pairs. Single queries and joint queries get their own thresholds,
each fitted on its own presence outputs: single queries on the validation pairs, joint queries on those pairs without
the USC12K single-concept scenes used in training, plus their cross-concept negatives and the held-out USC12K scenes
with both concepts (``catalog.val_pairs``).
"""

from __future__ import annotations

import numpy as np

from unveil import prompts as P

GRID = np.round(np.arange(1, 1000) / 1000, 3)


def best_tau(s: np.ndarray, y: np.ndarray) -> dict:
    tpr = np.array([(s[y == 1] >= t).mean() for t in GRID])
    tnr = np.array([(s[y == 0] < t).mean() for t in GRID])
    ba = (tpr + tnr) / 2
    best = np.flatnonzero(ba == ba.max())
    k = best[np.argmin(np.abs(GRID[best] - 0.5))]
    return {"tau": float(GRID[k]), "balanced_acc": float(ba[k]), "tpr": float(tpr[k]), "fpr": float(1 - tnr[k]),
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def fit(records: list[dict], logits: list[dict]) -> dict:
    """records: validation pairs {id, prompts: [p], presence}; logits: [{id, prompt, logit}] of the same pairs.
    -> {"per_prompt": {prompt: fit}, "pooled": fit}."""
    lut = {(x["id"], x["prompt"]): x["logit"] for x in logits}
    s = np.array([1.0 / (1.0 + np.exp(-lut[(r["id"], p)])) for r in records for p in r["prompts"]])
    y = np.array([r["presence"] for r in records for _ in r["prompts"]])
    prompt = np.array([p for r in records for p in r["prompts"]])
    pooled = best_tau(s, y)
    per = {}
    for pid, name in enumerate(P.TASK_PROMPTS):
        m = prompt == pid
        if (y[m] == 1).any() and (y[m] == 0).any():
            per[name] = {**best_tau(s[m], y[m]), "source": "own validation pairs"}
        else:
            per[name] = {"tau": pooled["tau"], "source": "pooled"}
    return {"per_prompt": per, "pooled": pooled}


def checkpoint_thresholds(single: dict, joint: dict) -> dict:
    """The threshold table stored in the checkpoint."""
    return {"single": {p: v["tau"] for p, v in single["per_prompt"].items()},
            "joint": {p: joint["per_prompt"][p]["tau"] for p in P.JOINT_PAIR},
            "free_text": single["pooled"]["tau"],
            "rule": "max balanced accuracy per prompt on validation pairs from the training splits, ties -> 0.5; "
                    "prompts without both classes and free text use the pooled tau"}

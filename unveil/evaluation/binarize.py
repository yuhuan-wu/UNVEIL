"""Binarization thresholds of the thresholded metrics, calibrated on training data.

BER (shadows) and IoU (transparent objects, mirrors) binarize a map: pixel positive iff p >= t on the 8-bit map. Each
model gets one t per concept in 1..255, chosen on the official training split of the concept's sets with the
model's own deployed maps (presence-gated, at the GT size):
- shadow: minimize the mean of SBU-train BER and ISTD-train BER (each summed over its set, as the test protocol);
- transparent object: maximize the mean per-image IoU on Trans10K-train;
- mirror: maximize the mean per-image IoU on MSD-train.
Ties go to the t closest to 128. The values of the released model are stored in its checkpoint ("binarize").
``calibrate_maps`` applies the same rule to the stored maps of another model on these training images.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

T_FIXED = 128
GRID = np.arange(1, 256)
OBJECTIVE = {"shadow": ("BER", "min"), "transparent": ("IoU", "max"), "mirror": ("IoU", "max")}
CONCEPT_SETS = {"shadow": ("sbu_train", "istd_train"), "transparent": ("trans10k_train",), "mirror": ("msd_train",)}
TASK_CONCEPT = {"shadow": "shadow", "transparent": "transparent", "mirror": "mirror"}       # test-set task -> concept


def cumulative(fg: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-image TP and FP of p >= t for t = 0..256; fg / bg: (n, 256) histograms of the map over GT fg / bg."""
    z = np.zeros((fg.shape[0], 1), np.int64)
    tp = np.concatenate([np.cumsum(fg[:, ::-1], 1)[:, ::-1], z], 1)
    fp = np.concatenate([np.cumsum(bg[:, ::-1], 1)[:, ::-1], z], 1)
    return tp, fp


def train_curve(fg: np.ndarray, bg: np.ndarray, metric: str) -> np.ndarray:
    """The metric at t = 1..255 on one training set: BER summed over the set, or the mean per-image IoU."""
    tp_c, fp_c = cumulative(fg, bg)
    npos, nneg = fg.sum(1, keepdims=True), bg.sum(1, keepdims=True)
    if metric == "BER":
        tp, fp = tp_c.sum(0)[GRID], fp_c.sum(0)[GRID]
        p, n = npos.sum(), nneg.sum()
        return 100 * (1 - 0.5 * (tp / max(p, 1) + (n - fp) / max(n, 1)))
    tp, fp = tp_c[:, GRID], fp_c[:, GRID]
    den = tp + fp + (npos - tp)
    return np.where(den > 0, tp / np.maximum(den, 1), 1.0).mean(0)


def pick(curve: np.ndarray, how: str) -> int:
    best = curve.min() if how == "min" else curve.max()
    cand = GRID[np.flatnonzero(curve == best)]
    return int(cand[np.lexsort((cand, np.abs(cand - T_FIXED)))][0])


def calibrate(hists: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """hists: training set -> (fg, bg) per-image histograms. -> {concept: t, ...} with the objective per concept."""
    out, detail = {}, {}
    for concept, sets in CONCEPT_SETS.items():
        if not all(s in hists for s in sets):            # a concept without (all) its training sets is not fitted
            continue
        metric, how = OBJECTIVE[concept]
        curves = [train_curve(*hists[s], metric) for s in sets]
        obj = np.mean(curves, 0)
        t = pick(obj, how)
        out[concept] = t
        detail[concept] = {"metric": metric, "t": t, "objective": float(obj[t - 1]),
                           "objective_at_128": float(obj[T_FIXED - 1])}
    return {**out, "detail": detail}


def _map_hist(args) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    from unveil.evaluation.metrics import region_record

    map_path, gt_path = args
    pred = np.asarray(Image.open(map_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    if pred.shape != gt.shape:
        raise ValueError(f"{map_path}: map {pred.shape} and GT {gt.shape} differ")
    r = region_record(pred, gt)
    return r["fg"], r["bg"]


def map_histograms(map_dir, recs: list[dict], workers: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Per-image histograms of stored 8-bit maps (``<map_dir>/<id>.png`` at the size of the GT file) over GT
    foreground / background (GT read as 8-bit gray, foreground >= 128), as ``inference.histograms`` computes them."""
    from unveil.evaluation.inference import map_name

    paths = [Path(map_dir) / map_name(r["id"]) for r in recs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{map_dir}: {len(missing)} of {len(recs)} maps missing, e.g. {missing[:2]}")
    args = list(zip(paths, [r["gt"] for r in recs]))
    if workers <= 1:
        out = [_map_hist(x) for x in args]
    else:
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(workers) as pool:
            out = pool.map(_map_hist, args, chunksize=16)
    return np.stack([f for f, _ in out]).astype(np.int64), np.stack([b for _, b in out]).astype(np.int64)


def calibrate_maps(maps_root, records: dict[str, list[dict]], workers: int = 12) -> dict:
    """The thresholds of another model from its stored maps of the training images. records: training set ->
    [{id, gt}] (``catalog.calibration_records``); maps in ``<maps_root>/<training set>/``. Concepts whose training
    sets are not all given are left out."""
    return calibrate({s: map_histograms(Path(maps_root) / s, recs, workers) for s, recs in records.items()})

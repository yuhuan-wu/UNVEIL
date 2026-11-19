"""The standard metric of each benchmark, on 8-bit maps and uint8 GT.

- Salient / camouflaged objects: PySODMetrics 1.6.2 (map min-max normalized per image, GT > 128): S-measure,
  E-measure (max / mean over the mean curve, adaptive), weighted F-measure, MAE.
- Shadows, transparent objects (and other region sets): per-image histograms of the map over GT foreground and
  background (GT >= 128); binarized map = p >= t, with the model's calibrated t (``binarize``; 128 by default).
  BER sums the pixel counts over the dataset, as SBU and ISTD do; IoU is per image (empty vs empty = 1), averaged.
- Mirrors (MSD, and the GSD / PMD extra sets): IoU per image at t as above; max F and MAE of the HetNet / MirrorNet
  script (map min-max normalized per image).
- Edges (BSDS500): pyEdgeEval 0.2.8, map / 255, NMS + thinning, 99 thresholds, max_dist 0.0075, every annotator.
- Skeletons (SK-LARGE, SYM-PASCAL): map / max, NMS + thinning, 30 thresholds, max_dist 0.01, GT = binary_thin(GT).
pyEdgeEval's pixel matching samples outlier edges from a clock-seeded generator, so ODS / OIS move by about 1e-4
between two scorings of the same maps.
"""

from __future__ import annotations

import numpy as np

BETA2, BIN_THR = 0.3, 128
EDGE_MAX_DIST, SKELETON_MAX_DIST = 0.0075, 0.01
EDGE_THRESHOLDS, SKELETON_THRESHOLDS = 99, 30


# --------------------------------------------------------------------------------------------- salient / camouflaged

def sod_record(pred: np.ndarray, gt: np.ndarray) -> dict:
    import py_sod_metrics as M

    sm, em, wfm, mae = M.Smeasure(), M.Emeasure(), M.WeightedFmeasure(), M.MAE()
    for m in (sm, em, wfm, mae):
        m.step(pred, gt)
    return {"sm": float(sm.sms[0]), "wfm": float(wfm.weighted_fms[0]), "mae": float(mae.maes[0]),
            "em_adp": float(em.adaptive_ems[0]), "em_curve": np.asarray(em.changeable_ems[0], dtype=np.float64)}


def sod_summary(recs: list[dict]) -> dict:
    curve = np.mean([r["em_curve"] for r in recs], axis=0)
    return {"n": len(recs), "S": float(np.mean([r["sm"] for r in recs])), "maxE": float(curve.max()),
            "meanE": float(curve.mean()), "adpE": float(np.mean([r["em_adp"] for r in recs])),
            "wF": float(np.mean([r["wfm"] for r in recs])), "MAE": float(np.mean([r["mae"] for r in recs]))}


# --------------------------------------------------------------------------------------------- regions

def region_record(pred: np.ndarray, gt: np.ndarray) -> dict:
    g = gt >= 128
    p = pred.astype(np.int64)
    fg, bg = np.bincount(p[g], minlength=256), np.bincount(p[~g], minlength=256)
    return {"fg": fg, "bg": bg, "mae": float(np.abs(pred.astype(np.float64) / 255.0 - g).mean())}


def _at(rec: dict, t: int) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN of p >= t."""
    tp, fp = int(rec["fg"][t:].sum()), int(rec["bg"][t:].sum())
    return tp, fp, int(rec["fg"].sum()) - tp, int(rec["bg"].sum()) - fp


def _summed(recs: list[dict], t: int) -> tuple[int, int, int, int]:
    tot = np.zeros(4, dtype=np.int64)
    for r in recs:
        tot += np.array(_at(r, t), dtype=np.int64)
    return tuple(int(x) for x in tot)


def iou(rec: dict, t: int = BIN_THR) -> float:
    tp, fp, fn, _ = _at(rec, t)
    return 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)


def ber_dataset(recs: list[dict], t: int = BIN_THR) -> dict:
    tp, fp, fn, tn = _summed(recs, t)
    pos, neg = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"BER": 100 * (1 - 0.5 * (pos + neg)), "pos_err": 100 * (1 - pos), "neg_err": 100 * (1 - neg),
            "Acc": (tp + tn) / max(tp + fp + fn + tn, 1)}


def f_beta(p, r, beta2: float = BETA2):
    return np.where(p + r > 0, (1 + beta2) * p * r / np.maximum(beta2 * p + r, 1e-12), 0.0)


def f_curves(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean precision and recall over images at the thresholds t = 0..255 (p >= t)."""
    prec, rec_ = [], []
    for r in recs:
        tp = np.cumsum(r["fg"][::-1])[::-1].astype(np.float64)
        fp = np.cumsum(r["bg"][::-1])[::-1].astype(np.float64)
        prec.append(np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1.0), 0.0))
        rec_.append(tp / max(float(r["fg"].sum()), 1.0))
    return np.mean(prec, 0), np.mean(rec_, 0)


def region_summary(recs: list[dict], t: int = BIN_THR) -> dict:
    """BER and IoU at the binarization threshold t; max / mean F and MAE need none."""
    p, r = f_curves(recs)
    f = f_beta(p, r)
    return {"n": len(recs), "t": t, "IoU": float(np.mean([iou(x, t) for x in recs])),
            "MAE": float(np.mean([x["mae"] for x in recs])), **ber_dataset(recs, t), "maxF": float(f.max()),
            "meanF": float(f.mean())}


# --------------------------------------------------------------------------------------------- mirrors

def msd_record(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-4) -> dict:
    g = gt >= 128
    n_pos, n_neg = np.count_nonzero(g), np.count_nonzero(~g)
    p = pred.astype(np.float64)
    q = (p - p.min()) / (p.max() - p.min() + 1e-8)
    qq = np.ceil(q * 255.0 - 1e-9).astype(np.int64).clip(0, 256)            # q > t/255  <=>  ceil(255 q) > t
    le_fg = np.cumsum(np.bincount(qq[g], minlength=257))[:256]
    le_bg = np.cumsum(np.bincount(qq[~g], minlength=257))[:256]
    tp_t, fp_t = n_pos - le_fg, n_neg - le_bg
    return {**region_record(pred, gt), "msd_mae": float(np.abs(q - g).mean()),
            "prec": tp_t / (tp_t + fp_t + eps), "rec": tp_t / (n_pos + eps)}


def msd_summary(recs: list[dict], t: int = BIN_THR) -> dict:
    """IoU per image at the binarization threshold t (at t = 128 the HetNet script's IoU); max F and MAE of the
    script, on the min-max normalized map."""
    p, r = np.mean([x["prec"] for x in recs], 0), np.mean([x["rec"] for x in recs], 0)
    f = (1 + BETA2) * p * r / (BETA2 * p + r + 1e-12)
    return {"n": len(recs), "t": t, "IoU": float(np.mean([iou(x, t) for x in recs])), "maxF": float(f.max()),
            "MAE": float(np.mean([x["msd_mae"] for x in recs]))}


# --------------------------------------------------------------------------------------------- edges / skeletons

def _boundary_one(sample: dict):
    from pyEdgeEval.common.binary_label.evaluate_boundaries import evaluate_boundaries_threshold_multiple_gts
    from pyEdgeEval.common.utils import check_thresholds

    pred = np.asarray(sample["pred"](), dtype=np.float64)
    gts = sample["gts"]()
    if pred.shape != gts[0].shape:
        raise ValueError(f"{sample['name']}: map {pred.shape} and GT {gts[0].shape} differ")
    return evaluate_boundaries_threshold_multiple_gts(
        thresholds=check_thresholds(sample["thresholds"]), pred=pred, gts=[np.asarray(g, bool) for g in gts],
        max_dist=sample["max_dist"], apply_thinning=True, apply_nms=True)


def ods_ois_ap(preds: list, gts: list, names: list, max_dist: float, thresholds: int, workers: int) -> dict:
    """preds / gts: zero-argument callables that load a map in [0, 1] / a list of boolean GT maps. AP is the BSDS
    definition, the area under the precision-recall curve."""
    from pyEdgeEval.common.binary_label.calculate_metrics import calculate_metrics

    samples = [{"name": n, "pred": p, "gts": g, "max_dist": max_dist, "thresholds": thresholds}
               for n, p, g in zip(names, preds, gts)]
    _, _, overall = calculate_metrics(_boundary_one, thresholds, samples, nproc=workers)
    return {"ODS": float(overall["ODS_f1"]), "OIS": float(overall["OIS_f1"]), "AP": float(overall["AUC"]),
            "ODS_threshold": float(overall["ODS_threshold"])}


def thin_gt(gt: np.ndarray) -> list[np.ndarray]:
    from pyEdgeEval.preprocess import binary_thin

    return [binary_thin(gt >= 128).astype(bool)]

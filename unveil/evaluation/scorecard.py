"""PCM-Bench scoring: per-set metrics on stored maps, task scores and the scorecard row.

Maps: ``<maps>/<set>/<id>.png`` (8-bit, ungated) and ``<maps>/<set>/presence_r*.json`` ({id: presence logit}).
Gate: a map counts as empty when sigmoid(presence) is below its prompt's threshold. Only gated sets need the presence
files, so the maps of another model can be scored as stored (gate "none"). Every test image shows its concept. With
the default gate, the salient and camouflaged sets are scored on the ungated maps and the other sets are gated, where
the released model accepts every image, so its scores are those of the maps before the presence decision, as in the
paper.
A task score averages the primary metric over the task's test sets, x 100 (100 - BER for shadows), from per-set values
rounded to four decimals; the mean is the unweighted mean of the seven task scores.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from functools import partial
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from unveil.evaluation import metrics as Mx
from unveil.evaluation.inference import map_name, read_presence

# (task key, column head, test sets, primary metric)
TASKS = (("salient", "Salient", ("duts_te", "ecssd", "hkuis", "dutomron", "pascals"), "S"),
         ("camouflaged", "Camo.", ("camo", "cod10k", "nc4k"), "S"),
         ("shadow", "Shadow", ("sbu", "istd"), "BER"),
         ("transparent", "Transp.", ("trans10k",), "IoU"),
         ("mirror", "Mirror", ("msd",), "IoU"),
         ("edge", "Edge", ("bsds500",), "ODS"),
         ("skeleton", "Skeleton", ("sklarge", "sympascal"), "ODS"))
SHOW = {"sod": ("S", "maxE", "meanE", "wF", "MAE"), "region": ("BER", "IoU", "maxF", "MAE"),
        "msd": ("IoU", "maxF", "MAE"), "edge": ("ODS", "OIS", "AP"), "skeleton": ("ODS", "OIS", "AP")}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _read(path, on: bool) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return a if on else np.zeros_like(a)


def _pair(path, rec: dict, on: bool) -> tuple[np.ndarray, np.ndarray]:
    from unveil.data import catalog

    pred, gt = _read(path, on), catalog.load_gt(rec)
    if pred.shape != gt.shape:
        raise ValueError(f"{rec['id']}: map {pred.shape} and GT {gt.shape} differ")
    return pred, gt


def _sod(args):
    return Mx.sod_record(*_pair(*args))


def _region(args):
    return Mx.region_record(*_pair(*args))


def _msd(args):
    return Mx.msd_record(*_pair(*args))


def _edge_pred(path, on: bool) -> np.ndarray:
    return _read(path, on).astype(np.float64) / 255.0


def _skel_pred(path, on: bool) -> np.ndarray:
    a = _read(path, on).astype(np.float64)
    return a / a.max() if a.max() > 0 else a


def _skel_gts(rec: dict) -> list[np.ndarray]:
    from unveil.data import catalog

    return Mx.thin_gt(catalog.load_gt(rec))


def _init_worker():
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


def score_set(name: str, family: str, recs: list[dict], map_dir, on: list[bool], workers: int = 12,
              t: int = Mx.BIN_THR) -> dict:
    """Metrics of one test set; on[k] False makes the k-th map empty (presence gate); t: binarization threshold of
    the region and mirror metrics."""
    from unveil.data import catalog

    map_dir, workers = Path(map_dir), max(1, workers)
    preds = [map_dir / map_name(r["id"]) for r in recs]
    missing = [str(p) for p in preds if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} maps missing, e.g. {missing[:2]}")
    ids = [r["id"] for r in recs]
    if family == "edge":
        return Mx.ods_ois_ap([partial(_edge_pred, p, o) for p, o in zip(preds, on)],
                             [partial(catalog.edge_gts, r) for r in recs], ids, Mx.EDGE_MAX_DIST,
                             Mx.EDGE_THRESHOLDS, workers)
    if family == "skeleton":
        return Mx.ods_ois_ap([partial(_skel_pred, p, o) for p, o in zip(preds, on)],
                             [partial(_skel_gts, r) for r in recs], ids, Mx.SKELETON_MAX_DIST,
                             Mx.SKELETON_THRESHOLDS, workers)
    one, summary = {"sod": (_sod, Mx.sod_summary), "region": (_region, Mx.region_summary),
                    "msd": (_msd, Mx.msd_summary)}[family]
    args = list(zip(preds, recs, on))
    if workers <= 1:
        recs_out = [one(x) for x in args]
    else:
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            recs_out = pool.map(one, args, chunksize=8)
    return summary(recs_out) if family == "sod" else summary(recs_out, t)


def gate_of(recs: list[dict], presence: dict[str, float], tau: float) -> list[bool]:
    missing = [r["id"] for r in recs if r["id"] not in presence]
    if missing:
        raise KeyError(f"{len(missing)} images without a presence logit, e.g. {missing[:2]}; gated sets need "
                       "presence_r*.json, and gate 'none' scores the maps as stored")
    return [sigmoid(presence[r["id"]]) >= tau for r in recs]


def score(maps_root, names: list[str], thresholds: dict[str, float] | None = None,
          binarize: dict[str, int] | None = None, workers: int = 12, gate: str = "paper", data_root=None,
          limit: int | None = None) -> dict:
    """Score the given sets. thresholds: presence tau per prompt (+ "free_text"), needed for gated sets only;
    binarize: t per concept (shadow, transparent, mirror), 128 when missing. gate: "paper" (salient / camouflaged sets
    ungated, the other benchmark sets gated, the extra sets ungated), "all" or "none". below_tau counts the maps under
    the presence threshold when presence logits and a threshold exist, else None. limit: the first N images of each
    set (smoke tests)."""
    from unveil.data import catalog
    from unveil.evaluation.binarize import TASK_CONCEPT

    out = {}
    for name in names:
        meta = catalog.TEST_SETS[name]
        recs = catalog.test_records(name, data_root)[:limit]
        d = Path(maps_root) / name
        pres = read_presence(d)
        tau = (thresholds or {}).get(meta["prompt"], (thresholds or {}).get("free_text"))
        gated = {"paper": meta["gated"], "all": True, "none": False}[gate]
        if gated and tau is None:
            raise ValueError(f"{name}: no presence threshold for {meta['prompt']!r}; gate 'none' scores the maps as "
                             "stored")
        known = tau is not None and bool(pres) and all(r["id"] in pres for r in recs)
        above = gate_of(recs, pres, tau) if gated or known else None
        t = (binarize or {}).get(TASK_CONCEPT.get(meta["task"]), Mx.BIN_THR)
        res = score_set(name, meta["family"], recs, d, above if gated else [True] * len(recs), workers, t)
        out[name] = {"prompt": meta["prompt"], "gated": gated, "tau": tau, "n": len(recs),
                     "below_tau": None if above is None else int(len(above) - sum(above)),
                     **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "n"}}
    return out


def task_scores(per_set: dict) -> tuple[dict, float | None]:
    t = {}
    for key, _, sets, metric in TASKS:
        vals = [per_set.get(s, {}).get(metric) for s in sets]
        t[key] = None if any(v is None for v in vals) else mean(100.0 - v if metric == "BER" else 100.0 * v
                                                                for v in vals)
    return t, (mean(t.values()) if all(v is not None for v in t.values()) else None)


def markdown(per_set: dict, tasks: dict, overall: float | None) -> str:
    from unveil.data import catalog

    rows = ["| Set | Prompt | Images | Gate | Metrics |", "| --- | --- | --- | --- | --- |"]
    for name, r in per_set.items():
        meta = catalog.TEST_SETS[name]
        ms = ", ".join(f"{k} {r[k]:.4f}" for k in SHOW[meta["family"]] if k in r)
        below = r.get("below_tau")
        gate = (f"gated ({below} off)" if r["gated"] else "ungated" if below is None
                else f"ungated ({below} below tau)")
        rows.append(f"| {meta['name']} | {r['prompt']} | {r['n']} | {gate} | {ms} |")
    fmt = (lambda v: "--" if v is None else f"{v:.1f}")
    rows += ["", "| " + " | ".join(h for _, h, _, _ in TASKS) + " | Mean |", "|" + " --- |" * (len(TASKS) + 1),
             "| " + " | ".join(fmt(tasks.get(k)) for k, _, _, _ in TASKS) + f" | {fmt(overall)} |"]
    return "\n".join(rows)

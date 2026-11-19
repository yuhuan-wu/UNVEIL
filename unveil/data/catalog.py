"""Dataset catalog: where the data lives, what each test set and training stream holds, how to read ground truth.

Data root. `data_root(root)` returns `root` if given, else $UNVEIL_DATA, else <release>/datasets/data.
`scripts/prepare_data.py` fills it (layout in datasets/README.md). Manifests are small tracked files under
<release>/datasets/manifests; their paths are relative to the data root, and every loader below returns absolute
paths (str) joined with the root.

API
- data_root(root=None) -> Path
- TEST_SETS: OrderedDict name -> {name, prompt, task, family, primary, n, gated, extra}. The 15 sets of the
  benchmark table come first (extra=False), then the 6 extra sets (extra=True, scored ungated).
- test_records(name, root=None) -> [{id, image, gt, gt_mode}] (+ gt_raw for BSDS500), in file-name order.
- load_gt(rec) -> uint8 GT array; load_bsds_gts(rec) -> one bool boundary map per BSDS500 annotator;
  edge_gts(rec) -> the annotator maps of any edge record; gt_hw(rec) -> (h, w) of the map to save.
- aux_records(name, root=None): BSDS500 "bsds500_train" / "bsds500_val" in the test-record format.
- rejection_records(root=None) -> 4,784 queries {qid, set, source, k, cluster, id, image, prompt, w, h}
  (+ category_id, aq_qid for LVIS); a query is answered "none" when the output map is empty.
- retention_records(root=None, split="heldout") -> LVIS pairs {image_id, image, w, h, np, cid, mask, area, split};
  "heldout" (2,294 pairs: held-out images x categories never replayed) or "seen" (6,255).
- TRAIN_STREAMS, train_index(root=None) -> {stream: [item]} in training order (field reference in
  datasets/README.md). `downscale: true` marks items read at longer side 1008 (resize rule in the README).
- prompt_list() -> {"task": 8 prompts (ids 0..7), "replay": 574 noun phrases (index k = prompt id 8 + k),
  "negative_only": 25 noun phrases (ids 582..606)}: the 607 prompts the text encoder encodes, in id order.
- val_pairs(root=None) -> {"single": [...], "joint": [...]}, records {id, image, prompts: [p], presence, kind}
  for fitting the presence thresholds of single queries and of the joint {salient, camouflaged} query.
- calibration_records(root=None) -> {set: [{id, image, gt}]}: the training images of SBU, ISTD, Trans10K and MSD
  for the binarization-threshold calibration (the 18 Trans10K images with rotated masks left out).

GT modes (test records): "L" = the mask as 8-bit gray; "thr128" / "thr129" = (gray >= 128 / 129) x 255;
"any128" = (any of the first three channels >= 128) x 255; "skel_mat" = (.mat symmetry map > 0) x 255;
"bsds_mat" = every annotator's Boundaries map of the .mat file (`load_bsds_gts`).
"""

from __future__ import annotations

import gzip
import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

RELEASE = Path(__file__).resolve().parents[2]
MANIFESTS = RELEASE / "datasets" / "manifests"
DEFAULT_ROOT = RELEASE / "datasets" / "data"
ENV = "UNVEIL_DATA"

# name -> (display name, prompt, task, metric family, primary metric, scored images, presence-gated, extra)
_S, _C = "salient object", "camouflaged object"
TEST_SETS: "OrderedDict[str, dict]" = OrderedDict(
    (k, dict(zip(("name", "prompt", "task", "family", "primary", "n", "gated", "extra"), v))) for k, v in (
        ("duts_te", ("DUTS-TE", _S, "salient", "sod", "S", 5019, False, False)),
        ("ecssd", ("ECSSD", _S, "salient", "sod", "S", 1000, False, False)),
        ("hkuis", ("HKU-IS", _S, "salient", "sod", "S", 4447, False, False)),
        ("dutomron", ("DUT-OMRON", _S, "salient", "sod", "S", 5168, False, False)),
        ("pascals", ("PASCAL-S", _S, "salient", "sod", "S", 850, False, False)),
        ("camo", ("CAMO", _C, "camouflaged", "sod", "S", 250, False, False)),
        ("cod10k", ("COD10K", _C, "camouflaged", "sod", "S", 2026, False, False)),
        ("nc4k", ("NC4K", _C, "camouflaged", "sod", "S", 4121, False, False)),
        ("sbu", ("SBU", "shadow", "shadow", "region", "BER", 638, True, False)),
        ("istd", ("ISTD", "shadow", "shadow", "region", "BER", 540, True, False)),
        ("trans10k", ("Trans10K", "transparent object", "transparent", "region", "IoU", 4428, True, False)),
        ("msd", ("MSD", "mirror", "mirror", "msd", "IoU", 955, True, False)),
        ("bsds500", ("BSDS500", "edges", "edge", "edge", "ODS", 200, True, False)),
        ("sklarge", ("SK-LARGE", "skeleton", "skeleton", "skeleton", "ODS", 745, True, False)),
        ("sympascal", ("SYM-PASCAL", "symmetry axis", "skeleton", "skeleton", "ODS", 788, True, False)),
        ("ucf", ("UCF", "shadow", "shadow", "region", "BER", 110, False, True)),
        ("gsd", ("GSD", "transparent object", "transparent", "msd", "IoU", 813, False, True)),
        ("pmd", ("PMD", "mirror", "mirror", "msd", "IoU", 571, False, True)),
        ("chameleon", ("CHAMELEON", _C, "camouflaged", "sod", "S", 76, False, True)),
        ("biped2", ("BIPEDv2", "edges", "edge", "edge", "ODS", 50, False, True)),
        ("whsymmax", ("WH-SYMMAX", "skeleton", "skeleton", "skeleton", "ODS", 100, False, True)),
    ))
AUX_SETS = ("bsds500_train", "bsds500_val")

# Training streams in the order of the training sampler's groups. Dense groups and their prompt ids:
# sod 0, cod 1, edge 2, shadow 3, transparent 4, mirror 5, skeleton 6 (7 "symmetry axis" for src sympascal),
# switch (0, 1); presence-only: neg_cam 1, neg_sal 0; replay: 8 + np index; replay_neg: the ids in "negs";
# joint: the joint presence module's items.
TRAIN_STREAMS = ("sod", "cod", "edge", "shadow", "transparent", "mirror", "skeleton", "switch", "neg_cam", "neg_sal",
                 "replay", "replay_neg", "joint")
TASK_PROMPTS = ("salient object", "camouflaged object", "edges", "shadow", "transparent object", "mirror",
                "skeleton", "symmetry axis")
PATH_FIELDS = ("image", "gt", "gt2", "gt_raw", "mask")
# Trans10K training images whose mask is the image's annotation rotated by 90 degrees (no EXIF); trained as they
# are (mask stretched onto the image), left out of the threshold calibration.
TRANS10K_SWAPPED = ("150", "1519", "1544", "1557", "2511", "2538", "2545", "2642", "2672", "2706", "2707", "3703",
                    "3751", "3788", "3808", "467", "964", "981")
CALIBRATION = {"sbu_train": ("shadow", "sbu"), "istd_train": ("shadow", "istd"),
               "trans10k_train": ("transparent", "trans10k"), "msd_train": ("mirror", "msd")}


# ---------------------------------------------------------------------------------------------- files

def data_root(root=None) -> Path:
    """The data root (absolute): `root`, else $UNVEIL_DATA, else <release>/datasets/data."""
    return Path(root or os.environ.get(ENV) or DEFAULT_ROOT).expanduser().absolute()


@lru_cache(maxsize=None)
def _lines(rel: str) -> tuple[str, ...]:
    for p in (MANIFESTS / f"{rel}.jsonl", MANIFESTS / f"{rel}.jsonl.gz"):
        if p.exists():
            with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
                return tuple(x for x in f if x.strip())
    raise FileNotFoundError(f"manifest {rel}.jsonl[.gz] not found under {MANIFESTS}")


def _rows(rel: str) -> list[dict]:
    """Records of manifests/<rel>.jsonl (or .jsonl.gz), as stored (relative paths); fresh objects every call."""
    return [json.loads(x) for x in _lines(rel)]


def _join(rec: dict, root: Path) -> dict:
    """A fresh record with its paths made absolute (in place)."""
    for k in PATH_FIELDS:
        if rec.get(k):
            rec[k] = str(root / rec[k])
    if "cats" in rec:
        rec["cats"] = [[k, str(root / m)] for k, m in rec["cats"]]
    return rec


def _load(rel: str, root=None) -> list[dict]:
    r = data_root(root)
    return [_join(x, r) for x in _rows(rel)]


# ---------------------------------------------------------------------------------------------- test sets

def test_records(name: str, root=None) -> list[dict]:
    """Records of the scored images of a test set (TEST_SETS key), in file-name order."""
    if name not in TEST_SETS:
        raise KeyError(f"unknown test set {name!r}; sets: {', '.join(TEST_SETS)}")
    return _load(f"test/{name}", root)


def aux_records(name: str, root=None) -> list[dict]:
    """BSDS500 train / val records ("bsds500_train", "bsds500_val"), for the edge calibration."""
    if name not in AUX_SETS:
        raise KeyError(f"unknown auxiliary set {name!r}; sets: {', '.join(AUX_SETS)}")
    return _load(f"test/{name}", root)


def load_gt(rec: dict) -> np.ndarray:
    """uint8 GT of a region / skeleton record (modes in the module doc)."""
    mode, path = rec["gt_mode"], rec["gt"]
    if path is None:
        raise FileNotFoundError(f"{rec['id']}: no GT file")
    if mode == "L":
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mode in ("thr128", "thr129"):
        a = np.asarray(Image.open(path).convert("L"))
        return np.where(a >= int(mode[3:]), 255, 0).astype(np.uint8)
    if mode == "any128":
        a = np.asarray(Image.open(path))
        rgb = a[..., :3] if a.ndim == 3 else a[..., None]
        return (rgb >= 128).any(-1).astype(np.uint8) * 255
    if mode == "skel_mat":
        import scipy.io as sio

        m = sio.loadmat(path)
        return (m["symmetry" if "symmetry" in m else "skeleton"] > 0).astype(np.uint8) * 255
    raise ValueError(f"{rec['id']}: GT mode {mode!r} has no single mask (BSDS500: use load_bsds_gts)")


def load_bsds_gts(rec: dict) -> list[np.ndarray]:
    """BSDS500: one boolean boundary map per annotator (4 to 8 per image)."""
    import scipy.io as sio

    g = sio.loadmat(rec["gt_raw"])["groundTruth"]
    return [g[0, i]["Boundaries"][0, 0].astype(bool) for i in range(g.shape[1])]


def edge_gts(rec: dict) -> list[np.ndarray]:
    """Annotator maps of an edge record: BSDS500's .mat annotators, else [GT >= 128] (BIPEDv2)."""
    return load_bsds_gts(rec) if rec["gt_mode"] == "bsds_mat" else [load_gt(rec) >= 128]


def gt_hw(rec: dict) -> tuple[int, int]:
    """(h, w) the saved map must have: the GT size, the image size for BSDS500."""
    if rec["gt_mode"] == "bsds_mat":
        with Image.open(rec["image"]) as im:
            return im.height, im.width
    if rec["gt_mode"] == "skel_mat":
        return load_gt(rec).shape
    with Image.open(rec["gt"]) as g:
        return g.height, g.width


# ---------------------------------------------------------------------------------------------- rejection, retention

def rejection_records(root=None) -> list[dict]:
    """The 4,784 rejection queries (COD10K background 784, USC12K without the concept 3,600, LVIS absent 400)."""
    return _load("rejection", root)


def retention_records(root=None, split: str = "heldout") -> list[dict]:
    """LVIS retention pairs of `split` ("heldout" or "seen"), sorted by (image_id, cid); GT = `mask` ({0, 255})."""
    assert split in ("heldout", "seen"), split
    return [r for r in _load("retention", root) if r["split"] == split]


# ---------------------------------------------------------------------------------------------- training

def train_index(root=None) -> dict[str, list[dict]]:
    """Every training stream (TRAIN_STREAMS) in training order, paths absolute."""
    return {s: _load(f"train/{s}", root) for s in TRAIN_STREAMS}


@lru_cache(maxsize=None)
def _prompts() -> dict:
    return json.loads((MANIFESTS / "prompts.json").read_text())


def prompt_list() -> dict[str, list[str]]:
    """{"task", "replay", "negative_only"}: prompt ids run through the three lists in this order."""
    p = _prompts()
    assert tuple(p["task"]) == TASK_PROMPTS
    return {k: list(p[k]) for k in ("task", "replay", "negative_only")}


def val_pairs(root=None) -> dict[str, list[dict]]:
    """Presence validation pairs (never trained on): "single" for single-query thresholds, "joint" for the joint
    query (single minus the USC12K single-kind scenes moved to joint training, plus their cross-concept negatives
    and both prompts of the held-out "both" scenes, kind "usc12k_both")."""
    return {k: _load(f"val/{k}", root) for k in ("single", "joint")}


def calibration_records(root=None) -> dict[str, list[dict]]:
    """Training images of the four thresholded region sets, from the training index (GT = the training GT)."""
    idx = train_index(root)
    out = {}
    for name, (stream, src) in CALIBRATION.items():
        out[name] = [{"id": it["id"], "image": it["image"], "gt": it["gt"]} for it in idx[stream]
                     if it["src"] == src and not (src == "trans10k" and it["id"] in TRANS10K_SWAPPED)]
    return out

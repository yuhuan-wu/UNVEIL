"""LVIS v1 val: the COCO 2017 images the manifests use, and the per-category union masks.

Images: coco/<train2017|val2017>/<12-digit id>.jpg, taken from train2017.zip / val2017.zip when one of the --raw
folders holds them (http://images.cocodataset.org/zips/), else downloaded one by one from images.cocodataset.org
(the `coco_url` of the LVIS annotation file).
Masks: derived/lvis_masks/<12-digit image id>_<category id>.png = the union of every instance of the category on
the image ({0, 255}, image size). An instance is the union of its polygons with at least 3 points, rasterized by
pycocotools (`frPyObjects`, `merge`, `decode`), as the LVIS / COCO API does.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .files import fetch
from .sources import COCO_URL, COCO_ZIPS

ANNOTATIONS = "lvis/lvis_v1_val.json"
MASKS = "derived/lvis_masks"


def _from_zip(zp: Path, names: list[str], root: Path, threads: int) -> int:
    with zipfile.ZipFile(zp) as z:
        off = {i.filename: i.header_offset for i in z.infolist()}
    names = sorted(names, key=lambda n: off[n])
    k = -(-len(names) // threads)

    def shard(ns: list[str]) -> int:
        with zipfile.ZipFile(zp) as z:
            for n in ns:
                dst = root / "coco" / n
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                tmp.write_bytes(z.read(n))
                tmp.replace(dst)
        return len(ns)

    with ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(shard, [names[i:i + k] for i in range(0, len(names), k)]))


def coco_images(root: Path, files: list[str], raw_dirs, download: bool, threads: int = 16) -> dict:
    """Place the coco/ files of `files` (paths relative to the root) that do not exist yet."""
    need = defaultdict(list)
    for r in files:
        if r.startswith("coco/") and not (root / r).exists():
            need[r.split("/")[1]].append(r[len("coco/"):])
    out = {"needed": sum(len(v) for v in need.values()), "from_zip": 0, "downloaded": 0, "missing": 0}
    for split, names in need.items():
        zp = next((Path(d) / COCO_ZIPS[split] for d in raw_dirs if (Path(d) / COCO_ZIPS[split]).is_file()), None)
        if zp is not None:
            out["from_zip"] += _from_zip(zp, names, root, threads)
        elif download:
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda n: fetch(COCO_URL.format(n), root / "coco" / n), names))
            out["downloaded"] += len(names)
        else:
            out["missing"] += len(names)
    return out


def _mask_job(job) -> int:
    import pycocotools.mask as mu

    dst, h, w, polys = job
    rles = []
    for p in polys:                                  # one instance: its valid polygons
        good = [list(map(float, x)) for x in p if len(x) >= 6]
        if good:
            rles.append(mu.merge(mu.frPyObjects(good, h, w)))
    m = mu.decode(mu.merge(rles)) if rles else np.zeros((h, w), np.uint8)
    tmp = Path(dst).with_name(Path(dst).name + ".tmp.png")
    Image.fromarray(m.astype(np.uint8) * 255).save(tmp)
    tmp.replace(dst)
    return 1


def masks(root: Path, files: list[str], workers: int = 8) -> dict:
    """Write the derived/lvis_masks files of `files` that do not exist yet (needs lvis/lvis_v1_val.json)."""
    todo = [r for r in files if r.startswith(MASKS + "/") and not (root / r).exists()]
    if not todo:
        return {"written": 0}
    ann = root / ANNOTATIONS
    if not ann.exists():
        return {"written": 0, "missing_source": len(todo), "missing_examples": [ANNOTATIONS]}
    want = {}
    for r in todo:
        img, cid = Path(r).stem.split("_")
        want[(int(img), int(cid))] = r
    d = json.loads(ann.read_text())
    size = {im["id"]: (im["height"], im["width"]) for im in d["images"]}
    polys = defaultdict(list)
    for a in d["annotations"]:
        key = (a["image_id"], a["category_id"])
        if key in want:
            polys[key].append(a["segmentation"])
    (root / MASKS).mkdir(parents=True, exist_ok=True)
    jobs = [(str(root / r), *size[k[0]], polys[k]) for k, r in want.items()]
    with ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(_mask_job, jobs, chunksize=64))
    return {"written": n, "without_instances": sum(not polys[k] for k in want)}

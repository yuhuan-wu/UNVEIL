"""Prepare the data root from the official downloads: find or download each archive, check its SHA-256, unpack
it, place the LVIS images, run the converters, then check every file the manifests reference.

    run(root, names=("all",), raw=(), download=False, verify=True, workers=8) -> report
    verify(root, names=("all",), workers=8) -> report
    check_urls(names=("all",)) -> report

`names` are dataset keys (`sources.DATASETS`), roles ("train", "test", "extra") or "all". A role selects the
datasets that serve it and only the archives and files that role needs; a dataset key selects all its roles.
Downloads land in <root>/_downloads (reused by a rerun; delete them once the check passes).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import catalog
from . import convert, files, lvis
from .sources import DATASETS, ROLES, dataset_of, select

__all__ = ["run", "verify", "check_urls", "needed_files"]


def plan(names) -> dict[str, set[str]]:
    """{dataset key: roles to prepare}: a role name adds that role to the datasets serving it, a dataset key
    adds all of its roles, "all" adds everything."""
    out: dict[str, set[str]] = {}
    for n in names:
        for key in select([n]):
            out.setdefault(key, set()).update(DATASETS[key].roles if n not in ROLES else {n})
    return out


def manifests_of(roles) -> list[str]:
    """Manifest names a role reads."""
    out = []
    if "train" in roles:
        out += [f"train/{s}" for s in catalog.TRAIN_STREAMS] + ["val/single", "val/joint"]
    if "test" in roles:
        out += [f"test/{n}" for n, s in catalog.TEST_SETS.items() if not s["extra"]]
        out += [f"test/{n}" for n in catalog.AUX_SETS] + ["rejection", "retention"]
    if "extra" in roles:
        out += [f"test/{n}" for n, s in catalog.TEST_SETS.items() if s["extra"]]
    return out


def needed_files(names=("all",)) -> list[str]:
    """Every file (relative to the data root) the selected datasets must provide for their selected roles."""
    want, out = plan(names), set()
    for role in ROLES:
        for m in manifests_of({role}):
            for r in catalog._rows(m):
                for f in [r[k] for k in catalog.PATH_FIELDS if r.get(k)] + [p for _, p in r.get("cats", ())]:
                    if role in want.get(dataset_of(f), ()):
                        out.add(f)
    if any(f.startswith(lvis.MASKS) for f in out):
        out.add(lvis.ANNOTATIONS)                    # the masks' source
    return sorted(out)


# ---------------------------------------------------------------------------------------------- prepare

def _archives(names, root: Path, raw, download: bool, hash_check: bool) -> dict:
    out = {}
    for key, roles in plan(names).items():
        for arc in DATASETS[key].archives:
            if not set(arc.roles) & roles:
                continue
            done = files.marker(root, arc)
            if done.exists():
                out[arc.file] = "unpacked before"
                continue
            path = files.find(arc, [*raw, root / "_downloads"])
            if path is None and download:
                print(f"[prepare] downloading {arc.file} ({arc.size / 2**20:.0f} MB)", flush=True)
                try:
                    path = files.fetch(arc.url, root / "_downloads" / arc.file, arc.size)
                except (PermissionError, RuntimeError) as e:    # go on with the other archives, report at the end
                    out[arc.file] = f"missing: {e}"
                    print(f"[prepare] {key}: {arc.file}: {e}", flush=True)
                    continue
            if path is None:
                out[arc.file] = f"missing: download {arc.url} (see datasets/README.md) and pass its folder with --raw"
                continue
            if hash_check:
                files.check(arc, path)
            if arc.unpack:
                n = files.unpack(arc, path, root)
            else:                                        # parquet test sets
                n = 2 * convert.parquet_test_set(path, root / DATASETS[key].top[0])
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(json.dumps({"archive": str(path), "files": n}) + "\n")
            out[arc.file] = f"unpacked {n} files"
            print(f"[prepare] {key}: {arc.file}: {out[arc.file]}", flush=True)
    return out


def run(root=None, names=("all",), raw=(), download: bool = False, verify_files: bool = True, workers: int = 8,
        hash_check: bool = True) -> dict:
    root = catalog.data_root(root)
    root.mkdir(parents=True, exist_ok=True)
    report = {"root": str(root), "datasets": {k: sorted(v) for k, v in plan(names).items()},
              "archives": _archives(names, root, raw, download, hash_check)}
    need = needed_files(names)
    if "lvis" in report["datasets"]:
        report["coco_images"] = lvis.coco_images(root, need, raw, download)
        report["lvis_masks"] = lvis.masks(root, need, workers)
    report["derived_gt"] = convert.derived_gt(root, [f for f in need if f.startswith("derived/")], workers)
    if verify_files:
        report["verify"] = verify(root, names, workers)
    return report


# ---------------------------------------------------------------------------------------------- verify

def checksums(key: str) -> dict[str, tuple[str, str]]:
    """{path: (kind, digest)} of a dataset; kind "md5" = file bytes, "pix" = decoded pixels (converter outputs)."""
    p = catalog.MANIFESTS / "checksums" / f"{key}.tsv.gz"
    with gzip.open(p, "rt") as f:
        return {r: (k, d) for k, d, r in (x.rstrip("\n").split("\t") for x in f if x.strip())}


def _check(job) -> str:
    path, kind, want = job
    p = Path(path)
    if not p.exists():
        return "missing"
    got = files.md5(p) if kind == "md5" else files.pixel_md5(p)
    return "ok" if got == want else "differs"


def verify(root=None, names=("all",), workers: int = 8) -> dict:
    """Every needed file exists and matches its checksum; per dataset: ok / missing / differs."""
    root = catalog.data_root(root)
    need = needed_files(names)
    sums = {}
    for key in {dataset_of(f) for f in need}:
        sums.update(checksums(key))
    jobs = [(str(root / f), *sums[f]) for f in need]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_check, jobs, chunksize=64))
    per: dict[str, Counter] = {}
    bad = []
    for f, s in zip(need, res):
        per.setdefault(dataset_of(f), Counter())[s] += 1
        if s != "ok":
            bad.append(f"{s}: {f}")
    return {"files": len(need), "ok": res.count("ok"), "per_dataset": {k: dict(v) for k, v in sorted(per.items())},
            "problems": bad[:20], "all_ok": not bad}


# ---------------------------------------------------------------------------------------------- URLs

def check_urls(names=("all",), timeout: int = 60) -> dict:
    """First byte of every selected archive URL: HTTP status, content type and whether the size matches."""
    out = {}
    for key in select(names):
        for arc in DATASETS[key].archives:
            req = urllib.request.Request(arc.url, headers={**files.UA, "Range": "bytes=0-0"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    total = r.headers.get("Content-Range", "/").rsplit("/", 1)[-1]
                    ctype = r.headers.get("Content-Type", "")
                    ok = r.status in (200, 206) and "text/html" not in ctype and \
                        (not total.isdigit() or int(total) == arc.size)
                    out[arc.file] = {"status": r.status, "type": ctype, "size": total, "ok": ok}
            except Exception as e:                   # report, never raise: this is a liveness check
                out[arc.file] = {"error": repr(e)[:200], "ok": False}
            print(f"[urls] {key}/{arc.file}: {out[arc.file]}", flush=True)
    return out

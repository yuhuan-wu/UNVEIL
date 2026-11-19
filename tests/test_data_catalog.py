"""Catalog and manifest integrity (CPU, no data root needed)."""

import gzip
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from unveil.data import catalog

SUMMARY = json.loads((catalog.MANIFESTS / "summary.json").read_text())


def _manifests():
    names = [f"train/{s}" for s in catalog.TRAIN_STREAMS] + ["val/single", "val/joint", "rejection", "retention"]
    return names + [f"test/{n}" for n in list(catalog.TEST_SETS) + list(catalog.AUX_SETS)]


def _paths(rec):
    out = [rec[k] for k in catalog.PATH_FIELDS if rec.get(k)]
    return out + [m for _, m in rec.get("cats", ())]


@pytest.mark.parametrize("name", _manifests())
def test_manifest_sizes_and_relative_paths(name):
    rows = catalog._rows(name)
    assert len(rows) == SUMMARY["lists"][name]
    for r in rows:
        for p in _paths(r):
            assert not p.startswith("/") and ".." not in p.split("/"), p


def test_test_sets_match_their_counts_and_bench_totals():
    by_task = Counter()
    for name, meta in catalog.TEST_SETS.items():
        recs = catalog.test_records(name, "/data")
        assert len(recs) == meta["n"], name
        assert len({r["id"] for r in recs}) == len(recs), name
        assert all(r["image"].startswith("/data/") for r in recs)
        if not meta["extra"]:
            by_task[meta["task"]] += len(recs)
    assert dict(by_task) == {"salient": 16484, "camouflaged": 6397, "shadow": 1178, "transparent": 4428,
                             "mirror": 955, "edge": 200, "skeleton": 1533}
    assert [n for n, m in catalog.TEST_SETS.items() if m["extra"]] == list(catalog.TEST_SETS)[15:]


def test_gt_modes():
    modes = {n: {r["gt_mode"] for r in catalog.test_records(n)} for n in catalog.TEST_SETS}
    assert modes["pascals"] == {"thr129"} and modes["sbu"] == modes["trans10k"] == {"any128"}
    assert modes["bsds500"] == {"bsds_mat"} and modes["sklarge"] == modes["sympascal"] == {"skel_mat"}
    assert all(r["gt_raw"] == r["gt"] for r in catalog.test_records("bsds500"))


def test_rejection_and_retention():
    rej = catalog.rejection_records()
    assert Counter(q["source"] for q in rej) == {"cod10k_bg": 784, "usc12k": 3600, "lvis": 400}
    assert len({q["qid"] for q in rej}) == len(rej)
    held = catalog.retention_records(split="heldout")
    assert len(held) == 2294 and len(catalog.retention_records(split="seen")) == 6255
    assert held == sorted(held, key=lambda r: (r["image_id"], r["cid"]))


def test_prompt_list():
    p = catalog.prompt_list()
    assert tuple(p["task"]) == catalog.TASK_PROMPTS
    assert (len(p["replay"]), len(p["negative_only"])) == (574, 25)
    assert p["replay"] == sorted(p["replay"]) and "mirror" not in p["negative_only"]


def test_train_index_streams():
    idx = catalog.train_index("/data")
    assert tuple(idx) == catalog.TRAIN_STREAMS
    n_prompts = sum(len(v) for v in catalog.prompt_list().values())
    for s, items in idx.items():
        for it in items:
            assert it.get("downscale", True) is True              # present only when true
            assert it["image"].startswith("/data/")
    assert sum(bool(it.get("downscale")) for s in catalog.TRAIN_STREAMS[:8] for it in idx[s]) == 7938
    assert {it["src"] for it in idx["skeleton"]} == {"sklarge", "sympascal"}
    assert all(it["gt"] and it["gt2"] for it in idx["switch"])
    assert all(0 <= k < 574 for it in idx["replay"] for k, _ in it["cats"])
    assert all(0 <= p < n_prompts for it in idx["replay_neg"] for p in it["negs"])
    assert all(it["task"] in ([], [5]) for it in idx["replay_neg"])
    kinds = Counter(it["kind"] for it in idx["joint"])
    assert kinds == {"both": 1920, "single": 2081, "none": 1884, "bg": 2655}
    assert all(it["prompt"] in (0, 1) and it["presence"] == 1 for it in idx["joint"] if it["kind"] == "single")


def test_val_pairs():
    v = catalog.val_pairs()
    assert (len(v["single"]), len(v["joint"])) == (4243, 4628)
    for recs in v.values():
        assert all(set(r) == {"id", "image", "prompts", "presence", "kind"} for r in recs)
        assert len({(r["id"], r["prompts"][0]) for r in recs}) == len(recs)
    assert Counter(r["kind"] for r in v["joint"])["usc12k_both"] % 2 == 0


def test_calibration_records():
    c = catalog.calibration_records("/data")
    assert {k: len(v) for k, v in c.items()} == {"sbu_train": 4085, "istd_train": 1330, "trans10k_train": 4982,
                                                 "msd_train": 3063}


def test_every_referenced_file_has_a_checksum():
    sums = set()
    for p in (catalog.MANIFESTS / "checksums").glob("*.tsv.gz"):
        with gzip.open(p, "rt") as f:
            sums.update(line.rstrip("\n").split("\t")[2] for line in f if line.strip())
    for name in _manifests():
        for r in catalog._rows(name):
            assert set(_paths(r)) <= sums, name


def test_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(catalog.ENV, str(tmp_path))
    assert catalog.data_root() == tmp_path
    assert catalog.data_root("/x") == catalog.Path("/x")
    monkeypatch.delenv(catalog.ENV)
    assert catalog.data_root() == catalog.DEFAULT_ROOT


def test_load_gt_modes(tmp_path):
    a = np.array([[0, 127, 128], [129, 200, 255]], np.uint8)
    Image.fromarray(a).save(tmp_path / "g.png")
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[0, 0, 0] = 200                                          # red "thing"
    rgba[1, 2, :3] = 255                                         # white "stuff"
    Image.fromarray(rgba).save(tmp_path / "t.png")
    rec = {"id": "x", "image": str(tmp_path / "g.png"), "gt": str(tmp_path / "g.png")}
    assert (catalog.load_gt({**rec, "gt_mode": "L"}) == a).all()
    assert (catalog.load_gt({**rec, "gt_mode": "thr128"}) == np.where(a >= 128, 255, 0)).all()
    assert (catalog.load_gt({**rec, "gt_mode": "thr129"}) == np.where(a >= 129, 255, 0)).all()
    any128 = catalog.load_gt({**rec, "gt": str(tmp_path / "t.png"), "gt_mode": "any128"})
    assert any128.tolist() == [[255, 0, 0], [0, 0, 255]]
    assert catalog.gt_hw({**rec, "gt_mode": "L"}) == (2, 3)
    assert [g.tolist() for g in catalog.edge_gts({**rec, "gt_mode": "thr128"})] == [(a >= 128).tolist()]

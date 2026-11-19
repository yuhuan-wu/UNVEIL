"""Preparation pieces on synthetic files (CPU): registry, archive finding and unpacking, converters, file lists."""

import zipfile

import numpy as np
import pytest
from PIL import Image

from unveil.data import catalog, prepare
from unveil.data.prepare import convert, files
from unveil.data.prepare.sources import DATASETS, ROLES, Archive, dataset_of, select


def test_registry_is_consistent():
    tops = [t for d in DATASETS.values() for t in d.top]
    derived = [x for d in DATASETS.values() for x in d.derived]
    assert len(tops) == len(set(tops)) and len(derived) == len(set(derived))
    names = [a.file for d in DATASETS.values() for a in d.archives]
    assert len(names) == len(set(names))
    for d in DATASETS.values():
        assert set(d.roles) <= set(ROLES) and d.terms and d.page
        for a in d.archives:
            assert a.size > 0 and len(a.sha256) == 64 and a.url.startswith("http")
            assert set(a.roles) & set(d.roles)


def test_select_and_roles():
    assert select(["all"]) == list(DATASETS)
    assert "trans10k" in select(["test"]) and "hed_bsds" not in select(["test"])
    assert select(["extra"]) == ["ucf", "gsd", "pmd", "chameleon", "biped2", "whsymmax"]
    assert prepare.plan(["trans10k"]) == {"trans10k": {"train", "test"}}
    mixed = prepare.plan(["test", "duts_tr"])                      # a key does not widen the other datasets
    assert mixed["trans10k"] == {"test"} and mixed["duts_tr"] == {"train"}
    assert not any(f.startswith("Trans10K/train/") for f in prepare.needed_files(["test", "duts_tr"]))
    with pytest.raises(KeyError):
        select(["nope"])


def test_every_needed_file_has_an_owner_and_a_rule():
    need = prepare.needed_files(["all"])
    assert len(need) == len(set(need)) > 200_000
    owners = {dataset_of(f) for f in need}
    assert owners == set(DATASETS)
    derived = {f.split("/", 1)[1].rsplit("/", 1)[0] for f in need if f.startswith("derived/")}
    assert derived == set(convert.RULES) | {"lvis_masks"}
    test_only = prepare.needed_files(["test"])
    assert not any(f.startswith(("HED-BSDS/", "DUTS-TR/", "Trans10K/train/")) for f in test_only)
    assert {f.rsplit("/", 1)[0] for f in test_only if f.startswith("derived/")} == {"derived/lvis_masks"}


def _archive(folder, name="a.zip"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("top/keep/x.txt", "x")
        z.writestr("top/drop/y.txt", "y")
        z.writestr("__MACOSX/top/._x.txt", "junk")
        z.writestr("top/keep/._x.txt", "junk")
    return p


def test_find_check_unpack(tmp_path):
    p = _archive(tmp_path / "dl")                                  # found one folder below a --raw folder too
    arc = Archive("a.zip", "https://example.org/a.zip", files.sha256(p), p.stat().st_size, "here", ("top/keep/",))
    assert files.find(arc, [tmp_path]) == p and files.find(arc, [tmp_path / "dl"]) == p
    assert files.find(Archive("a.zip", arc.url, arc.sha256, arc.size + 1), [tmp_path]) is None
    files.check(arc, p)
    with pytest.raises(ValueError):
        files.check(Archive("a.zip", arc.url, "0" * 64, arc.size), p)
    root = tmp_path / "root"
    assert files.unpack(arc, p, root) == 1
    assert (root / "here/top/keep/x.txt").exists() and not (root / "here/top/drop").exists()
    assert not list(root.rglob("._*")) and files.unpack(arc, p, root) == 1      # second call: marker


def test_pixel_md5_ignores_encoding(tmp_path):
    a = (np.arange(64).reshape(8, 8) * 4).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png", compress_level=1)
    Image.fromarray(a).save(tmp_path / "b.png", compress_level=9)
    assert files.pixel_md5(tmp_path / "a.png") == files.pixel_md5(tmp_path / "b.png")
    assert files.md5(tmp_path / "a.png") != files.md5(tmp_path / "b.png")


def test_converters(tmp_path):
    import scipy.io as sio

    g = np.array([[0, 127], [128, 255]], np.uint8)
    Image.fromarray(g).save(tmp_path / "g.png")
    assert convert.any128(tmp_path / "g.png").tolist() == [[False, False], [True, True]]
    sio.savemat(tmp_path / "s.mat", {"skeleton": np.array([[0, 2], [0, 0]])})
    assert convert.skeleton_mat(tmp_path / "s.mat").tolist() == [[False, True], [False, False]]
    lab = Image.new("P", (2, 2))
    lab.putdata([0, 15, 40, 15])
    pal = [0] * 768
    pal[15 * 3:15 * 3 + 3], pal[40 * 3:40 * 3 + 3] = [255, 0, 0], [0, 255, 0]
    lab.putpalette(pal)
    lab.save(tmp_path / "u.png")
    assert convert.usc12k(tmp_path / "u.png", "sal").tolist() == [[False, True], [False, True]]
    assert convert.usc12k(tmp_path / "u.png", "cam").tolist() == [[False, False], [True, False]]


def test_derived_gt_writes_binary_png(tmp_path):
    src = tmp_path / "Trans10K/train/masks"
    src.mkdir(parents=True)
    m = np.zeros((2, 2, 4), np.uint8)
    m[0, 1, 0] = 255
    Image.fromarray(m).save(src / "7_mask.png")
    out = convert.derived_gt(tmp_path, ["derived/trans10k_gt/train/7.png", "derived/trans10k_gt/train/8.png"], 1)
    assert out["written"] == 1 and out["missing_source"] == 1
    got = np.asarray(Image.open(tmp_path / "derived/trans10k_gt/train/7.png"))
    assert got.tolist() == [[0, 255], [0, 0]]


def test_catalog_records_resolve_against_root(tmp_path):
    rec = catalog.test_records("ecssd", tmp_path)[0]
    assert rec["image"] == str(tmp_path / "ECSSD/images/0001.jpg") and rec["gt_mode"] == "L"

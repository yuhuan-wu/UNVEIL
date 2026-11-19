"""Converters: training ground truth in derived/ and the parquet test sets, from the unpacked official files.

derived/<folder>/<id>.png is a {0, 255} 8-bit gray PNG made from the official annotation:
- sbu_gt/train: SBU train masks are graded (JPEG-like noise); foreground = any colour channel >= 128.
- trans10k_gt/train: Trans10K RGBA masks, red "things" and white "stuff"; foreground = any RGB channel >= 128.
- sklarge_gt/train, sympascal_gt/train: the .mat skeleton map ("symmetry", or "skeleton" in SymPASCAL train) > 0.
- usc12k_gt_train/{sal,cam}: the USC12K class-index PNG; salient = index 1 (palette red), camouflaged = index 2
  (palette green), decoded by palette colour when the PNG has a palette.
GSD / PMD: the authors' parquet files hold the test images and masks; both are written with their original bytes
(<GSD|PMD>/test/image/<image_id>.<jpg|png>, <...>/test/mask/<image_id>.png).
"""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
USC_SAL, USC_CAM = 1, 2
USC_RGB = {"sal": (255, 0, 0), "cam": (0, 255, 0)}


def any128(src: Path) -> np.ndarray:
    a = np.asarray(Image.open(src))
    rgb = a[..., :3] if a.ndim == 3 else a[..., None]
    return (rgb >= 128).any(-1)


def skeleton_mat(src: Path) -> np.ndarray:
    import scipy.io as sio

    m = sio.loadmat(src)
    return m["symmetry" if "symmetry" in m else "skeleton"] > 0


def usc12k(src: Path, which: str) -> np.ndarray:
    with Image.open(src) as im:
        lab, pal = np.asarray(im), (im.getpalette() if im.mode == "P" else None)
    if pal is None:
        return lab == (USC_SAL if which == "sal" else USC_CAM)
    p = np.asarray(pal, dtype=np.int32)[: 3 * (len(pal) // 3)].reshape(-1, 3)
    return np.isin(lab, np.flatnonzero((p == USC_RGB[which]).all(1)))


# derived folder -> (source folder, suffix stripped from source stems, converter)
RULES = {
    "sbu_gt/train": ("SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks", "", any128),
    "trans10k_gt/train": ("Trans10K/train/masks", "_mask", any128),
    "sklarge_gt/train": ("SK-LARGE/groundTruth/train", "", skeleton_mat),
    "sympascal_gt/train": ("SymPASCAL-by-KZ/groundTruth/train", "", skeleton_mat),
    "usc12k_gt_train/sal": ("VOC-USC12K/SegmentationClass", "", lambda p: usc12k(p, "sal")),
    "usc12k_gt_train/cam": ("VOC-USC12K/SegmentationClass", "", lambda p: usc12k(p, "cam")),
}


def _sources(d: Path, strip: str) -> dict[str, Path]:
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG_EXT + (".mat",) and not p.name.startswith("._"):
            out[p.stem[: len(p.stem) - len(strip)] if strip and p.stem.endswith(strip) else p.stem] = p
    return out


def _convert_one(job) -> str:
    folder, src, dst = job
    mask = RULES[folder][2](Path(src))
    tmp = Path(dst).with_name(Path(dst).name + ".tmp.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(tmp)
    tmp.replace(dst)
    return dst


def derived_gt(root: Path, files: list[str], workers: int = 8) -> dict:
    """Write the derived/ GT files in `files` (paths relative to the root) that do not exist yet."""
    jobs, missing = [], []
    by_folder: dict[str, list[str]] = {}
    for r in files:
        folder = r.split("/", 1)[1].rsplit("/", 1)[0]
        if folder in RULES and not (root / r).exists():
            by_folder.setdefault(folder, []).append(r)
    for folder, rs in by_folder.items():
        src_dir, strip, _ = RULES[folder]
        if not (root / src_dir).is_dir():
            missing += rs
            continue
        src = _sources(root / src_dir, strip)
        (root / "derived" / folder).mkdir(parents=True, exist_ok=True)
        for r in rs:
            stem = Path(r).stem
            if stem in src:
                jobs.append((folder, str(src[stem]), str(root / r)))
            else:
                missing.append(r)
    with ProcessPoolExecutor(workers) as ex:
        done = list(ex.map(_convert_one, jobs, chunksize=32))
    return {"written": len(done), "missing_source": len(missing), "missing_examples": missing[:5]}


def parquet_test_set(path: Path, top: Path) -> int:
    """GSD / PMD: write every row's image and mask bytes under `top`/test/{image,mask}/ (skips existing files)."""
    import pyarrow.parquet as pq

    (top / "test" / "image").mkdir(parents=True, exist_ok=True)
    (top / "test" / "mask").mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(path, columns=["image_id", "image", "mask"]).to_pylist()
    for row in rows:
        b = row["image"]["bytes"]
        ext = {"JPEG": ".jpg", "PNG": ".png"}.get(Image.open(io.BytesIO(b)).format, ".png")
        for dst, data in ((top / "test" / "image" / f"{row['image_id']}{ext}", b),
                          (top / "test" / "mask" / f"{row['image_id']}.png", row["mask"]["bytes"])):
            if not dst.exists():
                tmp = dst.with_name(dst.name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(dst)
    return len(rows)

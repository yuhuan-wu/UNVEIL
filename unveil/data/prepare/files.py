"""Archives: find them in the reader's folders, download them, check them, unpack them; file digests.

An unpacked archive leaves a marker <root>/.unveil/unpacked/<archive>.json, so a rerun skips it. Unpacking keeps the
archive's own folders, skips macOS metadata (__MACOSX/, ._*) and, when the archive lists member prefixes to keep,
everything else. A .zip inside a .zip is unpacked next to itself.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from .sources import Archive

CHUNK = 1 << 20
UA = {"User-Agent": "unveil-prepare-data/1.0"}


# ---------------------------------------------------------------------------------------------- digests

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * CHUNK):
            h.update(b)
    return h.hexdigest()


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def pixel_md5(path) -> str:
    """MD5 of the decoded image (mode, size and pixels): equal for two PNG encodings of the same pixels."""
    with Image.open(path) as im:
        im.load()
        return hashlib.md5(f"{im.mode} {im.width} {im.height}\n".encode() + im.tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- finding

def find(arc: Archive, dirs) -> Path | None:
    """An archive of the right size in one of `dirs` (or one folder below), under its own name or the URL's."""
    names = {arc.file, arc.url.rsplit("/", 1)[-1].split("?")[0]}
    for d in map(Path, dirs):
        for cand in [d / n for n in names] + [p / n for p in sorted(d.glob("*/")) for n in names]:
            if cand.is_file() and cand.stat().st_size == arc.size:
                return cand
    return None


def check(arc: Archive, path: Path) -> None:
    """Size and (when known) SHA-256 of an archive."""
    if path.stat().st_size != arc.size:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {arc.size}")
    if arc.sha256 and (got := sha256(path)) != arc.sha256:
        raise ValueError(f"{path}: sha256 {got}, expected {arc.sha256}")


# ---------------------------------------------------------------------------------------------- download

def fetch(url: str, dest: Path, size: int | None = None, retries: int = 5) -> Path:
    """Download `url` to `dest`, resuming a partial file; the size must equal `size` (else the server's
    Content-Length). Refuses an HTML page (a Drive quota or login page)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:          # finished before an interruption
            part.replace(dest)
            return dest
        headers = {**UA, **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise PermissionError(f"{url} returned a web page (quota, login or a moved file); "
                                          "download it in a browser and pass its folder with --raw")
                resumed = have and r.status == 206
                total = r.headers.get("Content-Range", "/").rsplit("/", 1)[-1] if resumed else \
                    r.headers.get("Content-Length", "")
                want = size if size is not None else int(total) if total.isdigit() else None
                with open(part, "ab" if resumed else "wb") as f:
                    shutil.copyfileobj(r, f, CHUNK)
            if want is None or part.stat().st_size == want:
                part.replace(dest)
                return dest
        except PermissionError:
            raise
        except Exception as e:                     # network hiccup: resume from the new offset
            print(f"[download] {dest.name}: attempt {attempt + 1} failed ({e})", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"{url}: download incomplete after {retries} attempts")


# ---------------------------------------------------------------------------------------------- unpacking

def _keep(name: str, keep: tuple[str, ...]) -> bool:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    if "__MACOSX" in name or base.startswith("._"):
        return False
    return not keep or name.startswith(keep)


def _unzip(path: Path, dest: Path, keep: tuple[str, ...]) -> int:
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or not _keep(info.filename, keep):
                continue
            inner = Path(z.extract(info, dest))       # zipfile drops absolute and ".." parts
            n += 1
            if inner.suffix == ".zip":
                n += _unzip(inner, inner.parent, ())
    return n


def _untar(path: Path, dest: Path, keep: tuple[str, ...]) -> int:
    with tarfile.open(path) as t:
        members = [m for m in t.getmembers() if m.isfile() and _keep(m.name, keep)
                   and not m.name.startswith("/") and ".." not in m.name.split("/")]
        if hasattr(tarfile, "data_filter"):           # Python 3.10.12+, 3.11.4+, 3.12+
            t.extractall(dest, members=members, filter="data")
        else:
            t.extractall(dest, members=members)
    return len(members)


def _rar_tool() -> list[str]:
    """bsdtar (libarchive), unrar or 7z, whichever is installed."""
    here = Path(sys.executable).parent
    for name, args in (("bsdtar", ["-xf"]), ("unrar", ["x", "-o+", "-idq"]), ("7z", ["x", "-y", "-bd"])):
        exe = shutil.which(name) or (str(here / name) if (here / name).exists() else None)
        if exe:
            return [exe, *args]
    raise RuntimeError("a .rar archive needs bsdtar (libarchive), unrar or 7z on PATH")


def _unrar(path: Path, dest: Path, keep: tuple[str, ...]) -> int:
    """Unpack into a scratch folder next to `dest`, then move the kept members into `dest`."""
    tmp = dest / ".unveil-rar"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tool = _rar_tool()
    exe = Path(tool[0]).name
    cmd = tool + ([str(path), "-C", str(tmp)] if exe == "bsdtar" else
                  [str(path), str(tmp) + "/"] if exe == "unrar" else [str(path), f"-o{tmp}"])
    subprocess.run(cmd, check=True)
    n = 0
    for f in sorted(p for p in tmp.rglob("*") if p.is_file()):
        name = f.relative_to(tmp).as_posix()
        if _keep(name, keep):
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            f.replace(dest / name)
            n += 1
    shutil.rmtree(tmp)
    return n


def marker(root: Path, arc: Archive) -> Path:
    return root / ".unveil" / "unpacked" / f"{arc.file}.json"


def unpack(arc: Archive, path: Path, root: Path) -> int:
    """Unpack `path` into root/arc.dest (once); returns the number of files written."""
    m = marker(root, arc)
    if m.exists():
        return json.loads(m.read_text())["files"]
    dest = root / arc.dest
    dest.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith(".zip"):
        n = _unzip(path, dest, arc.keep)
    elif name.endswith(".rar"):
        n = _unrar(path, dest, arc.keep)
    elif name.endswith((".tar", ".tgz", ".tar.gz", ".tar.xz", ".tar.bz2")):
        n = _untar(path, dest, arc.keep)
    else:
        raise ValueError(f"{path}: unknown archive type")
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps({"archive": str(path), "files": n, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return n

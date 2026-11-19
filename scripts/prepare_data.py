"""Prepare the UNVEIL data root: official archives -> unpacked folders + converted ground truth, then a check of
every file the training index and the test sets reference (datasets/README.md lists sources, terms and archives).

    python scripts/prepare_data.py --root $UNVEIL_DATA --datasets all --raw ~/downloads [--download]
    python scripts/prepare_data.py --datasets test --verify-only       # check an existing root
    python scripts/prepare_data.py --datasets all --check-urls         # are the download links alive?
    python scripts/prepare_data.py --list                              # dataset keys, roles, archives

--datasets takes dataset keys, roles (train, test, extra) or all. --raw folders (repeatable) are searched for the
archives under their file names; --download fetches what is not found. The exit status is 0 only if every
archive was found and every needed file is present and matches its checksum.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unveil.data import prepare  # noqa: E402
from unveil.data.prepare.sources import DATASETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="data root (default: $UNVEIL_DATA, else datasets/data)")
    ap.add_argument("--datasets", nargs="+", default=["all"], help="dataset keys, train / test / extra, or all")
    ap.add_argument("--raw", action="append", default=[], help="folder holding downloaded archives (repeatable)")
    ap.add_argument("--download", action="store_true", help="download the archives not found in --raw")
    ap.add_argument("--workers", type=int, default=8, help="processes for converting and checking")
    ap.add_argument("--skip-sha256", action="store_true", help="do not hash the archives before unpacking")
    ap.add_argument("--no-verify", action="store_true", help="skip the final file check")
    ap.add_argument("--verify-only", action="store_true", help="only check an existing data root")
    ap.add_argument("--check-urls", action="store_true", help="only probe the download links")
    ap.add_argument("--list", action="store_true", help="list the datasets and exit")
    ap.add_argument("--report", type=Path, help="write the report here (JSON)")
    a = ap.parse_args()
    if a.list:
        for d in DATASETS.values():
            print(f"{d.key:10s} {','.join(d.roles):16s} {d.title}")
            for arc in d.archives:
                print(f"{'':10s} {arc.file} ({arc.size / 2**20:.0f} MB) {arc.url}")
        return 0
    if a.check_urls:
        rep = prepare.check_urls(a.datasets)
        ok = all(v["ok"] for v in rep.values())
    elif a.verify_only:
        rep = prepare.verify(a.root, a.datasets, a.workers)
        ok = rep["all_ok"]
    else:
        rep = prepare.run(a.root, a.datasets, a.raw, a.download, not a.no_verify, a.workers, not a.skip_sha256)
        ok = not any(str(v).startswith("missing") for v in rep["archives"].values())
        ok = ok and (a.no_verify or rep["verify"]["all_ok"])
    text = json.dumps(rep, indent=1)
    if a.report:
        a.report.write_text(text + "\n")
    print(text[-3000:])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Scoring the stored maps of another model (CPU): thresholds from its training maps, scores without presence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from unveil import checkpoint as C
from unveil.data import catalog
from unveil.evaluation import binarize as Bz
from unveil.evaluation import scorecard as SC
from unveil.evaluation.inference import map_name

ROOT = Path(__file__).resolve().parents[1]
N = 3                                   # images per set
FG, BG = 60, 20                         # map values on GT foreground / background: t = 21..60 separate them
SETS = ("sbu", "istd", "trans10k", "msd")


def _evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(recs: list[dict], maps_dir: Path) -> None:
    """A GT block at each record's GT path and the model's map next to it, FG on the block and BG elsewhere."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    for r in recs:
        gt = np.zeros((12, 16), np.uint8)
        gt[3:9, 4:12] = 255
        Path(r["gt"]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gt).save(r["gt"])
        Image.fromarray(np.where(gt > 0, FG, BG).astype(np.uint8)).save(maps_dir / map_name(r["id"]))


@pytest.fixture
def other(tmp_path, monkeypatch):
    """An output folder with another model's maps (no presence files, no thresholds.json) and a data root with GT."""
    out = tmp_path / "other"
    monkeypatch.setenv(catalog.ENV, str(tmp_path / "data"))
    cal = {s: r[:N] for s, r in catalog.calibration_records().items()}
    for s, recs in cal.items():
        _write(recs, out / "maps" / s)
    for s in SETS:
        _write(catalog.test_records(s)[:N], out / "maps" / s)
    monkeypatch.setattr(catalog, "calibration_records", lambda root=None: cal)
    monkeypatch.setattr(C, "read_meta", lambda *a, **k: pytest.fail("the UNVEIL checkpoint was read"))
    return out


def test_calibrate_maps_fits_the_separating_threshold(other):
    res = Bz.calibrate_maps(other / "maps", catalog.calibration_records(), workers=1)
    assert {c: res[c] for c in Bz.CONCEPT_SETS} == {"shadow": FG, "transparent": FG, "mirror": FG}  # ties -> 128
    part = {k: v for k, v in catalog.calibration_records().items() if k != "istd_train"}
    res = Bz.calibrate_maps(other / "maps", part, workers=1)
    assert "shadow" not in res and res["mirror"] == FG


def test_calibrate_stage_then_score_without_presence(other, capsys):
    ev = _evaluate()
    assert ev.main(["--out", str(other), "--stage", "calibrate", "--workers", "1"]) == 0
    th = json.loads((other / "thresholds.json").read_text())
    assert th == {"binarize": {"shadow": FG, "transparent": FG, "mirror": FG}}
    res = SC.score(other / "maps", list(SETS), th.get("presence"), th["binarize"], workers=1, gate="none", limit=N)
    assert all(r["below_tau"] is None and r["IoU"] == 1.0 for r in res.values())
    assert res["sbu"]["BER"] == 0.0 and res["istd"]["BER"] == 0.0
    assert ev.main(["--out", str(other), "--stage", "score", "--gate", "none", "--sets", *SETS, "--limit", str(N),
                    "--workers", "1"]) == 0
    assert "| Shadow |" in capsys.readouterr().out


def test_score_needs_thresholds_json_but_not_the_checkpoint(other):
    ev = _evaluate()
    argv = ["--out", str(other), "--stage", "score", "--gate", "none", "--sets", "msd", "--limit", str(N),
            "--workers", "1"]
    with pytest.raises(SystemExit, match="thresholds.json not found"):
        ev.main(argv)
    (other / "thresholds.json").write_text(json.dumps({"binarize": {"mirror": FG}}))
    assert ev.main(argv) == 0


def test_gated_sets_still_need_presence(other):
    with pytest.raises(KeyError, match="presence"):
        SC.score(other / "maps", ["msd"], {"mirror": 0.5}, {"mirror": FG}, workers=1, gate="all", limit=N)
    with pytest.raises(ValueError, match="presence threshold"):
        SC.score(other / "maps", ["msd"], None, {"mirror": FG}, workers=1, gate="all", limit=N)


def test_unveil_runs_keep_the_checkpoint_thresholds(other, monkeypatch):
    ev = _evaluate()
    meta = {"thresholds": {"single": {"mirror": 0.28}, "free_text": 0.28}, "binarize": {"mirror": 132}}
    monkeypatch.setattr(C, "read_meta", lambda path: meta)
    a = SimpleNamespace(out=other, checkpoint=Path("unveil_v1.safetensors"), checkpoint_given=False)
    with pytest.raises(SystemExit):
        ev.scoring_thresholds(a)
    (other / "run.json").write_text("{}")                       # written by UNVEIL's own inference
    assert ev.scoring_thresholds(a) == {"presence": {"mirror": 0.28, "free_text": 0.28}, "binarize": {"mirror": 132}}

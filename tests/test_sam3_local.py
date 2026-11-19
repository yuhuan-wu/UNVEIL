"""Where SAM 3's checkpoint comes from: the argument, UNVEIL_SAM3, checkpoints/sam3.pt, else the download."""

import sys
import types
from pathlib import Path

import pytest
import torch

from unveil.model import sam3 as S

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def box(monkeypatch, tmp_path):
    """No environment variable, and a repository default inside tmp_path."""
    monkeypatch.delenv(S.SAM3_ENV, raising=False)
    monkeypatch.setattr(S, "SAM3_LOCAL", tmp_path / "checkpoints" / "sam3.pt")
    return tmp_path


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0")
    return path


def test_default_is_in_the_repository():
    assert S.SAM3_LOCAL == ROOT / "checkpoints" / "sam3.pt"


def test_none_without_a_local_file(box):
    assert S.local_checkpoint() is None


def test_order(box, monkeypatch):
    repo, env, arg = touch(S.SAM3_LOCAL), touch(box / "env" / "sam3.pt"), touch(box / "arg" / "sam3.pt")
    assert S.local_checkpoint() == str(repo)
    monkeypatch.setenv(S.SAM3_ENV, str(env))
    assert S.local_checkpoint() == str(env)
    assert S.local_checkpoint(arg) == str(arg)


def test_missing_file_is_an_error(box, monkeypatch):
    touch(S.SAM3_LOCAL)                     # a wrong path never falls back to another file
    with pytest.raises(FileNotFoundError, match="checkpoints/README.md"):
        S.local_checkpoint(box / "none.pt")
    monkeypatch.setenv(S.SAM3_ENV, str(box / "none.pt"))
    with pytest.raises(FileNotFoundError, match="none.pt"):
        S.local_checkpoint()


def test_builder_gets_the_local_file(box, monkeypatch):
    calls = []

    def build_sam3_image_model(**kw):
        calls.append(kw)
        return torch.nn.Linear(1, 1)

    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = build_sam3_image_model
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)

    S.build_sam3("cpu")
    assert calls[-1]["checkpoint_path"] is None and calls[-1]["load_from_HF"] is True
    local = touch(S.SAM3_LOCAL)
    S.build_sam3("cpu")
    assert calls[-1]["checkpoint_path"] == str(local) and calls[-1]["load_from_HF"] is False
    other = touch(box / "other.pt")
    S.build_sam3("cpu", str(other))
    assert calls[-1]["checkpoint_path"] == str(other) and calls[-1]["load_from_HF"] is False


def test_docs_name_the_three_ways():
    notes = (ROOT / "checkpoints" / "README.md").read_text()
    for word in ("--sam3-checkpoint", "sam3_checkpoint=", S.SAM3_ENV, "checkpoints/sam3.pt"):
        assert word in notes, word
    assert "checkpoints/sam3.pt" in (ROOT / "README.md").read_text()

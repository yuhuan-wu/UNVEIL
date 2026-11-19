"""Checkpoint format: tensor groups, metadata, sha256 file."""

import pytest
import torch

from unveil import checkpoint as C
from unveil import prompts as P


def test_roundtrip(tmp_path):
    t = C.assemble({"encoder.w": torch.ones(2)}, {"blocks.0.attn.qkv.lora_A": torch.zeros(1, 2)},
                   {"query_embed.weight": torch.ones(1)}, torch.zeros(2, 256))
    th = {"single": {p: 0.28 for p in P.TASK_PROMPTS}, "joint": {p: 0.5 for p in P.JOINT_PAIR}, "free_text": 0.28}
    path = tmp_path / "m.safetensors"
    digest = C.save(path, t, C.make_meta(th, extra={"binarize": {"shadow": 45}}))
    back, meta, _ = C.load(path)
    assert set(back) == set(t) and all(torch.equal(back[k], t[k]) for k in t)
    assert meta["binarize"] == {"shadow": 45} and meta["prompts"] == list(P.TASK_PROMPTS)
    heads = C.group(back, "heads.")
    assert list(heads) == ["encoder.w"] and torch.equal(heads["encoder.w"], t["heads.encoder.w"])
    assert C.verify(path) == digest
    path.with_name(path.name + ".sha256").write_text("0" * 64 + "  m.safetensors\n")
    with pytest.raises(ValueError):
        C.load(path)


def test_prompts():
    assert P.task_id(" Salient   Object ") == 0 and P.task_id("dog") is None
    assert P.is_joint_pair(["camouflaged object", "salient object"]) and not P.is_joint_pair(["shadow", "mirror"])

"""After training: fit the thresholds on training data and write the release checkpoint.

The trained tensors (``final.pt``) are loaded into the inference model. (1) Presence thresholds: single-query and
joint-query presence logits on the validation pairs of the training splits (``catalog.val_pairs``), fitted by the
rule of ``evaluation.thresholds``. (2) Binarization thresholds of the region metrics: the gated maps of the SBU, ISTD,
Trans10K and MSD training images (``catalog.calibration_records``), fitted by the rule of ``evaluation.binarize``.
The checkpoint holds the tensors and both kinds of threshold. No test image is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from unveil import checkpoint as C
from unveil.evaluation import binarize as Bz
from unveil.evaluation import inference as I
from unveil.evaluation import thresholds as T

CONCEPT_PROMPT = {"shadow": "shadow", "transparent": "transparent object", "mirror": "mirror"}


def tensors_of(final: dict) -> dict:
    net = final["net"]
    return C.assemble(net["heads"], net["lora"], final["joint_decoder"], net["convention"])


def _segments(recs: list[dict]) -> list[list[dict]]:
    """The joint validation pairs in the three lists the paper's run inferred separately: the validation pairs, the
    cross-concept negatives and the held-out USC12K scenes with both concepts."""
    parts = {"usc12k_xcon": [], "usc12k_both": {}, "val": []}
    for r in recs:
        if r["kind"] == "usc12k_both":                   # one record per scene, both prompts (one image, two pairs)
            parts["usc12k_both"].setdefault(r["id"], {**r, "prompts": []})["prompts"] += r["prompts"]
        else:
            parts[r["kind"] if r["kind"] in parts else "val"].append(r)
    return [p for p in (parts["val"], parts["usc12k_xcon"], list(parts["usc12k_both"].values())) if p]


def binarization(model, records: dict[str, list[dict]], batch: int = 8, workers: int = 8) -> dict:
    """records: training set -> [{id, image, gt}] (sbu_train, istd_train, trans10k_train, msd_train)."""
    hists = {}
    for name, recs in records.items():
        concept = next(c for c, sets in Bz.CONCEPT_SETS.items() if name in sets)
        prompt = CONCEPT_PROMPT[concept]
        hists[name] = I.histograms(model, recs, prompt, model.threshold(prompt), batch, workers)
    return Bz.calibrate(hists)


def calibrate(final_path, out_path, val_pairs: dict | None = None, calibration: dict | None = None,
              sam3_checkpoint: str | None = None, batch: int = 8, workers: int = 8, shards: int = 8,
              device: str = "cuda") -> dict:
    """shards: batches of the validation pairs as in the paper's 8-GPU runs (``inference.presence_logits``)."""
    from unveil.model.sam3 import build_sam3, set_tf32
    from unveil.model.unveil import UNVEIL

    if val_pairs is None or calibration is None:
        from unveil.data import catalog

        val_pairs = val_pairs or catalog.val_pairs()
        calibration = calibration or catalog.calibration_records()
    final = torch.load(final_path, map_location="cpu")
    tensors = tensors_of(final)
    lora = {"r": final["config"]["lora"]["r"], "alpha": final["config"]["lora"]["alpha"],
            "first_block": final["config"]["lora"]["first_block"]}
    placeholder = {"single": {}, "joint": {"salient object": 0.5, "camouflaged object": 0.5}, "free_text": 0.5}
    sam3 = build_sam3(device, sam3_checkpoint)
    set_tf32(False)
    model = UNVEIL(sam3, tensors, C.make_meta(placeholder, lora), device)
    single = T.fit(val_pairs["single"], I.presence_logits(model, val_pairs["single"], False, batch, workers, shards))
    joint_logits = [x for part in _segments(val_pairs["joint"])
                    for x in I.presence_logits(model, part, True, batch, workers, shards)]
    joint = T.fit(val_pairs["joint"], joint_logits)
    th = T.checkpoint_thresholds(single, joint)
    model.tau, model.tau_free = dict(th["single"]), th["free_text"]          # gate the calibration maps with them
    bz = binarization(model, calibration, batch, workers)
    params = {g: int(sum(v.numel() for k, v in tensors.items() if k.startswith(g)))
              for g in ("heads.", "lora.", "joint.decoder.", "joint.convention")}
    binarize = {c: bz[c] for c in Bz.CONCEPT_SETS}
    digest = C.save(out_path, tensors, C.make_meta(th, lora, {"params": params, "binarize": binarize}))
    report = {"checkpoint": str(out_path), "sha256": digest, "thresholds": th, "binarize": bz,
              "fits": {"single": single, "joint": joint}}
    Path(out_path).with_suffix(".calibration.json").write_text(json.dumps(report, indent=1))
    return report

"""Evaluate a checkpoint on PCM-Bench: maps for every test image, then the scorecard (paper, Tab. 3).

    torchrun --nproc_per_node 8 scripts/evaluate.py --out outputs/eval              # infer + score
    python scripts/evaluate.py --out outputs/eval --sets sbu istd                    # one GPU, two sets
    python scripts/evaluate.py --out outputs/eval --stage score                      # score stored maps (CPU)
    python scripts/evaluate.py --out outputs/eval --stage decisions                  # presence decisions (Tab. 6)
    python scripts/evaluate.py --out outputs/eval --stage retention                  # LVIS categories left out
    python scripts/evaluate.py --out outputs/other --stage calibrate                 # another model: thresholds.json
    python scripts/evaluate.py --out outputs/other --stage score --gate none         # another model: scorecard

Inference writes ``<out>/maps/<set>/<id>.png`` (ungated 8-bit maps) and the presence logits; scoring writes
``<out>/scorecard.json`` and prints the table. The paper ran 8 GPUs, each on the images k with k % 8 == rank in
batches of 8. ``--shards 8`` (default) forms the same batches on any number of GPUs, so the maps are those of the
paper; batch composition otherwise moves bf16 results in the last bits.

The maps of another model are scored the same way (README, "Score another model"). ``--stage calibrate`` fits their
binarization thresholds on their maps of the training images (``<out>/maps/sbu_train/`` and likewise ``istd_train``,
``trans10k_train``, ``msd_train``) and writes ``<out>/thresholds.json``, which scoring reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_run(a, rank: int = 0, create: bool = True) -> None:
    """One --out directory holds the outputs of one checkpoint, one SAM 3 and one batching; mixing them is refused."""
    from unveil.checkpoint import sha256_file
    from unveil.model.sam3 import local_checkpoint

    want = {"checkpoint_sha256": sha256_file(a.checkpoint),
            "sam3": local_checkpoint(a.sam3_checkpoint) or "hf:facebook/sam3", "shards": a.shards, "batch": a.batch}
    f = a.out / "run.json"
    if f.exists():
        have = json.loads(f.read_text())
        if have != want:
            raise SystemExit(f"{a.out} holds the outputs of another run ({have}); use a new --out")
    elif not create:
        raise SystemExit(f"{a.out} has no run.json: run the inference first (--stage infer)")
    elif rank == 0:
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(want, indent=1) + "\n")
        os.replace(tmp, f)


def infer(a, names: list[str]) -> None:
    from unveil.data import catalog
    from unveil.evaluation import inference as I
    import unveil

    rank, world = I.dist_info()
    check_run(a, rank)
    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    for name in names:
        meta, t0 = catalog.TEST_SETS[name], time.time()
        recs = catalog.test_records(name)
        if a.limit:
            recs = recs[:a.limit]
        n = 0
        for shard in range(rank, a.shards, world):          # this GPU's shards of the paper's 8-GPU layout
            st = I.write_maps(model, recs, meta["prompt"], a.out / "maps" / name, catalog.gt_hw, shard, a.shards,
                              a.batch, a.workers)
            n += st["images"]
        if rank == 0:
            print(f"[infer] {name}: {n} images on GPU 0 ({time.time() - t0:.0f} s)", flush=True)
    if rank == 0:
        th = {"presence": {**model.tau, "free_text": model.tau_free}, "binarize": model.binarize}
        (a.out / "thresholds.json").write_text(json.dumps(th, indent=1))
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


def scoring_thresholds(a) -> dict:
    """``<out>/thresholds.json`` (written by the inference, or by ``--stage calibrate`` for another model's maps).
    Without it, the checkpoint's thresholds, but only for UNVEIL's own runs (``run.json``) or an explicit
    ``--checkpoint``; the maps of another model never fall back to UNVEIL's thresholds."""
    th_f = a.out / "thresholds.json"
    if th_f.exists():
        th = json.loads(th_f.read_text())
        return {"presence": th.get("presence", {}), "binarize": th.get("binarize", {})}
    if (a.out / "run.json").exists() or a.checkpoint_given:
        from unveil.checkpoint import read_meta

        meta = read_meta(a.checkpoint)
        return {"presence": {**meta["thresholds"]["single"], "free_text": meta["thresholds"]["free_text"]},
                "binarize": meta.get("binarize", {})}
    raise SystemExit(f"{th_f} not found. For the maps of another model, fit the binarization thresholds with "
                     "--stage calibrate or write the file (README, \"Score another model\"); for UNVEIL's maps, "
                     "pass --checkpoint.")


def score(a, names: list[str]) -> dict:
    from unveil.data import catalog
    from unveil.evaluation import scorecard as SC
    from unveil.evaluation.binarize import TASK_CONCEPT

    th = scoring_thresholds(a)
    no_t = sorted({TASK_CONCEPT[catalog.TEST_SETS[n]["task"]] for n in names
                   if catalog.TEST_SETS[n]["task"] in TASK_CONCEPT} - set(th["binarize"]))
    if no_t:
        print(f"[score] no binarization threshold for {', '.join(no_t)}: BER and IoU binarize at 128", flush=True)
    card_f = a.out / ("scorecard.json" if a.gate == "paper" else f"scorecard_gate-{a.gate}.json")
    card = json.loads(card_f.read_text()) if card_f.exists() and not a.limit else {"sets": {}}
    for name in names:
        t0 = time.time()
        res = SC.score(a.out / "maps", [name], th["presence"], th["binarize"], a.workers, a.gate, limit=a.limit)
        card["sets"][name] = res[name]
        print(f"[score] {name}: {json.dumps(res[name])} ({time.time() - t0:.0f} s)", flush=True)
    card["sets"] = {k: card["sets"][k] for k in catalog.TEST_SETS if k in card["sets"]}
    card["tasks"], card["mean"] = SC.task_scores(card["sets"])
    card["thresholds"], card["gate"] = th, a.gate
    if not a.limit:
        card_f.write_text(json.dumps(card, indent=1) + "\n")
    print(SC.markdown(card["sets"], card["tasks"], card["mean"]))
    return card


def calibrate(a) -> dict:
    """Binarization thresholds (shadow, transparent object, mirror) from stored maps of the training images, by the
    rule of ``evaluation.binarize``; merged into ``<out>/thresholds.json``. Needs no model and no GPU."""
    from unveil.data import catalog
    from unveil.evaluation import binarize as Bz

    recs = catalog.calibration_records()
    have = {s: r[:a.limit] for s, r in recs.items() if (a.out / "maps" / s).is_dir()}
    if not have:
        raise SystemExit(f"no maps of training images in {a.out / 'maps'}: expected the folders {', '.join(recs)}")
    res = Bz.calibrate_maps(a.out / "maps", have, a.workers)
    fitted = {c: res[c] for c in Bz.CONCEPT_SETS if c in res}
    print(json.dumps({"binarize": fitted, "detail": res["detail"],
                      "not fitted": [c for c in Bz.CONCEPT_SETS if c not in res]}, indent=1))
    if not a.limit:
        th_f = a.out / "thresholds.json"
        th = json.loads(th_f.read_text()) if th_f.exists() else {}
        th["binarize"] = {**th.get("binarize", {}), **fitted}
        th_f.write_text(json.dumps(th, indent=1) + "\n")
    return fitted


def decisions(a) -> dict:
    """Presence decisions (paper, Tab. 6) on one GPU; needs the single-query presence of the 15 benchmark sets."""
    import unveil
    from unveil import prompts as P
    from unveil.data import catalog
    from unveil.evaluation import inference as I
    from unveil.evaluation import presence as PD
    from unveil.evaluation.scorecard import TASKS

    check_run(a, create=False)
    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    sets_of = {t: ss for t, _, ss, _ in TASKS}
    prompt_of = {s: catalog.TEST_SETS[s]["prompt"] for ss in sets_of.values() for s in ss}
    single = {s: I.read_presence(a.out / "maps" / s) for s in prompt_of}
    missing = [s for s, v in single.items() if len(v) != catalog.TEST_SETS[s]["n"]]
    if missing:
        raise SystemExit(f"run the inference of {missing} first (scripts/evaluate.py --stage infer)")
    joint_f = a.out / "joint_presence.json"
    joint = json.loads(joint_f.read_text()) if joint_f.exists() else {}
    for s in (s for t in PD.JOINT_TASKS for s in sets_of[t] if s not in joint):
        own = P.JOINT_PAIR.index(prompt_of[s])
        recs = [{"id": r["id"], "image": r["image"], "prompts": [own]} for r in catalog.test_records(s)]
        joint[s] = {x["id"]: x["logit"] for x in I.presence_logits(model, recs, True, a.batch, a.workers)}
        joint_f.write_text(json.dumps(joint))
        print(f"[decisions] joint presence {s}: {len(recs)} images", flush=True)
    tasks = [t for t, *_ in TASKS]
    present_single = PD.answered(single, tasks, sets_of, prompt_of, model.threshold)
    mixed = {s: (joint[s] if s in joint else single[s]) for s in single}
    present_joint = PD.answered(mixed, tasks, sets_of, prompt_of,
                                lambda p: model.threshold(p, joint=p in P.JOINT_PAIR))
    recs = catalog.rejection_records()
    out = {"single": {"present": present_single, "absent": PD.rejection(model, recs, joint=False)},
           "joint": {"present": present_joint, "absent": PD.rejection(model, recs, joint=True)}}
    (a.out / "presence_decisions.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| Query | Answers when present | Says none when absent | COD10K bg. | USC12K | LVIS |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, v in out.items():
        ab = v["absent"]
        print(f"| {k} | {v['present']['mean']:.1f} | {ab['mean']:.1f} | "
              + " | ".join(f"{ab[s]['none']:.1f}" for s in PD.SOURCES) + " |")
    return out


def retention(a) -> dict:
    """LVIS categories left out of training (paper: 0.841, frozen SAM 3 0.831), UNVEIL and SAM 3's own path."""
    import unveil
    from unveil.data import catalog
    from unveil.evaluation.retention import retention as run

    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    recs = catalog.retention_records()
    names = sorted({r["np"] for split in ("heldout", "seen") for r in catalog.retention_records(split=split)})
    out = {p: run(model, recs, p, a.limit, names, a.shards, a.batch) for p in ("unveil", "sam3")}
    if not a.limit:
        (a.out / "retention.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    from unveil.checkpoint import DEFAULT

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=None, help=f"default: {DEFAULT.name} in checkpoints/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="test sets (default: the 15 scorecard sets)")
    ap.add_argument("--extra", action="store_true", help="also the six extra test sets of the paper's Tab. 15")
    ap.add_argument("--stage", choices=("all", "infer", "score", "calibrate", "decisions", "retention"),
                    default="all",
                    help="calibrate: binarization thresholds from stored maps of the training images (CPU); "
                         "decisions: the presence decisions of Tab. 6 (one GPU, after the inference); retention: "
                         "LVIS categories left out of training (one GPU)")
    ap.add_argument("--decisions", action="store_true", help="with --stage all: also the presence decisions")
    ap.add_argument("--gate", choices=("paper", "all", "none"), default="paper",
                    help="paper: salient / camouflaged sets ungated, the others gated by presence; none: every "
                         "set ungated, no presence files needed")
    ap.add_argument("--data-root", default=None, help="prepared data (default: $UNVEIL_DATA or datasets/data)")
    ap.add_argument("--sam3-checkpoint", default=None,
                    help="local sam3.pt (default: $UNVEIL_SAM3, else checkpoints/sam3.pt, else Hugging Face)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shards", type=int, default=8,
                    help="image k goes to shard k %% shards, batched in order; 8 = the paper's 8-GPU batches on any "
                         "number of GPUs (bit-identical maps)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N images per set (smoke test; not comparable)")
    a = ap.parse_args(argv)
    a.checkpoint_given, a.checkpoint = a.checkpoint is not None, a.checkpoint or DEFAULT

    if a.data_root:
        os.environ["UNVEIL_DATA"] = str(a.data_root)
    from unveil.data import catalog

    names = a.sets or [k for k, v in catalog.TEST_SETS.items() if not v.get("extra")]
    if a.extra and not a.sets:
        names += [k for k, v in catalog.TEST_SETS.items() if v.get("extra")]
    unknown = set(names) - set(catalog.TEST_SETS)
    if unknown:
        raise SystemExit(f"unknown sets {sorted(unknown)}; known: {', '.join(catalog.TEST_SETS)}")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "infer"):
        infer(a, names)
    if a.stage == "calibrate" and int(os.environ.get("RANK", 0)) == 0:
        calibrate(a)
    if a.stage in ("all", "score") and int(os.environ.get("RANK", 0)) == 0:
        score(a, names)
    if (a.stage == "decisions" or a.stage == "all" and a.decisions) and int(os.environ.get("RANK", 0)) == 0:
        decisions(a)
    if a.stage == "retention" and int(os.environ.get("RANK", 0)) == 0:
        retention(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

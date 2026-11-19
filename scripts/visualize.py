"""Figures: prompts on one image, test examples next to their ground truth, and SAM 3's presence veto.

    python scripts/visualize.py panel photo.jpg --prompts shadow mirror edges --out outputs/panel.png
    python scripts/visualize.py examples --eval-dir outputs/eval --sets sbu cod10k bsds500 --n 3 --out outputs/ex.png
    python scripts/visualize.py veto photo.jpg --prompts "camouflaged object" --out outputs/veto.png

``panel`` runs the model on one image (``--joint`` adds the joint query). ``examples`` draws random test images of the
given sets from the maps ``scripts/evaluate.py`` stored (gated as in the scorecard), with the ground truth.
``veto`` compares, per prompt, SAM 3's instance map, the same map with the presence score set to one, and UNVEIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def panel(a) -> Path:
    from PIL import Image

    import unveil
    from unveil.viz import figures as F
    from unveil.viz import render as R

    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    image = Image.open(a.image).convert("RGB")
    maps, pres = model.predict(image, a.prompts, gate=False)
    res = [(p, m, q, q >= model.threshold(p)) for p, m, q in zip(a.prompts, maps, pres)]
    out = F.prompts_panel(image, res, a.out)
    if a.joint:
        from unveil.prompts import JOINT_PAIR

        jm, _ = model.predict(image, list(JOINT_PAIR), joint=True)
        R.joint_overlay(image, jm, list(JOINT_PAIR)).save(a.out.with_name(a.out.stem + "_joint.jpg"), quality=92)
    return out


def examples(a) -> Path:
    import numpy as np
    from PIL import Image

    from unveil.data import catalog
    from unveil.evaluation.inference import map_name, read_presence
    from unveil.evaluation.scorecard import sigmoid
    from unveil.viz import figures as F

    th = json.loads((a.eval_dir / "thresholds.json").read_text())["presence"]
    rng = np.random.default_rng(a.seed)
    rows = []
    for name in a.sets:
        meta, recs = catalog.TEST_SETS[name], catalog.test_records(name)
        d = a.eval_dir / "maps" / name
        pres = read_presence(d)
        tau = th.get(meta["prompt"], th["free_text"])
        for k in sorted(rng.choice(len(recs), size=min(a.n, len(recs)), replace=False)):
            r = recs[k]
            pred = np.asarray(Image.open(d / map_name(r["id"])).convert("L"), np.float32) / 255
            if meta["gated"] and sigmoid(pres[r["id"]]) < tau:
                pred = np.zeros_like(pred)
            if meta["family"] == "edge":
                gt = np.mean(catalog.edge_gts(r), axis=0).astype(np.float32)
            else:
                gt = catalog.load_gt(r).astype(np.float32) / 255
            img = Image.open(r["image"]).convert("RGB").resize(pred.shape[::-1], Image.BILINEAR)
            rows.append({"title": f"{meta['name']}\n{meta['prompt']}", "prompt": meta["prompt"],
                         "image": np.asarray(img), "pred": pred, "gt": gt})
    return F.examples_grid(rows, a.out)


def veto(a) -> Path:
    import numpy as np
    from PIL import Image

    import unveil
    from unveil.model.veto import MAP_FLOOR, instance_map
    from unveil.viz import figures as F
    from unveil.viz import render as R

    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    image = Image.open(a.image).convert("RGB")
    cells, titles = [], []
    for prompt in a.prompts:
        st = model.instances(image, prompt, presence_one=False, confidence=MAP_FLOOR)
        native, p = instance_map(st), st["presence"]
        lifted = instance_map(model.instances(image, prompt, presence_one=True, confidence=MAP_FLOOR))
        ours, q = model.predict(image, prompt, gate=False)
        row = [np.asarray(image)] + [np.asarray(R.overlay(image, m, prompt)) for m in (native, lifted, ours)]
        cells.append(row)
        titles.append(f"{prompt}\nSAM 3 presence {p:.2f}\nUNVEIL presence {q:.2f}")
    heads = ["image", "SAM 3 instance map", "presence set to one", "UNVEIL"]
    return F.grid(cells, heads, titles, a.out)


def main(argv=None) -> int:
    from unveil.checkpoint import DEFAULT

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("panel", "veto"):
        p = sub.add_parser(name)
        p.add_argument("image", type=Path)
        p.add_argument("--prompts", nargs="+", required=True)
        p.add_argument("--checkpoint", type=Path, default=DEFAULT)
        p.add_argument("--sam3-checkpoint", default=None,
                       help="local sam3.pt (default: $UNVEIL_SAM3, else checkpoints/sam3.pt, else Hugging Face)")
        p.add_argument("--out", type=Path, required=True)
        if name == "panel":
            p.add_argument("--joint", action="store_true", help="also draw the joint query (salient, camouflaged)")
    e = sub.add_parser("examples")
    e.add_argument("--eval-dir", type=Path, required=True, help="the --out directory of scripts/evaluate.py")
    e.add_argument("--sets", nargs="+", required=True)
    e.add_argument("--n", type=int, default=3, help="images per set")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out = {"panel": panel, "examples": examples, "veto": veto}[a.cmd](a)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

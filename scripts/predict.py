"""Run UNVEIL on your own images: one map per (image, prompt), or a joint query.

    python scripts/predict.py photo.jpg --prompts shadow "transparent object" edges --out outputs/predict
    python scripts/predict.py photos/ --prompts "salient object" "camouflaged object" --joint --out outputs/joint
    python scripts/predict.py photo.jpg --prompts dog --path sam3 --out outputs/sam3     # SAM 3's own semantic path

For every image and prompt it writes ``<stem>__<prompt>.png`` (8-bit map, all zero when the presence decoder says
"none"; ``--no-gate`` keeps the map) and, with ``--overlay``, the map drawn on the image. ``predictions.json`` lists
the presence probabilities and decisions. Prompts outside training are answered as free text (README, "Usage").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in text.strip())


def images_of(inputs: list[Path]) -> list[Path]:
    out = []
    for p in inputs:
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    return out


def main(argv=None) -> int:
    import numpy as np
    from PIL import Image

    import unveil
    from unveil.checkpoint import DEFAULT
    from unveil.viz import render as R

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", type=Path, nargs="+", help="image files or folders")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--joint", action="store_true", help='joint query {"salient object", "camouflaged object"}')
    ap.add_argument("--path", choices=("unveil", "sam3"), default="unveil")
    ap.add_argument("--no-gate", action="store_true", help="keep maps the presence decoder rejects")
    ap.add_argument("--overlay", action="store_true", help="also write the maps drawn on the images")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT)
    ap.add_argument("--sam3-checkpoint", default=None,
                    help="local sam3.pt (default: $UNVEIL_SAM3, else checkpoints/sam3.pt, else Hugging Face)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    model = unveil.load(a.checkpoint, sam3_checkpoint=a.sam3_checkpoint)
    a.out.mkdir(parents=True, exist_ok=True)
    log, seen = [], {}
    for img_path in images_of(a.inputs):
        image = Image.open(img_path).convert("RGB")
        base = safe(img_path.stem)
        seen[base] = seen.get(base, 0) + 1
        base = base if seen[base] == 1 else f"{base}_{seen[base]}"          # images with the same name
        maps, pres = model.predict(image, a.prompts, joint=a.joint, gate=not a.no_gate, path=a.path)
        for prompt, m, p in zip(a.prompts, maps, pres):
            stem = f"{base}__{safe(prompt)}"
            R.mask_image(m).save(a.out / f"{stem}.png")
            if a.overlay:
                R.overlay(image, m, prompt).save(a.out / f"{stem}__overlay.jpg", quality=92)
            ours = a.path == "unveil"                     # SAM 3's semantic path has no presence decision
            log.append({"image": str(img_path), "output": f"{stem}.png", "prompt": prompt, "presence": round(p, 4),
                        "threshold": model.threshold(prompt, a.joint) if ours else None,
                        "accepted": p >= model.threshold(prompt, a.joint) if ours else None,
                        "mean": round(float(np.mean(m)), 4)})
            print(json.dumps(log[-1]), flush=True)
        if a.overlay and a.joint:
            R.joint_overlay(image, maps, a.prompts).save(a.out / f"{base}__joint.jpg", quality=92)
    (a.out / "predictions.json").write_text(json.dumps(log, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train UNVEIL on the prepared data and write a checkpoint in the release format.

    torchrun --nproc_per_node 8 scripts/train.py --out outputs/train
    torchrun --nproc_per_node 4 scripts/train.py --out outputs/train --accumulate 2      # same batch on 4 GPUs
    python scripts/train.py --out outputs/train --calibrate-only        # thresholds + checkpoint from final.pt

Step 1 trains on every visible GPU (``final.pt``; a rolling ``resume.pt`` lets a relaunch continue). Step 2 runs on
one GPU: presence thresholds on the validation pairs of the training splits, then ``<out>/unveil.safetensors``, which
``scripts/evaluate.py`` and ``scripts/predict.py`` take as ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "unveil.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", default=None,
                    help="prepared data (default: the config's data_root, else $UNVEIL_DATA, else datasets/data)")
    ap.add_argument("--sam3-checkpoint", default=None,
                    help="local sam3.pt (default: $UNVEIL_SAM3, else checkpoints/sam3.pt, else Hugging Face)")
    ap.add_argument("--iterations", type=int, default=None, help="override the config (smoke tests)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--accumulate", type=int, default=None,
                    help="ranks per GPU (gradient accumulation); GPUs x accumulate = 8 gives the paper's batch")
    ap.add_argument("--max-steps", type=int, default=None, help="stop this launch after N steps (smoke tests)")
    ap.add_argument("--calibrate-only", action="store_true", help="skip training; fit thresholds from final.pt")
    a = ap.parse_args(argv)

    from unveil.train.trainer import TrainConfig

    cfg = TrainConfig.from_yaml(a.config, iterations=a.iterations, workers=a.workers, accumulate=a.accumulate,
                                sam3_checkpoint=a.sam3_checkpoint, data_root=a.data_root)
    if cfg.data_root:
        os.environ["UNVEIL_DATA"] = str(cfg.data_root)
    if not a.calibrate_only:
        from unveil.train.trainer import train

        result = train(cfg, a.out, max_steps=a.max_steps)
        if not result:                     # other ranks, or a launch stopped by --max-steps
            return 0
    if int(os.environ.get("RANK", 0)) != 0:
        return 0
    from unveil.train.calibrate import calibrate

    report = calibrate(a.out / "final.pt", a.out / "unveil.safetensors", sam3_checkpoint=cfg.sam3_checkpoint,
                       workers=min(8, cfg.workers))
    print(json.dumps({k: report[k] for k in ("checkpoint", "sha256", "thresholds", "binarize")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

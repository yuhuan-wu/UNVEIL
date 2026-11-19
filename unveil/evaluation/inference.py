"""Batched inference over test records: maps and presence logits.

The protocol of the paper's runs: PIL bilinear stretch to 1008 x 1008, mean = std = 0.5, bf16 autocast, TF32 off,
batches of eight images. A map is sigmoid(logit) with the 288 x 288 logit resized bilinearly to the GT size (BSDS500:
the image size), saved as an 8-bit PNG, round(255 p). Under ``torchrun`` every GPU takes the records k with
k % world == rank. The paper ran 8 GPUs; other GPU counts form other batches, which moves bf16 results in the
last bits.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unveil.model.sam3 import to_input


def dist_info() -> tuple[int, int]:
    import torch.distributed as dist

    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, world


def map_name(rec_id: str) -> str:
    return rec_id.replace("/", "__") + ".png"


class _Images(torch.utils.data.Dataset):
    def __init__(self, recs: list[dict], hw_of=None):
        self.recs, self.hw_of = recs, hw_of

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["image"]).convert("RGB")          # no EXIF transpose, as SAM 3's loader
        hw = self.hw_of(r) if self.hw_of else (img.height, img.width)
        return {"i": i, "input": to_input(img), "hw": hw}


def _batches(recs: list[dict], hw_of, batch: int, workers: int):
    return torch.utils.data.DataLoader(_Images(recs, hw_of), batch_size=batch, shuffle=False, num_workers=workers,
                                       pin_memory=True, collate_fn=lambda xs: {
                                           "i": [x["i"] for x in xs], "input": torch.stack([x["input"] for x in xs]),
                                           "hw": [x["hw"] for x in xs]})


def shard(recs: list, rank: int, world: int) -> list:
    return [r for k, r in enumerate(recs) if k % world == rank]


@torch.no_grad()
def write_maps(model, recs: list[dict], prompt: str, out_dir, hw_of, rank: int = 0, world: int = 1, batch: int = 8,
               workers: int = 8, path: str = "unveil") -> dict:
    """Single queries with ``prompt`` on this GPU's share of ``recs``: ungated maps as PNG files in ``out_dir`` and
    presence logits in ``out_dir/presence_r<rank>.json`` ({id: logit}). Skips a share that is already complete."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pres_f = out / f"presence_r{rank}.json"
    mine = shard(recs, rank, world)
    if pres_f.exists() and len(json.loads(pres_f.read_text())) == len(mine):
        return {"images": len(mine), "skipped": True}
    pres, futs = {}, []
    with ThreadPoolExecutor(4) as pool:
        for b in _batches(mine, hw_of, batch, workers):
            n = len(b["i"])
            coarse, logit = model.single(b["input"], torch.arange(n), [prompt] * n, path)
            for k in range(n):
                rec = mine[b["i"][k]]
                m = torch.nn.functional.interpolate(coarse[k:k + 1].float(), size=b["hw"][k], mode="bilinear",
                                                    align_corners=False)[0, 0]
                arr = (m.sigmoid() * 255).round().to(torch.uint8).cpu().numpy()
                futs.append(pool.submit(lambda a, p: Image.fromarray(a).save(p), arr, out / map_name(rec["id"])))
                pres[rec["id"]] = round(float(logit[k]), 5)
        for f in futs:
            f.result()
    pres_f.write_text(json.dumps(pres))
    return {"images": len(mine), "skipped": False}


def read_presence(out_dir) -> dict[str, float]:
    """Presence logits of every GPU's share, merged."""
    out = {}
    for f in sorted(Path(out_dir).glob("presence_r*.json")):
        out.update(json.loads(f.read_text()))
    return out


@torch.no_grad()
def presence_logits(model, recs: list[dict], joint: bool = False, batch: int = 8, workers: int = 8,
                    shards: int = 1) -> list[dict]:
    """Presence logits of validation records {id, image, prompts: [prompt ids]} -> [{id, prompt, logit}].
    joint=True asks the joint query for each pair (prompt ids 0 / 1); otherwise single queries. shards: batches are
    formed within the shards k % shards (8 = the batches of the paper's 8-GPU runs)."""
    from unveil import prompts as P

    out = []
    for s in range(shards):
        out += _presence(model, shard(recs, s, shards), joint, batch, workers, P)
    return out


def _presence(model, recs: list[dict], joint: bool, batch: int, workers: int, P) -> list[dict]:
    out = []
    for b in _batches(recs, None, batch, workers):
        img_ids = [k for k in range(len(b["i"])) for _ in recs[b["i"][k]]["prompts"]]
        ids = [p for k in range(len(b["i"])) for p in recs[b["i"][k]]["prompts"]]
        if joint:
            _, logit = model.joint(b["input"], torch.tensor(img_ids), torch.tensor(ids))
        else:
            feats = model.features(b["input"], torch.tensor(img_ids), [P.TASK_PROMPTS[p] for p in ids])
            with model._autocast():
                prompt, pmask = model.heads.prompt_of(feats)
                enc = model.heads.encode(feats["src"], feats["pos"], feats["size"], prompt, pmask)
                logit = model.heads.presence(enc, prompt, pmask)
        for j, (k, p) in enumerate(zip(img_ids, ids)):
            out.append({"id": recs[b["i"][k]]["id"], "prompt": p, "logit": round(float(logit[j]), 5)})
    return out


def to_u8(prob: np.ndarray) -> np.ndarray:
    return np.round(np.clip(prob, 0, 1) * np.float32(255)).astype(np.uint8)


@torch.no_grad()
def histograms(model, recs: list[dict], prompt: str, tau: float, batch: int = 8, workers: int = 8
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-image 256-bin histograms of the gated 8-bit map over GT foreground / background (GT read as 8-bit gray,
    foreground >= 128) for records {id, image, gt}; maps at the GT size, as stored by ``write_maps``."""
    from unveil.evaluation.metrics import region_record

    def gt_size(r):
        with Image.open(r["gt"]) as g:
            return g.height, g.width

    fg, bg = [], []
    for b in _batches(recs, gt_size, batch, workers):
        n = len(b["i"])
        coarse, logit = model.single(b["input"], torch.arange(n), [prompt] * n)
        for k in range(n):
            m = torch.nn.functional.interpolate(coarse[k:k + 1].float(), size=b["hw"][k], mode="bilinear",
                                                align_corners=False)[0, 0]
            pred = (m.sigmoid() * 255).round().to(torch.uint8).cpu().numpy()
            if 1.0 / (1.0 + np.exp(-round(float(logit[k]), 5))) < tau:
                pred = np.zeros_like(pred)
            gt = np.asarray(Image.open(recs[b["i"][k]]["gt"]).convert("L"), dtype=np.uint8)
            r = region_record(pred, gt)
            fg.append(r["fg"])
            bg.append(r["bg"])
    return np.stack(fg).astype(np.int64), np.stack(bg).astype(np.int64)

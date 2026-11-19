"""The training run: data-parallel over the visible GPUs, under ``torchrun``.

Each step, on every GPU: the frozen ViT prefix runs without gradients; the network (LoRA branch, heads, convention
vectors) maps every pair of the batch; the loss is the dense group mean plus the presence term; then the joint
presence module takes its own step (``joint.py``); then the network's backward pass, a global gradient-norm clip and an
AdamW update. bf16 autocast with fp32 losses, TF32 on. The learning rate warms up linearly for ``warmup`` steps and
then follows a cosine to zero. Outputs in ``out_dir``: ``config.yaml``, ``log.jsonl``, a rolling ``resume.pt`` and, at
the end, ``final.pt`` (the trained tensors only).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist

from unveil.data import streams as D
from unveil.model.backbone import vit_prefix
from unveil.model.sam3 import build_sam3, normalize, set_tf32
from unveil.train.joint import JointPresenceTrainer
from unveil.train.losses import total_loss
from unveil.train.network import LoraConfig, TrainNet, param_groups


@dataclass
class TrainConfig:
    iterations: int = 11_000
    seed: int = 0
    lr: float = 8e-5                     # heads
    weight_decay: float = 0.1
    lr_lora: float = 2.5e-4
    weight_decay_lora: float = 0.1
    lr_convention: float = 8e-4
    clip: float = 0.1                    # global gradient-norm clip
    accumulate: int = 1                  # ranks per GPU: 8 GPUs x 1 = 4 x 2 = 1 x 8 give the paper's batch
    warmup: int = 300
    presence_weight: float = 1.0
    lora: LoraConfig = field(default_factory=LoraConfig)
    per_gpu: dict = field(default_factory=dict)       # stream name -> images per GPU per step (default: streams.py)
    joint_lr: float = 8e-5               # the joint presence module
    joint_weight_decay: float = 0.1
    workers: int = 10
    log_every: int = 20
    save_every: int = 1000
    sam3_checkpoint: str | None = None
    data_root: str | None = None

    @classmethod
    def from_yaml(cls, path, **overrides) -> "TrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        raw["lora"] = LoraConfig(**raw.get("lora", {}))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def streams(self) -> tuple[D.Stream, ...]:
        return tuple(dataclasses.replace(s, per_gpu=int(self.per_gpu.get(s.name, s.per_gpu))) for s in D.TRUNK_STREAMS)

    def lr_factor(self, it: int) -> float:
        if it < self.warmup:
            return (it + 1) / self.warmup
        t = (it - self.warmup) / max(1, self.iterations - self.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


# --------------------------------------------------------------------------------------------- setup

def init_dist() -> tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def _worker_init(_):
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def encode_prompts(sam3, prompts: dict, device: str) -> dict:
    """Text features of every training prompt in id order: the task prompts and the replay categories in one call,
    the negative-only LVIS categories in a second one."""
    from unveil.model.backbone import encode_text

    text = encode_text(sam3, list(prompts["task"]) + list(prompts["replay"]), device)
    if prompts.get("negative_only"):
        extra = encode_text(sam3, list(prompts["negative_only"]), device)
        text = {k: torch.cat([text[k], extra[k]], 0 if k == "language_mask" else 1)
                for k in ("language_features", "language_mask", "language_embeds")}
    return text


def _log(rank: int, f, rec: dict) -> None:
    if rank == 0:
        print(json.dumps(rec), flush=True)
        f.write(json.dumps(rec) + "\n")
        f.flush()


def _save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------- run

def train(cfg: TrainConfig, out_dir, index: dict | None = None, prompts: dict | None = None,
          max_steps: int | None = None) -> dict:
    """index / prompts: the training index and prompt lists (default: ``catalog.train_index`` / ``prompt_list``).
    max_steps: stop after this many steps of this launch (smoke tests; no final.pt unless the run completes)."""
    rank, world, local = init_dist()
    device = f"cuda:{local}"
    out = Path(out_dir)
    k = cfg.accumulate
    ranks, vworld = [rank * k + j for j in range(k)], world * k          # the (virtual) ranks this GPU runs
    torch.manual_seed(cfg.seed + ranks[0])
    if rank == 0:
        import yaml

        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    if world > 1:
        dist.barrier()
    resume = out / "resume.pt"
    st = torch.load(resume, map_location="cpu") if resume.exists() else None
    start = st["iter"] if st else 0

    if index is None or prompts is None:
        from unveil.data import catalog

        index = index if index is not None else catalog.train_index(cfg.data_root)
        prompts = prompts if prompts is not None else catalog.prompt_list()
    sam3 = build_sam3(device, cfg.sam3_checkpoint)
    set_tf32(True)
    text = encode_prompts(sam3, prompts, device)
    net = TrainNet(sam3, text, cfg.lora, device)
    joint = JointPresenceTrainer(net, index["joint"], cfg.iterations, cfg.lr_factor, device, ranks, vworld, world,
                                 cfg.joint_lr, cfg.joint_weight_decay, cfg.clip, cfg.presence_weight, cfg.seed, start,
                                 workers=min(2, cfg.workers))
    net.train()
    params = net.trainable()
    opt = torch.optim.AdamW(param_groups(net, cfg.lr, cfg.weight_decay, cfg.lr_lora, cfg.weight_decay_lora,
                                         cfg.lr_convention), betas=(0.9, 0.999))
    if st is not None:
        net.load(st["net"])
        opt.load_state_dict(st["opt"])
        joint.load_resume(st["joint"])
    ddp = (torch.nn.parallel.DistributedDataParallel(net, device_ids=[local], find_unused_parameters=False,
                                                     gradient_as_bucket_view=True, broadcast_buffers=False)
           if world > 1 else net)

    streams = cfg.streams()
    sizes = {s.name: len(index[s.name]) for s in streams}
    sampler = D.MicroBatchSampler([D.StreamSampler(streams, sizes, cfg.iterations, r, vworld, cfg.seed, start)
                                   for r in ranks])
    rng = [None] * k                     # each virtual rank's CUDA generator (dropout), seeded as its own GPU would be
    if k > 1:
        for j, r in enumerate(ranks):
            torch.cuda.manual_seed(cfg.seed + r)
            rng[j] = torch.cuda.get_rng_state()
    loader = torch.utils.data.DataLoader(D.TrunkDataset(index, streams, len(prompts["task"])), batch_sampler=sampler,
                                         num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                                         prefetch_factor=4 if cfg.workers else None, collate_fn=D.collate,
                                         worker_init_fn=_worker_init)
    flog = open(out / "log.jsonl", "a") if rank == 0 else None
    n_params = sum(p.numel() for p in params)
    _log(rank, flog, {"start_iter": start, "world": world, "accumulate": k, "trainable_params": n_params,
                      "stream_sizes": {s.name: len(index[s.name]) for s in streams},
                      "per_gpu": {s.name: s.per_gpu for s in streams}})
    keys = [s.name for s in streams if s.per_gpu and s.kind in ("region", "edge", "skeleton", "replay")]
    keys += ["presence", "presence_joint"]                                     # logged loss terms
    zero = {k: torch.zeros((), device=device) for k in keys}
    acc, cnt = dict(zero), {k: 0 for k in keys}
    t_win, t_data, n_win, t_run = time.time(), 0.0, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    batches = iter(loader)
    for it in range(start, cfg.iterations):
        f = cfg.lr_factor(it)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        opt.zero_grad(set_to_none=True)
        for j in range(k):                                   # one pass per (virtual) rank; gradients accumulate
            t0 = time.time()
            batch = next(batches)
            t_data += time.time() - t0
            if k > 1:
                torch.cuda.set_rng_state(rng[j])
            images = normalize(batch["input"].to(device, non_blocking=True))
            labels = [None if lb is None else lb.to(device, non_blocking=True) for lb in batch["labels"]]
            sizes = [None if lb is None else tuple(lb.shape) for lb in labels]
            sync = ddp.no_sync() if world > 1 and j < k - 1 else contextlib.nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix = vit_prefix(sam3, images, cfg.lora.first_block)
                    logits, pres = ddp(prefix, batch["img_ids"], batch["prompts"], batch["convention"], sizes)
                loss, parts = total_loss(logits, labels, batch["kinds"], batch["groups"], pres, batch["presence"],
                                         cfg.presence_weight)
                parts["presence_joint"] = joint.micro(j)
                if j == k - 1:
                    joint.update()
                (loss / k if k > 1 else loss).backward()
            if k > 1:
                rng[j] = torch.cuda.get_rng_state()
            for key, v in parts.items():
                acc[key] = acc[key] + v
                cnt[key] += 1
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        n_win += 1
        if (it + 1) % cfg.log_every == 0 or it + 1 == cfg.iterations:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            vals = torch.stack([acc[k] / max(cnt[k], 1) for k in keys])
            if world > 1:
                dist.all_reduce(vals)
                vals /= world
            _log(rank, flog, {"iter": it + 1, "lr": opt.param_groups[0]["lr"],
                              "loss": {k: round(float(v), 4) for k, v in zip(keys, vals)},
                              "grad_norm": round(float(gnorm), 4), "ms_step": round(1000 * dt / n_win, 1),
                              "data_wait_ms": round(1000 * t_data / n_win, 1),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                              "elapsed_min": round((time.time() - t_run) / 60, 1)})
            acc, cnt = dict(zero), {k: 0 for k in keys}
            t_win, t_data, n_win = time.time(), 0.0, 0
        if max_steps is not None and it + 1 - start >= max_steps:
            break
        if rank == 0 and (it + 1) % cfg.save_every == 0 and it + 1 < cfg.iterations:
            _save({"net": net.state(), "opt": opt.state_dict(), "iter": it + 1,
                   "joint": joint.resume_state()}, resume)

    result = {}
    if rank == 0 and it + 1 == cfg.iterations:
        final = {"net": net.state(), "joint_decoder": joint.state(), "config": dataclasses.asdict(cfg),
                 "iterations": cfg.iterations, "world": world}
        _save(final, out / "final.pt")
        resume.unlink(missing_ok=True)
        result = {"final": str(out / "final.pt"), "trainable_params": n_params,
                  "hours_this_launch": round((time.time() - t_run) / 3600, 3), "world": world}
        _log(rank, flog, {"done": result})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result

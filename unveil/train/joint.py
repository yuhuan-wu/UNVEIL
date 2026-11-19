"""The joint presence module and its training step.

A joint query {salient object, camouflaged object} runs both members with their convention vectors and asks a separate
presence decoder whether each member occurs. That decoder is a copy of SAM 3's presence decoder made before training.
It trains in the same run as the rest of the model, isolated from it:
- its inputs are the current network's features, computed without gradients and with the network in eval mode;
- it has its own data stream (``streams.JOINT_STREAM``: USC12K scenes with both, one or neither concept, and COD10K
  background images), its own optimizer, gradient clipping and gradient all-reduce;
- its random numbers come from forked generators seeded per step, so the main network's random streams (dropout)
  are the ones it would draw without this module.
Its loss is the class-balanced presence BCE on both members of every image.
"""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist

from unveil.data import streams as D
from unveil.model.heads import make_trainable_decoder, presence_logits
from unveil.model.sam3 import normalize
from unveil.model.backbone import vit_prefix
from unveil.train.losses import presence_loss
from unveil.train.network import TrainNet, decoder_param_groups

INIT_SEED, STEP_SEED = 7_006, 7_007


def _seed(s: int, dev: torch.device) -> None:
    torch.random.default_generator.manual_seed(s)
    with torch.cuda.device(dev):
        torch.cuda.manual_seed(s)


class JointPresenceTrainer:
    """ranks: the (virtual) ranks this GPU runs, out of ``virtual_world`` (several with gradient accumulation)."""

    def __init__(self, net: TrainNet, items: list[dict], n_iters: int, lr_factor, device: str, ranks: list[int],
                 virtual_world: int, world: int, lr: float = 8e-5, weight_decay: float = 0.1, clip: float = 0.1,
                 lam: float = 1.0, seed: int = 0, start: int = 0, workers: int = 2):
        self.net, self.device, self.ranks, self.vworld, self.world = net, torch.device(device), ranks, virtual_world, world
        self.lr_factor, self.clip, self.lam = lr_factor, clip, lam
        with torch.random.fork_rng(devices=[self.device]):
            _seed(INIT_SEED, self.device)
            self.decoder = make_trainable_decoder(copy.deepcopy(net.heads.decoder)).to(device).train()
        self.params = [p for p in self.decoder.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(decoder_param_groups(self.decoder, lr, weight_decay), betas=(0.9, 0.999))
        self.it = start
        sampler = D.MicroBatchSampler([D.JointSampler(len(items), n_iters, r, virtual_world, seed, start) for r in ranks])
        self.loader = torch.utils.data.DataLoader(
            D.JointDataset(items), batch_sampler=sampler, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0, prefetch_factor=4 if workers else None, collate_fn=D.joint_collate,
            generator=torch.Generator().manual_seed(INIT_SEED + ranks[0]))
        self._batches = None

    # ------------------------------------------------------------------------------------------------ state

    def state(self) -> dict:
        keep = {k for k, p in self.decoder.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.decoder.state_dict().items() if k in keep}

    def resume_state(self) -> dict:
        return {"decoder": self.state(), "opt": self.opt.state_dict(), "iter": self.it}

    def load_resume(self, st: dict) -> None:
        self.decoder.load_state_dict(st["decoder"], strict=False)
        self.opt.load_state_dict(st["opt"])
        assert st["iter"] == self.it, (st["iter"], self.it)

    # ------------------------------------------------------------------------------------------------ step

    def _inputs(self, images: torch.Tensor):
        """Both members of every image, with the convention vectors, through the current network (no gradients)."""
        net = self.net
        was = net.training
        net.eval()
        try:
            b = images.shape[0]
            img = torch.arange(b).repeat_interleave(2)
            member = torch.tensor([0, 1] * b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = vit_prefix(net.sam3, images, net.first)
                # The branch runs with autograd on (its training kernels; the fused inference MLP needs merged
                # weights); the outputs are detached.
                with torch.enable_grad():
                    f = net.pair_inputs(prefix, img, member)
                f = {k: ([x.detach() for x in v] if isinstance(v, list) else v.detach() if torch.is_tensor(v) else v)
                     for k, v in f.items()}
                with torch.no_grad():
                    prompt, pmask = net.heads.prompt_of(f, net.convention.detach()[member.to(self.device)])
                    enc = net.heads.encode(f["src"], f["pos"], f["size"], prompt, pmask)
        finally:
            net.train(was)
        return enc, prompt, pmask

    def micro(self, j: int) -> torch.Tensor:
        """Forward and backward of this GPU's j-th (virtual) rank at the current iteration -> its loss (detached).
        Gradients accumulate over the ranks; ``update`` applies them."""
        if self._batches is None:
            self._batches = iter(self.loader)
        batch = next(self._batches)
        if j == 0:
            self.opt.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[self.device]):
            _seed(STEP_SEED + self.it * self.vworld + self.ranks[j], self.device)
            enc, prompt, pmask = self._inputs(normalize(batch["input"].to(self.device, non_blocking=True)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = presence_logits(self.decoder, enc, prompt, pmask)
            loss = self.lam * presence_loss(logit, batch["labels"].to(self.device))
            (loss / len(self.ranks) if len(self.ranks) > 1 else loss).backward()
        return loss.detach()

    def update(self) -> None:
        """Average the gradients over the GPUs, clip, step the optimizer at the iteration's learning rate."""
        f = self.lr_factor(self.it)
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * f
        self._allreduce()
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.opt.step()
        self.it += 1

    def _allreduce(self) -> None:
        if self.world == 1:
            return
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= self.world
        for p, g in zip(self.params, torch._utils._unflatten_dense_tensors(flat, grads)):
            p.grad = g

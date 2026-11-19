"""Training streams, datasets and the distributed batch sampler.

One training step gives every GPU a fixed number of images from each stream (``Stream.per_gpu``):

    stream            pairs per image                      loss
    sod               "salient object"                     region (BCE + IoU)
    cod               "camouflaged object"                 region
    edge              "edges"                              edge (RCF class-balanced BCE)
    shadow            "shadow"                             shadow (class-balanced BCE + IoU)
    transparent       "transparent object"                 region
    mirror            "mirror"                             region
    skeleton          "skeleton" (SYM-PASCAL: "symmetry axis")   skeleton (HED class-balanced BCE, dilated target)
    switch            both prompts, own masks, with the convention vectors (USC12K scenes with both concepts)
    neg_cam, neg_sal  "camouflaged object" / "salient object", presence 0 (verified negatives)
    replay            an LVIS category name of the image (concept replay), region loss + presence
    replay_neg        an LVIS category verified absent from the image, presence 0
    joint             the joint presence module's own stream (``JointDataset``)

Sampling. Each stream walks through its items in passes; pass q of stream s is the permutation
``default_rng([seed, s.perm_key, q])``. The k-th draw of iteration ``it`` on GPU ``rank`` of ``world`` has the global
position ``pos = (it * per_gpu + k) * world + rank`` and the sample seed
``s.seed_offset + seed * 1_000_003 + pos * s.seed_stride + s.seed_add``. The stream depends only on (seed, it), so a
resumed run draws the same data. The constants below reproduce the released model's run; other distinct values give
an equally valid sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from unveil.data import transforms as T

PROMPT_SYMMETRY, SYMMETRY_SRC = 7, "sympascal"


@dataclass(frozen=True)
class Stream:
    name: str                       # group of the training index
    kind: str                       # region | edge | skeleton | negative | replay | replay_negative | joint
    prompts: tuple = ()             # prompt ids of each image's pairs (dense and negative streams)
    per_gpu: int = 2                # images per GPU per step
    perm_key: int = 0
    seed_offset: int = 0
    seed_stride: int = 11
    seed_add: int = 0
    convention: bool = False        # pairs read the convention vectors (USC12K pairs)

    def sample_seed(self, seed: int, pos: int) -> int:
        return int(self.seed_offset + seed * 1_000_003 + pos * self.seed_stride + self.seed_add)


def _dense(name: str, kind: str, prompts: tuple, key: int, convention: bool = False) -> Stream:
    return Stream(name, kind, prompts, 2, key, 0, 11, key, convention)


TRUNK_STREAMS: tuple[Stream, ...] = (
    _dense("sod", "region", (0,), 0),
    _dense("cod", "region", (1,), 1),
    _dense("edge", "edge", (2,), 2),
    _dense("shadow", "region", (3,), 3),
    _dense("transparent", "region", (4,), 4),
    _dense("mirror", "region", (5,), 5),
    _dense("skeleton", "skeleton", (6,), 6),
    _dense("switch", "region", (0, 1), 7, convention=True),
    _dense("neg_cam", "negative", (1,), 8),
    _dense("neg_sal", "negative", (0,), 9),
    Stream("replay", "replay", (), 2, 11, 10**12, 1, 0),
    Stream("replay_neg", "replay_negative", (), 2, 14, 3 * 10**12, 3, 0),
)
JOINT_STREAM = Stream("joint", "joint", (0, 1), 2, 7_005, 8 * 10**12, 1, 0)


class StreamSampler:
    """Yields, for iterations [start, n_iters), this GPU's list of (stream index, item index, sample seed)."""

    def __init__(self, streams: tuple[Stream, ...], sizes: dict[str, int], n_iters: int, rank: int, world: int,
                 seed: int = 0, start: int = 0):
        self.streams, self.sizes = streams, sizes
        self.n_iters, self.rank, self.world, self.seed, self.start = n_iters, rank, world, seed, start
        self._perm: dict[tuple[int, int], np.ndarray] = {}
        empty = [s.name for s in streams if s.per_gpu and not sizes.get(s.name)]
        if empty:
            raise ValueError(f"empty training streams: {empty}")

    def __len__(self):
        return self.n_iters - self.start

    def _item(self, s: int, pos: int) -> int:
        st = self.streams[s]
        n = self.sizes[st.name]
        q, k = divmod(pos, n)
        if (s, q) not in self._perm:
            self._perm = {kk: v for kk, v in self._perm.items() if kk[0] != s or kk[1] >= q - 1}
            self._perm[(s, q)] = np.random.default_rng([self.seed, st.perm_key, q]).permutation(n)
        return int(self._perm[(s, q)][k])

    def batch(self, it: int) -> list[tuple[int, int, int]]:
        out = []
        for s, st in enumerate(self.streams):
            for k in range(st.per_gpu):
                pos = (it * st.per_gpu + k) * self.world + self.rank
                out.append((s, self._item(s, pos), st.sample_seed(self.seed, pos)))
        return out

    def __iter__(self):
        for it in range(self.start, self.n_iters):
            yield self.batch(it)


class MicroBatchSampler:
    """Several ranks' samplers on one GPU (gradient accumulation): for each iteration, rank 0's batch, then rank 1's.
    With one sampler it yields that sampler's batches."""

    def __init__(self, samplers: list):
        self.samplers = samplers

    def __len__(self):
        return len(self.samplers[0]) * len(self.samplers)

    def __iter__(self):
        first = self.samplers[0]
        for it in range(first.start, first.n_iters):
            for s in self.samplers:
                yield s.batch(it)


class TrunkDataset(torch.utils.data.Dataset):
    """Key (stream index, item index, sample seed) -> {"input": uint8 (3, 1008, 1008), "pairs": [...]}."""

    def __init__(self, index: dict[str, list[dict]], streams: tuple[Stream, ...] = TRUNK_STREAMS,
                 n_task_prompts: int = 8, augment: bool = True):
        self.index, self.streams, self.augment = index, streams, augment
        self.prompt_base = n_task_prompts          # replay category k has prompt id prompt_base + k

    def __len__(self):
        return sum(len(self.index[s.name]) for s in self.streams)

    def __getitem__(self, key: tuple[int, int, int]) -> dict:
        s, i, seed = key
        st = self.streams[s]
        item = self.index[st.name][i]
        if st.kind == "replay":
            return T.replay_sample(item, self.prompt_base, seed, self.augment)
        if st.kind == "replay_negative":
            return T.replay_negative_sample(item, seed, self.augment)
        prompts = st.prompts
        if st.name == "skeleton" and item.get("src") == SYMMETRY_SRC:
            prompts = (PROMPT_SYMMETRY,)
        return T.dense_sample(item, st.kind, prompts, st.name, seed, self.augment, st.convention)


def collate(samples: list[dict]) -> dict:
    pairs = [(k, p) for k, s in enumerate(samples) for p in s["pairs"]]
    return {"input": torch.stack([s["input"] for s in samples]),
            "img_ids": torch.tensor([k for k, _ in pairs], dtype=torch.long),
            "prompts": torch.tensor([p["prompt"] for _, p in pairs], dtype=torch.long),
            "convention": torch.tensor([p["convention"] for _, p in pairs], dtype=torch.bool),
            "labels": [p["label"] for _, p in pairs], "kinds": [p["kind"] for _, p in pairs],
            "groups": [p["group"] for _, p in pairs],
            "presence": torch.tensor([p["presence"] for _, p in pairs], dtype=torch.float32)}


# --------------------------------------------------------------------------------------------- joint presence stream

# kind of a joint item -> (per-sample code, prompts of its sample)
JOINT_KINDS = {"both": ("region", (0, 1)), "none": ("negative", (0,)), "bg": ("negative", (1,))}
NO_TARGET = -2.0


def joint_targets(kind: str, pairs: list[dict]) -> list[float]:
    """Presence targets of the two members (salient, camouflaged) of a joint query on this sample.
    both: each member's own mask decides (1, or -1 if the crop removed it); single: the own member as its mask,
    the other 0; none: both 0; bg (COD10K background): camouflaged 0, salient no target."""
    row = [NO_TARGET, NO_TARGET]
    if kind == "both":
        for p in pairs:
            row[p["prompt"]] = float(p["presence"])
    elif kind == "single":
        (p,) = pairs
        row[p["prompt"]], row[1 - p["prompt"]] = float(p["presence"]), 0.0
    elif kind == "none":
        row = [0.0, 0.0]
    elif kind == "bg":
        row[1] = 0.0
    else:
        raise ValueError(kind)
    return row


class JointDataset(torch.utils.data.Dataset):
    """Items {"kind": both | single | none | bg, ...item}; key (item index, sample seed). A sample uses the per-sample
    code of the trunk stream the item comes from (USC12K pairs, salient / camouflaged positives, negatives); "single"
    items carry their own prompt id in "prompt"."""

    def __init__(self, items: list[dict], augment: bool = True):
        self.items, self.augment = items, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key: tuple[int, int]) -> dict:
        i, seed = key
        item = self.items[i]
        kind = item["kind"]
        code, prompts = JOINT_KINDS.get(kind) or ("region", (int(item["prompt"]),))
        s = T.dense_sample(item, code, prompts, "joint", seed, self.augment)
        return {"input": s["input"], "labels": joint_targets(kind, s["pairs"])}


class JointSampler(StreamSampler):
    """``StreamSampler`` over the joint stream alone; yields lists of (item index, sample seed)."""

    def __init__(self, n_items: int, n_iters: int, rank: int, world: int, seed: int = 0, start: int = 0):
        super().__init__((JOINT_STREAM,), {JOINT_STREAM.name: n_items}, n_iters, rank, world, seed, start)

    def batch(self, it: int) -> list[tuple[int, int]]:
        return [(i, sd) for _, i, sd in super().batch(it)]


def joint_collate(samples: list[dict]) -> dict:
    return {"input": torch.stack([s["input"] for s in samples]),
            "labels": torch.tensor([t for s in samples for t in s["labels"]], dtype=torch.float32)}

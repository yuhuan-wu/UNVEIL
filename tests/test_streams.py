"""Training streams: deterministic sampling, the paper's seed rules, joint presence targets."""

import numpy as np

from unveil.data import streams as D


def _sizes():
    return {s.name: 50 + 7 * k for k, s in enumerate(D.TRUNK_STREAMS)}


def test_batch_layout_and_resume():
    s = D.StreamSampler(D.TRUNK_STREAMS, _sizes(), 100, rank=3, world=8, seed=0)
    b = s.batch(5)
    assert len(b) == sum(st.per_gpu for st in D.TRUNK_STREAMS) == 24
    assert [k[0] for k in b] == sorted(k[0] for k in b)                       # streams in table order
    resumed = D.StreamSampler(D.TRUNK_STREAMS, _sizes(), 100, rank=3, world=8, seed=0, start=5)
    assert next(iter(resumed)) == b


def test_seed_rules():
    st = D.TRUNK_STREAMS[3]                                                    # shadow: perm key 3
    s = D.StreamSampler(D.TRUNK_STREAMS, _sizes(), 10, rank=1, world=8, seed=0)
    it, k = 4, 1
    pos = (it * st.per_gpu + k) * 8 + 1
    key = s.batch(it)[3 * 2 + k]
    n = _sizes()[st.name]
    assert key[1] == np.random.default_rng([0, 3, pos // n]).permutation(n)[pos % n]
    assert key[2] == pos * 11 + 3
    replay = D.TRUNK_STREAMS[10]
    assert replay.sample_seed(0, 17) == 10**12 + 17 and D.TRUNK_STREAMS[11].sample_seed(0, 17) == 3 * 10**12 + 51


def test_ranks_partition_positions():
    s = [D.StreamSampler(D.TRUNK_STREAMS, _sizes(), 10, r, 4, 0) for r in range(4)]
    seeds = [k[2] for r in range(4) for k in s[r].batch(2) if k[0] == 0]
    assert len(set(seeds)) == len(seeds) == 8


def test_joint_targets():
    pairs = [{"prompt": 0, "presence": 1}, {"prompt": 1, "presence": -1}]
    assert D.joint_targets("both", pairs) == [1.0, -1.0]
    assert D.joint_targets("single", [{"prompt": 1, "presence": 1}]) == [0.0, 1.0]
    assert D.joint_targets("none", []) == [0.0, 0.0]
    assert D.joint_targets("bg", []) == [D.NO_TARGET, 0.0]

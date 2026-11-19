"""Benchmark metrics and threshold rules on toy data."""

import numpy as np

from unveil.evaluation import binarize as Bz
from unveil.evaluation import metrics as M
from unveil.evaluation import scorecard as SC
from unveil.evaluation import thresholds as T


def test_region_metrics_at_threshold():
    gt = np.array([[255, 255, 0, 0]], np.uint8)
    pred = np.array([[200, 100, 100, 0]], np.uint8)
    r = M.region_record(pred, gt)
    assert M.iou(r, 128) == 0.5 and M.iou(r, 90) == 2 / 3
    ber = M.ber_dataset([r], 128)["BER"]
    assert np.isclose(ber, 100 * (1 - 0.5 * (0.5 + 1.0)))


def test_empty_maps_count_as_perfect_iou():
    r = M.region_record(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
    assert M.iou(r) == 1.0


def test_binarize_pick_ties_go_to_128():
    curve = np.zeros(255)
    assert Bz.pick(curve, "max") == 128
    curve[[39, 199]] = 1.0                                       # t = 40 and t = 200 best
    assert Bz.pick(curve, "max") == 200


def test_presence_tau_rule():
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    fit = T.best_tau(s, y)
    assert fit["balanced_acc"] == 1.0 and fit["tau"] == 0.5     # the plateau 0.201..0.8 contains 0.5


def test_task_scores():
    per = {s: {"S": 0.9} for s in ("duts_te", "ecssd", "hkuis", "dutomron", "pascals", "camo", "cod10k", "nc4k")}
    per.update({"sbu": {"BER": 3.0}, "istd": {"BER": 1.0}, "trans10k": {"IoU": 0.95}, "msd": {"IoU": 0.96},
                "bsds500": {"ODS": 0.85}, "sklarge": {"ODS": 0.8}, "sympascal": {"ODS": 0.6}})
    tasks, mean = SC.task_scores(per)
    assert np.isclose(tasks["shadow"], 98.0) and np.isclose(tasks["skeleton"], 70.0)
    assert np.isclose(mean, np.mean([90, 90, 98, 95, 96, 85, 70]))

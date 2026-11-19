"""Losses on small maps."""

import torch
import torch.nn.functional as F

from unveil.train import losses as L


def test_region_loss_matches_definition():
    logit = torch.tensor([[2.0, -1.0], [0.5, -3.0]])
    lab = torch.tensor([[255, 0], [255, 0]], dtype=torch.uint8)
    g = (lab >= 128).float()
    p = logit.sigmoid()
    ref = F.binary_cross_entropy_with_logits(logit, g) + 1 - (p * g).sum() / (p + g - p * g).sum()
    assert torch.allclose(L.region_loss(logit, lab), ref)


def test_shadow_loss_without_shadow_is_plain_bce_plus_iou():
    logit = torch.randn(4, 4)
    lab = torch.zeros(4, 4, dtype=torch.uint8)
    ref = F.binary_cross_entropy_with_logits(logit, torch.zeros(4, 4)) + 1.0
    assert torch.allclose(L.shadow_loss(logit, lab), ref)


def test_edge_loss_ignores_uncertain_pixels():
    logit = torch.zeros(1, 3)
    a = L.edge_loss(logit, torch.tensor([[255, 0, 60]], dtype=torch.uint8))
    b = L.edge_loss(logit, torch.tensor([[255, 0, 90]], dtype=torch.uint8))
    assert torch.allclose(a, b)


def test_skeleton_target_is_dilated():
    lab = torch.zeros(5, 5, dtype=torch.uint8)
    lab[2, 2] = 255
    assert int((L.dilate(lab) >= 128).sum()) == 9


def test_total_loss_group_mean_and_presence():
    z = torch.zeros(2, 2)
    one = torch.full((2, 2), 255, dtype=torch.uint8)
    total, parts = L.total_loss([z, z, None], [one, one, None], ["region", "region", "negative"],
                                ["sod", "cod", "neg_cam"], torch.tensor([0.0, 0.0, 0.0]),
                                torch.tensor([1.0, -1.0, 0.0]))
    assert set(parts) == {"sod", "cod", "presence"}
    assert torch.allclose(total, (parts["sod"] + parts["cod"]) / 2 + parts["presence"])
    assert torch.allclose(parts["presence"], torch.tensor(F.softplus(torch.tensor(0.0)).item()))

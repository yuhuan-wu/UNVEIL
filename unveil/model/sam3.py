"""Building and feeding the official SAM 3 image model.

UNVEIL never edits SAM 3's source. It builds the official model at the pinned commit, reuses its modules and adds a
trained copy of the last ViT blocks and of the heads next to them (``unveil.model``).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# github.com/facebookresearch/sam3 commit UNVEIL was trained and evaluated with.
SAM3_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
RES = 1008                      # SAM 3's input side; every image is stretched to RES x RES
SAM3_ENV = "UNVEIL_SAM3"        # environment variable with the path of a local sam3.pt
SAM3_LOCAL = Path(__file__).resolve().parents[2] / "checkpoints" / "sam3.pt"


def local_checkpoint(path=None) -> str | None:
    """The local ``sam3.pt`` to load, or None when there is none.

    Order: ``path``, the environment variable ``UNVEIL_SAM3``, ``checkpoints/sam3.pt`` of this repository.
    """
    given = path or os.environ.get(SAM3_ENV)
    if not given:
        return str(SAM3_LOCAL) if SAM3_LOCAL.is_file() else None
    given = Path(given).expanduser()
    if not given.is_file():
        raise FileNotFoundError(f"SAM 3 checkpoint {given} not found: see checkpoints/README.md")
    return str(given)


def build_sam3(device: str = "cuda", checkpoint: str | None = None):
    """The official SAM 3 image model, frozen and in eval mode.

    checkpoint: a local ``sam3.pt``, looked up as in ``local_checkpoint``. Without a local file, ``facebook/sam3`` is
    downloaded from Hugging Face (gated: accept its licence on the model page and log in once with ``hf auth login``).
    """
    from sam3.model_builder import build_sam3_image_model

    checkpoint = local_checkpoint(checkpoint)

    # The builder only moves the model when device == "cuda"; build on the CPU and move it to the caller's device.
    # Its positional-encoding caches are created on the current CUDA device, so that must be the target device.
    ctx = torch.cuda.device(device) if str(device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        model = build_sam3_image_model(device="cpu", eval_mode=True, enable_segmentation=True,
                                       checkpoint_path=checkpoint, load_from_HF=checkpoint is None)
    return model.to(device).eval().requires_grad_(False)


def set_tf32(on: bool) -> None:
    """Training runs with TF32 on, evaluation with TF32 off (SAM 3's builder turns it on at import)."""
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def to_input(image, res: int = RES) -> torch.Tensor:
    """PIL image -> uint8 tensor (3, res, res) by a PIL bilinear stretch (the transform of training and evaluation)."""
    from PIL import Image

    a = np.asarray(image.resize((res, res), Image.BILINEAR), dtype=np.uint8)
    return torch.from_numpy(a.copy()).permute(2, 0, 1)


def normalize(img_u8: torch.Tensor) -> torch.Tensor:
    """uint8 -> float with mean = std = 0.5 (SAM 3's transform)."""
    return (img_u8.float() / 255.0 - 0.5) / 0.5


def check_trunk(sam3, first: int) -> None:
    t = sam3.backbone.vision_backbone.trunk
    if len(t.blocks) != 32 or t.full_attn_ids[-1] != 31 or t.retain_cls_token or not 0 < first < 32:
        raise RuntimeError(f"unexpected SAM 3 ViT; install SAM 3 at commit {SAM3_COMMIT}")


# --------------------------------------------------------------------------------------------- ViT MLP for training

_ORIG_MLP_FORWARD = None


def _mlp_forward(self, x):
    """SAM 3's ``vitdet.Mlp`` unfused when gradients are on. The pinned SAM 3 runs this MLP through an
    inference-only fused kernel (bf16 addmm with a tanh-GELU epilogue) that raises under autograd. This computes the same
    function: fc1 under the caller's autocast, tanh-approximate GELU, then norm, fc2 and the dropouts."""
    if not torch.is_grad_enabled():
        return _ORIG_MLP_FORWARD(self, x)
    h = self.fc1(x)
    h = F.relu(h) if isinstance(self.act, nn.ReLU) else F.gelu(h, approximate="tanh")
    return self.drop2(self.fc2(self.norm(self.drop1(h))))


def enable_vit_training() -> None:
    """Make the ViT MLP differentiable (training of the LoRA branch). Inference keeps the fused kernel. Idempotent."""
    global _ORIG_MLP_FORWARD
    from sam3.model import vitdet

    if vitdet.Mlp.forward is not _mlp_forward:
        _ORIG_MLP_FORWARD = vitdet.Mlp.forward
        vitdet.Mlp.forward = _mlp_forward

"""The UNVEIL checkpoint: one safetensors file with every trained tensor and the presence thresholds.

Tensor groups (key prefixes):
- ``heads.``  fusion encoder, semantic path and presence decoder (``ConceptHeads``; the trained parts only).
- ``lora.blocks.<k>.<target>.lora_A|lora_B``  LoRA of ViT block ``first_block + k``; merged in fp32 at load time.
- ``joint.decoder.``  the joint presence decoder (its trained parts).
- ``joint.convention``  the convention vectors (2 x 256), rows in the order of ``prompts.JOINT_PAIR``.

Untrained parts (the ViT, the text and geometry encoders, the box head, ...) come from SAM 3's official checkpoint.
The safetensors metadata (key "unveil", JSON) holds the format version, the SAM 3 commit, the LoRA settings, the
prompts, the presence thresholds and the binarization thresholds of the region metrics (``evaluation.binarize``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

FORMAT, FORMAT_VERSION, META_KEY = "unveil-checkpoint", 1, "unveil"
DEFAULT = Path(__file__).resolve().parents[1] / "checkpoints" / "unveil_v1.safetensors"
LORA_DEFAULT = {"r": 16, "alpha": 32.0, "first_block": 24, "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]}


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def group(tensors: dict, prefix: str) -> dict:
    """Tensors under ``prefix``, with the prefix removed."""
    n = len(prefix)
    return {k[n:]: v for k, v in tensors.items() if k.startswith(prefix)}


def assemble(heads: dict, lora: dict, joint_decoder: dict, convention: torch.Tensor) -> dict[str, torch.Tensor]:
    """Trained state -> the flat tensor dict of the file (fp32, contiguous, on the CPU)."""
    out = {}
    for prefix, part in (("heads.", heads), ("lora.", lora), ("joint.decoder.", joint_decoder)):
        out.update({prefix + k: v.detach().float().cpu().contiguous() for k, v in sorted(part.items())})
    out["joint.convention"] = convention.detach().float().cpu().contiguous()
    return out


def make_meta(thresholds: dict, lora: dict | None = None, extra: dict | None = None) -> dict:
    """thresholds: {"single": {prompt: tau}, "joint": {prompt: tau}, "free_text": tau, "rule": str}."""
    from unveil import prompts as P
    from unveil.model.sam3 import SAM3_COMMIT

    return {"model": "UNVEIL", "sam3": {"repo": "facebook/sam3", "file": "sam3.pt", "commit": SAM3_COMMIT},
            "lora": {**LORA_DEFAULT, **(lora or {})}, "prompts": list(P.TASK_PROMPTS), "joint_pair": list(P.JOINT_PAIR),
            "thresholds": thresholds, **(extra or {})}


def save(path, tensors: dict, meta: dict) -> str:
    """Write the file and a ``.sha256`` file next to it -> sha256. The same inputs give the same bytes."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT, "format_version": FORMAT_VERSION, **meta}
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp), metadata={META_KEY: json.dumps(meta, sort_keys=True)})
    os.replace(tmp, path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def read_meta(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(META_KEY)
    if raw is None:
        raise ValueError(f"{path} is not an UNVEIL checkpoint (no '{META_KEY}' metadata)")
    meta = json.loads(raw)
    if meta.get("format") != FORMAT or int(meta.get("format_version", -1)) > FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {meta.get('format')} v{meta.get('format_version')}")
    return meta


def verify(path, sha256: str | None = None) -> str | None:
    """Check the sha256 against the argument or the ``.sha256`` file next to it -> the digest, or None if neither."""
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    want = sha256 or (side.read_text().split()[0] if side.exists() else None)
    if want is None:
        return None
    got = sha256_file(path)
    if got != want.lower():
        raise ValueError(f"sha256 mismatch for {path}: expected {want}, got {got}")
    return got


def load(path=None, sha256: str | None = None) -> tuple[dict, dict, Path]:
    """-> (tensors on the CPU, metadata, path)."""
    from safetensors.torch import load_file

    path = Path(path or DEFAULT).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path}: see checkpoints/README.md for the download")
    verify(path, sha256)
    return load_file(str(path), device="cpu"), read_meta(path), path

"""UNVEIL inference: SAM 3 plus the trained copies of its last ViT blocks and heads.

    import unveil
    model = unveil.load("checkpoints/unveil_v1.safetensors")
    mask, presence = model.predict(image, "shadow")
    masks, presences = model.predict(image, ["salient object", "camouflaged object"], joint=True)

``mask`` is a float32 (H, W) probability map at the image size. It is all zero when the presence probability is below
the prompt's threshold (``gate=False`` returns the map regardless). The image passes SAM 3's shared ViT blocks and the
copies once, however many prompts it gets. SAM 3 itself stays unchanged: ``model.sam3`` is the official model,
``model.instances`` returns its native instance output and ``path="sam3"`` reads its frozen semantic path.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from unveil import checkpoint as ckpt
from unveil import prompts as P
from unveil.model import backbone as B
from unveil.model import sam3 as S
from unveil.model.heads import ConceptHeads, coarse_to, joint_decoder_from
from unveil.model.lora import merged_blocks

TEXT_CACHE = 4096                   # free-text phrases kept encoded; the task prompts are always kept


def as_pil(image):
    """PIL image, (H, W[, 3]) array or path -> RGB PIL image."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        a = image if image.ndim == 3 else np.stack([image] * 3, -1)
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).round().astype(np.uint8)
        return Image.fromarray(a[..., :3]).convert("RGB")
    raise TypeError(f"unsupported image type {type(image).__name__}")


class UNVEIL:
    def __init__(self, sam3, tensors: dict, meta: dict, device: str, dtype=torch.bfloat16):
        self.sam3, self.meta, self.device, self.dtype = sam3, meta, device, dtype
        lora = meta["lora"]
        self.first = lora["first_block"]
        S.check_trunk(sam3, self.first)
        self.tail = merged_blocks(sam3, ckpt.group(tensors, "lora."), lora["r"], lora["alpha"], self.first).to(device)
        self.native_tail = B.native_tail(sam3, self.first)
        self.heads = ConceptHeads.from_sam3(sam3, copy=True)
        self.heads.load_trained(ckpt.group(tensors, "heads."))
        self.heads.to(device)
        self.native_heads = ConceptHeads.from_sam3(sam3, copy=False)   # SAM 3's own modules (frozen semantic path)
        self.joint_decoder = joint_decoder_from(self.heads, ckpt.group(tensors, "joint.decoder.")).to(device)
        self.convention = tensors["joint.convention"].float().to(device)   # (2, C), rows in the order of JOINT_PAIR
        th = meta["thresholds"]
        self.tau = {P.normalize(k): float(v) for k, v in th["single"].items()}
        self.tau_free = float(th["free_text"])
        self.tau_joint = {P.normalize(k): float(v) for k, v in th["joint"].items()}
        self.binarize = {k: int(v) for k, v in meta.get("binarize", {}).items()}      # metric thresholds per concept
        self._text: dict[str, dict] = {}
        self._processor = None
        for p, t in zip(P.TASK_PROMPTS, self._encode(list(P.TASK_PROMPTS))):   # one call, as in training
            self._text[p] = t

    # ------------------------------------------------------------------------------------------------ text

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.dtype)

    @torch.no_grad()
    def _encode(self, phrases: list[str]) -> list[dict]:
        with self._autocast():
            t = self.sam3.backbone.forward_text(list(phrases), device=self.device)
        return [{"language_features": t["language_features"][:, i:i + 1], "language_mask": t["language_mask"][i:i + 1],
                 "language_embeds": t["language_embeds"][:, i:i + 1]} for i in range(len(phrases))]

    def text(self, phrases: list[str]) -> dict:
        """Encoded phrases (cached), concatenated in the layout of ``backbone.forward_text``."""
        keys = [P.normalize(p) for p in phrases]
        new = list(dict.fromkeys(k for k in keys if k not in self._text))
        if new:
            self._text.update(zip(new, self._encode(new)))
        free = [k for k in self._text if k not in P.TASK_PROMPTS and k not in keys]
        for k in free[:max(0, len(free) - TEXT_CACHE)]:                  # oldest free-text phrases first
            del self._text[k]
        parts = [self._text[k] for k in keys]
        return {"language_features": torch.cat([x["language_features"] for x in parts], 1),
                "language_mask": torch.cat([x["language_mask"] for x in parts], 0),
                "language_embeds": torch.cat([x["language_embeds"] for x in parts], 1)}

    # ------------------------------------------------------------------------------------------------ features

    @torch.no_grad()
    def features(self, images: torch.Tensor, img_ids, phrases: list[str], tail: str = "unveil") -> dict:
        """Pair features. images: uint8 (B, 3, 1008, 1008); pair k is image ``img_ids[k]`` with ``phrases[k]``. Each
        image passes the ViT once, whatever its number of pairs."""
        x = S.normalize(images.to(self.device))
        with self._autocast():
            pre = B.vit_prefix(self.sam3, x, self.first)
            bb = B.image_features(self.sam3, pre, self.tail if tail == "unveil" else self.native_tail)
            return B.pair_features(self.sam3, bb, self.text(phrases), torch.as_tensor(img_ids),
                                   torch.arange(len(phrases)))

    @torch.no_grad()
    def single(self, images: torch.Tensor, img_ids, phrases: list[str], path: str = "unveil"):
        """Single queries -> (semantic logits (P, 1, 288, 288), presence logits (P,)); path "sam3": SAM 3's own
        semantic path and presence."""
        heads = self.heads if path == "unveil" else self.native_heads
        feats = self.features(images, img_ids, phrases, path)
        with self._autocast():
            return heads.single(feats)

    @torch.no_grad()
    def joint(self, images: torch.Tensor, img_ids: torch.Tensor, own: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint queries. images: uint8 (B, 3, 1008, 1008); query k asks for member ``JOINT_PAIR[own[k]]`` of image
        ``img_ids[k]``. Each query runs both members, each with its convention vector, and returns its own member's
        (semantic logits (n, 1, 288, 288), joint presence logits (n,))."""
        n = len(own)
        t = self.text(list(P.JOINT_PAIR))
        with self._autocast():
            x = S.normalize(images.to(self.device))
            pre = B.vit_prefix(self.sam3, x, self.first)
            bb = B.image_features(self.sam3, pre, self.tail)
            f = B.pair_features(self.sam3, bb, t, img_ids, own)
            both = torch.cat([own, 1 - own]).to(self.device)            # own members first, then their partners
            members = {"size": f["size"], "src": torch.cat([f["src"]] * 2, 1), "pos": torch.cat([f["pos"]] * 2, 1),
                       "fpn": [torch.cat([x] * 2) for x in f["fpn"]], "geo": torch.cat([f["geo"]] * 2, 1),
                       "geo_mask": torch.cat([f["geo_mask"]] * 2),
                       "txt": t["language_features"][:, both].to(f["txt"].dtype),
                       "txt_mask": t["language_mask"][both].to(f["txt_mask"].dtype)}
            coarse, pres = self.heads.joint(members, self.convention[both], self.joint_decoder)
        return coarse[:n], pres[:n]

    def threshold(self, prompt: str, joint: bool = False) -> float:
        """Presence threshold of a prompt, fixed before testing; free text uses the pooled value."""
        key = P.normalize(prompt)
        return self.tau_joint[key] if joint else self.tau.get(key, self.tau_free)

    # ------------------------------------------------------------------------------------------------ predict

    @torch.no_grad()
    def predict(self, image, prompt, joint: bool = False, gate: bool = True, path: str = "unveil",
                return_logits: bool = False, size: tuple[int, int] | None = None):
        """One query or several on one image.

        prompt: a phrase or a list of phrases. joint=True: the joint query {"salient object", "camouflaged object"}.
        gate=False: the ungated map (the benchmarks score these). path="sam3": SAM 3's frozen semantic path, for
        comparison (no gate; ``presence`` is then SAM 3's own presence probability). size: (H, W) of the outputs
        (default: the image size); the logit is resized bilinearly before the sigmoid.
        -> (mask, presence) for a phrase, (masks, presences) for a list.
        """
        many = not isinstance(prompt, str)
        phrases = list(prompt) if many else [prompt]
        if not phrases:
            raise ValueError("no prompt")
        if joint and (path != "unveil" or not P.is_joint_pair(phrases)):
            raise ValueError(f"joint=True needs exactly the pair {list(P.JOINT_PAIR)} on the UNVEIL path")
        if path not in ("unveil", "sam3"):
            raise ValueError(f"unknown path {path!r}")
        pil = as_pil(image)
        hw = tuple(size) if size is not None else (pil.height, pil.width)
        x, zeros = S.to_input(pil)[None], torch.zeros(len(phrases), dtype=torch.long)
        if joint:
            own = torch.tensor([P.JOINT_PAIR.index(P.normalize(p)) for p in phrases])
            coarse, pres = self.joint(x, zeros, own)
        else:
            coarse, pres = self.single(x, zeros, phrases, path)
        masks, probs = [], []
        for k, ph in enumerate(phrases):
            logit = coarse_to(coarse[k:k + 1], hw)[0, 0]
            p = float(torch.sigmoid(pres[k]))
            m = logit if return_logits else torch.sigmoid(logit)
            if gate and not return_logits and path == "unveil" and p < self.threshold(ph, joint):
                m = torch.zeros_like(m)
            masks.append(m.float().cpu().numpy())
            probs.append(p)
        return (masks, probs) if many else (masks[0], probs[0])

    # ------------------------------------------------------------------------------------------------ native SAM 3

    @torch.no_grad()
    def instances(self, image, prompt: str, presence_one: bool = False, confidence: float = 0.5) -> dict:
        """SAM 3's native instance output through the official ``Sam3Processor`` (weights untouched).
        presence_one=True scores each detection without the presence factor (``unveil.model.veto``)."""
        from unveil.model.veto import processor_for

        if self._processor is None:
            self._processor = processor_for(self.sam3, self.device, presence_one, confidence)
        self._processor.lift = presence_one
        self._processor.confidence_threshold = confidence
        with self._autocast():
            state = self._processor.set_image(as_pil(image))
            return self._processor.set_text_prompt(prompt, state)


def load(weights=None, device: str = "cuda", sam3_checkpoint: str | None = None, sha256: str | None = None,
         dtype=torch.bfloat16, tf32: bool = False, sam3_model=None) -> UNVEIL:
    """Build SAM 3 and load the UNVEIL checkpoint.

    weights: a local .safetensors file (default: ``checkpoints/unveil_v1.safetensors`` of this repository).
    sam3_checkpoint: a local ``sam3.pt`` (default: ``$UNVEIL_SAM3``, else ``checkpoints/sam3.pt`` of this repository,
    else the download of ``facebook/sam3`` from Hugging Face, which is gated).
    sha256: expected hash of the weight file (default: the ``.sha256`` file next to it, if present).
    dtype: autocast dtype, bfloat16 as in the paper's runs (SAM 3's fused ViT MLP needs a 16-bit autocast).
    tf32: TF32 matmuls (off in the paper's runs).
    sam3_model: an already built official SAM 3 image model to reuse; it is not modified.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("UNVEIL needs a CUDA GPU (SAM 3's model builder allocates on CUDA)")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("dtype must be torch.bfloat16 (the paper's runs) or torch.float16")
    tensors, meta, _ = ckpt.load(weights, sha256)
    if sam3_model is None:
        sam3_model = S.build_sam3(device, sam3_checkpoint)
    S.set_tf32(tf32)
    return UNVEIL(sam3_model, tensors, meta, device, dtype)

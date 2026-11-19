"""The prompts UNVEIL was trained on.

The strings are exact. The presence thresholds stored in the checkpoint belong to them, and the training data refer to
them by index (``TASK_PROMPTS.index(p)`` is the prompt id). Any other phrase is accepted as free text: category names
work through SAM 3's vocabulary, kept by concept replay; other phrases use the pooled presence threshold.
"""

from __future__ import annotations

# Task prompts in prompt-id order (training data and the checkpoint's threshold table use these ids).
TASK_PROMPTS: tuple[str, ...] = (
    "salient object",       # 0
    "camouflaged object",   # 1
    "edges",                # 2
    "shadow",               # 3
    "transparent object",   # 4
    "mirror",               # 5
    "skeleton",             # 6  SK-LARGE
    "symmetry axis",        # 7  SYM-PASCAL
)

# The joint query {salient object, camouflaged object}: the only pair with joint labels (USC12K).
JOINT_PAIR: tuple[str, str] = ("salient object", "camouflaged object")

# The four kinds of concept of the paper (Sec. 3.1). Identity concepts are category names, asked as free text.
KINDS: dict[str, tuple[str, ...]] = {
    "Identity": (),
    "Light": ("shadow", "transparent object", "mirror"),
    "Context": ("salient object", "camouflaged object"),
    "Structure": ("edges", "skeleton", "symmetry axis"),
}

# Prompts whose presence decoder saw verified negatives in training (LVIS category names did as well).
TRAINED_REJECTION: tuple[str, ...] = ("salient object", "camouflaged object")

# "thin": soft line maps (benchmarks apply NMS and thinning); every other prompt gives a region map.
THIN: tuple[str, ...] = ("edges", "skeleton", "symmetry axis")


def normalize(prompt: str) -> str:
    """Matching key of a prompt: stripped, single spaces, lower case (SAM 3's tokenizer lower-cases too)."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
    key = " ".join(prompt.strip().split()).lower()
    if not key:
        raise ValueError("empty prompt")
    return key


def task_id(prompt: str) -> int | None:
    """Prompt id of a task prompt, or None for free text."""
    key = normalize(prompt)
    return TASK_PROMPTS.index(key) if key in TASK_PROMPTS else None


def is_joint_pair(prompts) -> bool:
    keys = [normalize(p) for p in prompts]
    return len(keys) == 2 and set(keys) == set(JOINT_PAIR)


def is_thin(prompt: str) -> bool:
    return normalize(prompt) in THIN

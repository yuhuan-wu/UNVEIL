"""UNVEIL: promptable concept maps from SAM 3's semantic path.

One text phrase in, one map (or "none") out, for salient and camouflaged objects, shadows, transparent objects,
mirrors, edges, skeletons and SAM 3's object categories, with one set of weights next to an unchanged SAM 3.

    import unveil
    model = unveil.load("checkpoints/unveil_v1.safetensors")
    mask, presence = model.predict("photo.jpg", "shadow")
"""

from unveil.prompts import JOINT_PAIR, KINDS, TASK_PROMPTS

__version__ = "1.0.0"
__all__ = ["load", "UNVEIL", "TASK_PROMPTS", "JOINT_PAIR", "KINDS", "__version__"]


def load(*args, **kwargs):
    """Build SAM 3 and load an UNVEIL checkpoint (``unveil.model.unveil.load``)."""
    from unveil.model.unveil import load as _load

    return _load(*args, **kwargs)


def __getattr__(name):                     # lazy: importing unveil does not import torch or SAM 3
    if name == "UNVEIL":
        from unveil.model.unveil import UNVEIL

        return UNVEIL
    raise AttributeError(name)

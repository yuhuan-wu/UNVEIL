"""Figure grids: one image with several prompts, and test examples next to their ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from unveil.viz import render as R


def grid(cells: list[list[Image.Image]], col_titles: list[str], row_titles: list[str], out, cell: float = 2.2,
          dpi: int = 150) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(cells), len(cells[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows), squeeze=False)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(cells[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0 and row_titles:
                ax.set_ylabel(row_titles[r], fontsize=9)
    fig.tight_layout(pad=0.4)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def prompts_panel(image, results: list[tuple[str, np.ndarray, float, bool]], out) -> Path:
    """One image and its answers. results: (prompt, map in [0, 1], presence probability, accepted)."""
    cells, titles = [R.to_rgb(image)], ["image"]
    for prompt, prob, pres, ok in results:
        cells.append(np.asarray(R.overlay(image, prob, prompt)))
        titles.append(f"{prompt}\npresence {pres:.2f}" + ("" if ok else " (none)"))
    return grid([cells], titles, [], out)


def examples_grid(rows: list[dict], out, row_names=("Image", "UNVEIL", "Ground truth")) -> Path:
    """Test examples as columns: image, the model's map and the ground truth, drawn the same way.
    rows: [{"title", "prompt", "image" (RGB array), "pred" (map in [0, 1]), "gt" (map in [0, 1])}]."""
    cells = [[r["image"] for r in rows],
             [np.asarray(R.overlay(r["image"], r["pred"], r["prompt"])) for r in rows],
             [np.asarray(R.overlay(r["image"], r["gt"], r["prompt"])) for r in rows]]
    return grid(cells, [r["title"] for r in rows], list(row_names), out)

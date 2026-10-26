"""Shared helpers for experiment scripts: output folders, JSON results, consistent figures.

Figure style: fixed categorical order (never cycled), one y-axis per chart, thin marks,
recessive grid, a legend whenever there are >= 2 series. Static PNG + PDF for the paper.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .trace import _jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES), "figure.dpi": 110,
})


def quick() -> bool:
    return os.environ.get("CAGE_QUICK", "0") == "1"


def outdir(paper: str, exp: str) -> Path:
    d = ROOT / "results" / paper / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(d: Path, obj, name: str = "results.json") -> None:
    (d / name).write_text(json.dumps(_jsonable(obj), indent=2))


def save_fig(fig, d: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(d / f"{name}.png", dpi=150)
    fig.savefig(d / f"{name}.pdf")
    plt.close(fig)


def write_summary(d: Path, title: str, lines: list[str], scenario: bool = True) -> None:
    """Every summary states whether numbers come from synthetic scenarios / analytic models
    (epistemic rule E1)."""
    head = [f"# {title}", "",
            f"_Generated {time.strftime('%Y-%m-%d %H:%M')}"
            + (" · quick mode_" if quick() else "_"), ""]
    if scenario:
        head += ["> **Scenario values.** Inputs are synthetic workload models and costs come from "
                 "an analytic model. These are not measurements on hardware.", ""]
    (d / "summary.md").write_text("\n".join(head + lines) + "\n")


def md_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> list[str]:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            cells.append(f.format(v) if f and isinstance(v, (int, float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out

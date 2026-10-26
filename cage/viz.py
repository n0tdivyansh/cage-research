"""Plots for single-trace reports."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import expkit as X  # noqa: F401  (applies figure style)
from . import metrics as m
import matplotlib.pyplot as plt

SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#2a78d6", "#0d2f5c"])


def traffic_heatmap(M: np.ndarray, path: Path) -> None:
    S = M + M.T
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if (S > 0).any():
        im = ax.imshow(np.log10(S + S[S > 0].min() * 0.1), cmap=SEQ, interpolation="nearest")
        fig.colorbar(im, ax=ax, label="log10 bytes per step", shrink=0.8)
    ax.grid(False)
    ax.set(xlabel="rank", ylabel="rank", title="Traffic matrix (symmetric)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def strength_bars(M: np.ndarray, path: Path) -> None:
    s = m.strength(M)["total"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(np.arange(len(s)), s, width=0.9, color=X.SERIES[0])
    ax.axhline(np.median(s), color=X.INK2, lw=1, ls="--")
    ax.set(xlabel="rank", ylabel="bytes per step (sent + received)", title="Communication load per rank")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def graph_layout(M: np.ndarray, labels: np.ndarray, path: Path, seed: int = 0) -> None:
    G = m.undirected(M)
    if G.number_of_edges() == 0:
        return
    pos = nx.spring_layout(G, weight="weight", seed=seed)
    s = m.strength(M)["total"]
    sizes = 20 + 280 * s / s.max()
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    k = len(np.unique(labels))
    palette = X.SERIES if k <= 8 else None
    colors = [palette[l] for l in labels] if palette else labels
    fig, ax = plt.subplots(figsize=(5.4, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3 + 2.5 * w / w.max(), edge_color="#b9b8b1", alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           cmap=None if palette else "tab20", edgecolors="#fcfcfb", linewidths=1.5)
    ax.set_axis_off()
    ax.set_title(f"Communication graph · {k} communities" + ("" if palette else " (>8: colors repeat)"))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

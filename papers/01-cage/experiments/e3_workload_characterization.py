"""E3 — Characterise communication graphs across workload classes (Claim 1, H1, H2).

For each synthetic workload: load concentration (Gini, top-20% share — the honest form of H2),
outlier hotspots, density, Louvain Q with null-model z. Also draws traffic-matrix heatmaps, the
most informative single picture of a communication graph.
Expected: DP collectives are balanced (H2 fails), Megatron and stencils carry significant
structure, MoE concentration grows with routing skew.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cage import expkit as X, metrics as m, workloads as W

d = X.outdir("01-cage", "e3_workload_characterization")
n_null = 20 if not X.quick() else 4
wl = {
    "DP ring": W.dp_ring(64),
    "DP hier. leader": W.dp_hier(64, variant="leader"),
    "DP hier. rail": W.dp_hier(64, variant="rail"),
    "Megatron 8·4·2": W.megatron_3d(8, 4, 2),
    "MoE zipf 0": W.moe(zipf=0.0),
    "MoE zipf 1.0": W.moe(zipf=1.0),
    "MoE zipf 1.5": W.moe(zipf=1.5),
    "MoE zipf 1 aff 0.6": W.moe(zipf=1.0, affinity=0.6),
    "Stencil 2D 8x8": W.stencil((8, 8)),
    "Stencil 3D 4x4x4": W.stencil((4, 4, 4)),
}
rows, mats = [], {}
for name, t in wl.items():
    M = t.matrix(per_step=True)
    mats[name] = M
    ld = m.load_summary(M)
    sig = m.modularity_significance(M, n_null=n_null, seed=0)
    rows.append({"workload": name, "ranks": t.n_ranks, "density": float((M + M.T > 0).sum() / (t.n_ranks * (t.n_ranks - 1))),
                 "gini": ld["gini"], "top20_share": ld["top20_share"], "max/median": ld["max_over_median"],
                 "hotspots": len(m.hotspots(M)), "Q": sig["Q"], "z": sig["z"], "p": sig["p_empirical"],
                 "H2_holds(top20>=0.7)": ld["top20_share"] >= 0.7})
X.save_json(d, rows)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
y = np.arange(len(rows))
ax.barh(y, [r["top20_share"] for r in rows], color=X.SERIES[0])
ax.axvline(0.2, color=X.INK2, lw=1, ls=":")
ax.axvspan(0.7, 0.8, color=X.SERIES[1], alpha=0.15, lw=0)
ax.text(0.75, len(rows) - 0.3, "H2 claim", ha="center", color=X.INK2, fontsize=8)
ax.text(0.21, len(rows) - 0.3, "perfect balance", color=X.INK2, fontsize=8)
for i, r in enumerate(rows):
    ax.text(r["top20_share"] + 0.01, i, f"{r['top20_share']:.2f}", va="center", fontsize=8, color=X.INK2)
ax.set_yticks(y, [r["workload"] for r in rows]); ax.invert_yaxis()
ax.set(xlim=(0, 1), xlabel="share of traffic handled by the busiest 20% of ranks",
       title="Load concentration by workload")
X.save_fig(fig, d, "load_concentration")

cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", X.SERIES[0], "#0d2f5c"])
fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
for ax, (name, M) in zip(axes.flat, mats.items()):
    S = M + M.T
    ax.imshow(np.log10(S + S[S > 0].min() * 0.1) if (S > 0).any() else S, cmap=cmap, interpolation="nearest")
    ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Traffic matrices (rank × rank, log bytes, darker = more)", fontsize=11)
X.save_fig(fig, d, "traffic_matrices")
X.write_summary(d, "E3 — Workload characterisation", X.md_table(
    rows, ["workload", "ranks", "density", "gini", "top20_share", "max/median", "hotspots", "Q", "z", "p", "H2_holds(top20>=0.7)"],
    {**{c: "{:.2f}" for c in ["density", "gini", "top20_share", "max/median", "Q", "z"]}, "p": "{:.3f}"}))
print(open(d / "summary.md").read())

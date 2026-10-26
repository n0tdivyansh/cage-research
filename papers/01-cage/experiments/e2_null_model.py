"""E2 — Raw modularity Q is not evidence of structure (fixes H1 / 'Q > 0.3' threshold).

Compares observed Louvain Q with degree-preserving null graphs for graphs that are known to have
no community structure (ring, random sparse, complete) and graphs that do (SBM, Megatron 3D).
Expected: structureless sparse graphs reach Q >= 0.3 with z ~ 0, so a fixed Q threshold misfires.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, metrics as m, workloads as W
from cage.trace import TraceBuilder

d = X.outdir("01-cage", "e2_null_model")
rng = np.random.default_rng(0)
n_null = 20 if not X.quick() else 5


def er(n, deg, seed):
    r = np.random.default_rng(seed)
    A = np.triu(r.random((n, n)) < deg / (n - 1), 1).astype(float)
    return A + A.T


def complete(n):
    return np.ones((n, n)) - np.eye(n)


cases = {
    "ring (64)": W.dp_ring(64).matrix(per_step=True),
    "random sparse, deg 4 (64)": er(64, 4, 1),
    "random dense, deg 32 (64)": er(64, 32, 2),
    "complete, uniform (64)": complete(64),
    "2D stencil 8x8": W.stencil((8, 8)).matrix(per_step=True),
    "hier. leader allreduce (32)": W.dp_hier(32).matrix(per_step=True),
    "SBM, mu≈0.15 (64)": W.planted_partition(64, 8, 0.6, 0.0129).matrix(per_step=True),
    "Megatron TP8·PP4·DP2": W.megatron_3d(8, 4, 2).matrix(per_step=True),
    "MoE, no affinity (64)": W.moe().matrix(per_step=True),
}
rows = []
for name, M in cases.items():
    s = m.modularity_significance(M, n_null=n_null, seed=0)
    rows.append({"graph": name, "Q": s["Q"], "null_Q": s["null_mean"], "null_sd": s["null_std"],
                 "z": s["z"], "p": s["p_empirical"], "naive_rule_Q>0.3": s["Q"] > 0.3,
                 "significant_z>3": s["z"] > 3})
X.save_json(d, rows)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = np.arange(len(rows))
ax.barh(y + 0.2, [r["Q"] for r in rows], 0.38, label="observed Q", color=X.SERIES[0])
ax.barh(y - 0.2, [r["null_Q"] for r in rows], 0.38, label="null-model Q (mean)", color=X.SERIES[1])
ax.axvline(0.3, color=X.INK2, lw=1, ls="--")
ax.text(0.31, len(rows) - 0.4, "naive threshold 0.3", color=X.INK2, fontsize=8)
ax.set_yticks(y, [r["graph"] for r in rows])
ax.invert_yaxis()
ax.set(xlabel="Louvain modularity Q", title="Observed vs. null-model modularity")
ax.legend(loc="lower right")
X.save_fig(fig, d, "q_vs_null")
X.write_summary(d, "E2 — Modularity needs a null model", X.md_table(
    rows, ["graph", "Q", "null_Q", "z", "p", "naive_rule_Q>0.3", "significant_z>3"],
    {"Q": "{:.3f}", "null_Q": "{:.3f}", "z": "{:.1f}", "p": "{:.3f}"}), scenario=False)
print(open(d / "summary.md").read())

"""E1 — Does CAGE recover planted structure? (supports Claims 1 & 3; OQ4 option A)

Planted-partition graphs (8 blocks of 8 ranks) with mixing mu = expected share of a rank's
traffic that leaves its block. Ranks are randomly relabelled so rank order carries no hint.
Measures: Louvain NMI vs. truth, METIS (exact capacity 8) ARI vs. truth, Q and null-model z.
Expected: recovery near 1 at low mu, collapsing as mu -> (k-1)/k = 0.875 (random).
The interesting number is where z drops below ~3: that is CAGE's detection limit.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, metrics as m, partition as P, workloads as W

d = X.outdir("01-cage", "e1_synthetic_recovery")
mus = [0.05, 0.15, 0.3, 0.45, 0.6, 0.75] if not X.quick() else [0.05, 0.3, 0.6]
seeds = range(5 if not X.quick() else 2)
n, k, p_in = 64, 8, 0.6
rows = []
for mu in mus:
    p_out = mu * (n / k - 1) * p_in / ((n - n / k) * (1 - mu))
    for s in seeds:
        t = W.planted_partition(n, k, p_in, min(p_out, 1.0), seed=s)
        perm = np.random.default_rng(100 + s).permutation(n)
        t = t.relabel(perm)
        truth = t.meta["labels"]["block"]
        M = t.matrix(per_step=True)
        sig = m.modularity_significance(M, n_null=20 if not X.quick() else 4, seed=s)
        part = P.partition(M, n // k, "best", seed=s)
        Wm = M + M.T
        rows.append({"mu": mu, "seed": s, "louvain_nmi": m.nmi(sig["labels"], truth),
                     "metis_ari": m.ari(part, truth), "Q": sig["Q"], "z": sig["z"],
                     "n_comm": sig["n_communities"],
                     "cut_reduction_vs_block": 1 - P.cut_weight(Wm, part) / P.cut_weight(Wm, P.block(n, n // k))})
agg = []
for mu in mus:
    r = [x for x in rows if x["mu"] == mu]
    agg.append({"mu": mu, **{key: float(np.mean([x[key] for x in r])) for key in
                             ("louvain_nmi", "metis_ari", "Q", "z", "n_comm", "cut_reduction_vs_block")}})
X.save_json(d, {"rows": rows, "mean": agg})

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot([a["mu"] for a in agg], [a["louvain_nmi"] for a in agg], marker="o", label="Louvain NMI")
ax.plot([a["mu"] for a in agg], [a["metis_ari"] for a in agg], marker="s", label="METIS ARI (capacity 8)")
ax.set(xlabel="mixing μ (share of traffic leaving the block)", ylabel="agreement with planted blocks",
       ylim=(-0.05, 1.05), title="Recovery of planted structure")
ax.legend()
X.save_fig(fig, d, "recovery_vs_mixing")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot([a["mu"] for a in agg], [a["z"] for a in agg], marker="o", color=X.SERIES[0])
ax.axhline(3, color=X.INK2, lw=1, ls="--")
ax.text(agg[-1]["mu"], 3.3, "z = 3", ha="right", color=X.INK2)
ax.set(xlabel="mixing μ", ylabel="modularity z vs. null model", title="Significance of detected structure")
X.save_fig(fig, d, "z_vs_mixing")
X.write_summary(d, "E1 — Recovery of planted structure", X.md_table(
    agg, ["mu", "louvain_nmi", "metis_ari", "Q", "z", "n_comm", "cut_reduction_vs_block"],
    {c: "{:.3f}" for c in ["mu", "louvain_nmi", "metis_ari", "Q", "z", "n_comm", "cut_reduction_vs_block"]}))
print(open(d / "summary.md").read())

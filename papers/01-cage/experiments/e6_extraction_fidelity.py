"""E6 — How little of the trace does CAGE need? (OQ3: extraction overhead vs. fidelity)

Long MoE run with noisy routing. Capture only every k-th step and compare against the full trace:
hotspot ordering (Kendall tau), top-10 hotspot overlap, community agreement (NMI), and the quality
of the placement computed from the sample, evaluated on the full traffic (1.0 = as good as full).
Expected: placement quality saturates at a few percent of steps when routing is stationary.
Non-stationary routing is I-GAPE's problem (Paper 4).
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, metrics as m, partition as P, placement as PL, workloads as W

d = X.outdir("01-cage", "e6_extraction_fidelity")
steps = 200 if not X.quick() else 60
t = W.moe(affinity=0.6, steps=steps, step_noise=0.3, seed=3)
full = PL.pinned_matrix(t)
s_full = m.strength(full)["total"]
lab_full, _ = m.louvain(full)
G = 8
place_full = P.partition(full, G, "best")
cut_full = P.cut_weight(full + full.T, place_full)
cut_block = P.cut_weight(full + full.T, P.block(t.n_ranks, G))
rows = []
for k in [1, 2, 5, 10, 20, 50, 100]:
    if k > steps:
        continue
    sel = list(range(0, steps, k))
    S = PL.pinned_matrix(t.subset(steps=sel))
    ag = m.rank_agreement(s_full, m.strength(S)["total"])
    lab, _ = m.louvain(S)
    pl = P.partition(S, G, "best")
    cut = P.cut_weight(full + full.T, pl)
    rows.append({"every_k": k, "frac_steps": len(sel) / steps, **ag,
                 "community_nmi": m.nmi(lab, lab_full),
                 "placement_quality": (cut_block - cut) / (cut_block - cut_full) if cut_block > cut_full else 1.0})
X.save_json(d, rows)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
fr = [r["frac_steps"] for r in rows]
for i, (key, lab) in enumerate([("kendall_tau", "hotspot order (Kendall τ)"), ("community_nmi", "communities (NMI)"),
                                ("placement_quality", "placement gain retained")]):
    ax.plot(fr, [r[key] for r in rows], marker="o", color=X.SERIES[i], label=lab)
ax.set_xscale("log"); ax.set(xlabel="fraction of steps captured", ylabel="agreement with full trace",
                             ylim=(0, 1.05), title="Sampling fidelity (MoE, noisy routing)")
ax.legend(loc="lower right")
X.save_fig(fig, d, "fidelity_vs_sampling")
X.write_summary(d, "E6 — Extraction fidelity under step sampling", X.md_table(
    rows, ["every_k", "frac_steps", "kendall_tau", "top10_jaccard", "community_nmi", "placement_quality"],
    {c: "{:.3f}" for c in ["frac_steps", "kendall_tau", "top10_jaccard", "community_nmi", "placement_quality"]}))
print(open(d / "summary.md").read())

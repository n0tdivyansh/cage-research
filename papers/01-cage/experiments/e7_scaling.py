"""E7 — Analysis overhead vs. graph size (evaluation metric 'Analysis Overhead').

Times each pipeline stage on Megatron-style graphs of growing size. Betweenness uses exact Brandes
up to 256 ranks and k=64 pivot sampling above. Swap refinement is dense O(n^2) per iteration —
expect it to dominate at large n; that is a known limitation to state or fix.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, metrics as m, partition as P, workloads as W

d = X.outdir("01-cage", "e7_scaling")
configs = [(8, 2, 2), (8, 4, 4), (8, 4, 8), (8, 8, 8), (8, 8, 16)]
if X.quick():
    configs = configs[:3]
rows = []
for tp, pp, dp in configs:
    t0 = time.perf_counter(); t = W.megatron_3d(tp, pp, dp, steps=1); M = t.matrix(per_step=True)
    r = {"ranks": t.n_ranks, "edges": int((M > 0).sum()), "build_s": time.perf_counter() - t0}
    t0 = time.perf_counter(); m.load_summary(M); m.hotspots(M); r["load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); m.louvain(M); r["louvain_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); m.betweenness(M, k=None if t.n_ranks <= 256 else 64); r["betweenness_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); P.partition(M, 8, "metis", refine=False); r["metis_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); P.partition(M, 8, "best"); r["partition_best_s"] = time.perf_counter() - t0
    rows.append(r)
X.save_json(d, rows)
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for i, key in enumerate(["louvain_s", "betweenness_s", "metis_s", "partition_best_s"]):
    ax.plot([r["ranks"] for r in rows], [r[key] for r in rows], marker="o", color=X.SERIES[i],
            label=key.replace("_s", "").replace("_", " "))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set(xlabel="ranks", ylabel="seconds", title="Analysis time by stage")
ax.legend()
X.save_fig(fig, d, "scaling")
X.write_summary(d, "E7 — Analysis overhead", X.md_table(
    rows, ["ranks", "edges", "load_s", "louvain_s", "betweenness_s", "metis_s", "partition_best_s"],
    {c: "{:.3f}" for c in ["load_s", "louvain_s", "betweenness_s", "metis_s", "partition_best_s"]}) +
    ["", f"Machine-dependent timings; rerun on the target machine before quoting."], scenario=False)
print(open(d / "summary.md").read())

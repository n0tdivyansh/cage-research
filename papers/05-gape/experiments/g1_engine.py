"""G1 — Is an adaptive graph-aware engine worth it once migration costs are paid? (Paper 5)

Same drifting MoE workload as I1 on a 2:1 oversubscribed leaf-spine. Modes:
  block  never plans
  once   CAGE offline: plan once after warm-up (Paper 1 as deployed)
  gape   windowed drift detection + topology-aware re-planning + cost gate
Swept: migration cost per rank moved. Also an ablation: gape without the cost gate and gape
without topology awareness.
Expected: GAPE wins when migration is cheap relative to regime length; the cost gate keeps it
from losing when migration is expensive. The break-even curve is the Paper 5 result.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, placement as PL, workloads as W
from gape import engine as E, topology as TP

d = X.outdir("05-gape", "g1_engine")
spp = 30 if not X.quick() else 15
tr = W.moe_phases(n_phases=4, steps_per_phase=spp, affinity=0.6, seed=0)
mats = [PL.pinned_matrix(tr.subset(steps=[s]), per_step=False) for s in range(tr.n_steps)]
topo = TP.LeafSpine.for_ranks(tr.n_ranks, 8, 2, oversubscription=2)
costs = [0.0, 0.01, 0.1, 1.0, 5.0] if not X.quick() else [0.0, 0.1, 5.0]
variants = {
    "block": dict(mode="block", cfg={}),
    "once": dict(mode="once", cfg={}),
    "gape": dict(mode="gape", cfg={}),
    "gape, no cost gate": dict(mode="gape", cfg={"cost_gate": False}),
    "gape, topology-oblivious": dict(mode="gape", cfg={"topology_aware": False}),
}
rows = []
for c in costs:
    for name, v in variants.items():
        cfg = E.EngineConfig(migration_cost_s=c, horizon_steps=spp // 2, **v["cfg"])
        r = E.run(mats, topo, cfg, v["mode"])
        rows.append({"migration_s_per_rank": c, "variant": name, "comm_s": r["comm_time_s"],
                     "migration_s": r["migration_s"], "total_s": r["total_s"], "replans": r["replans"],
                     "skipped": r["skipped_replans"], "ranks_moved": r["ranks_moved"]})
X.save_json(d, rows)
fig, ax = plt.subplots(figsize=(5.8, 3.6))
for i, name in enumerate(variants):
    r = [x for x in rows if x["variant"] == name]
    base = [x for x in rows if x["variant"] == "block"]
    ax.plot([x["migration_s_per_rank"] for x in r], [x["total_s"] / b["total_s"] for x, b in zip(r, base)],
            marker="o", color=X.SERIES[i], label=name)
ax.set_xscale("symlog", linthresh=1e-2)
ax.set(xlabel="migration cost per rank moved (s)", ylabel="total time relative to block",
       title="Adaptive engine vs. migration cost")
ax.legend(fontsize=8)
X.save_fig(fig, d, "engine_vs_migration_cost")
X.write_summary(d, "G1 — GAPE engine (simulated)", X.md_table(
    rows, ["migration_s_per_rank", "variant", "comm_s", "migration_s", "total_s", "replans", "skipped", "ranks_moved"],
    {"migration_s_per_rank": "{:.3f}", "comm_s": "{:.3f}", "migration_s": "{:.3f}", "total_s": "{:.3f}"}))
print(open(d / "summary.md").read())

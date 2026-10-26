"""I1 — How stable are communication graphs, and when should a placement be recomputed? (Paper 4)

MoE with abrupt routing-regime changes (moe_phases). Part 1: drift metrics step-to-step; do they
fire at the boundaries, and does the permutation-invariant spectral distance miss them?
Part 2: re-planning policies with migration cost: static block, once, periodic, triggered, oracle.
Expected: triggered re-planning approaches the oracle with far fewer re-plans; 'once' decays
after the first regime change; spectral distance under-reacts to relabelling-type drift.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, placement as PL, workloads as W
from gape import temporal as T

d = X.outdir("04-igape", "i1_stability_and_replanning")
spp = 30 if not X.quick() else 15
tr = W.moe_phases(n_phases=4, steps_per_phase=spp, affinity=0.6, seed=0)
mats = [PL.pinned_matrix(tr.subset(steps=[s]), per_step=False) for s in range(tr.n_steps)]
drift = T.drift_series(mats)
bounds = tr.meta["phase_boundaries"]

fig, ax = plt.subplots(figsize=(6.4, 3.4))
steps = np.arange(1, len(mats))
for i, key in enumerate(["rel_l1", "spectral", "partition"]):
    v = np.asarray(drift[key]); v = v / v.max() if v.max() > 0 else v
    ax.plot(steps, v, color=X.SERIES[i], lw=1.5, label=f"{key} (scaled to max)")
for b in bounds:
    ax.axvline(b, color=X.INK2, lw=1, ls="--")
ax.set(xlabel="step", ylabel="drift vs. previous step (scaled)", title="Step-to-step drift; dashed = regime change")
ax.legend(loc="upper right", fontsize=8)
X.save_fig(fig, d, "drift")

def ratio_at_boundaries(v):
    v = np.asarray(v); idx = [b - 1 for b in bounds]
    inside = np.delete(v, idx)
    return float(v[idx].mean() / inside.mean()) if inside.mean() > 0 else float("inf")

detect = {k: ratio_at_boundaries(drift[k]) for k in drift}
cl = PL.Cluster.for_ranks(tr.n_ranks, 8)
rows = []
for mig in [0.0, 0.002, 0.01]:
    for pol, par in [("static_block", None), ("once", None), ("periodic", spp // 3), ("periodic", spp),
                     ("triggered", 0.1), ("triggered", 0.3), ("oracle", None)]:
        r = T.simulate_policy(mats, 8, cl, pol, par, migration_cost_s=mig)
        rows.append({"migration_s_per_rank": mig, "policy": pol, "param": par, "comm_s": r["comm_time_s"],
                     "migration_s": r["migration_s"], "total_s": r["total_s"], "replans": r["replans"],
                     "ranks_moved": r["ranks_moved"]})
X.save_json(d, {"drift": drift, "boundary_signal_ratio": detect, "policies": rows})
lines = ["## Drift signal at regime boundaries (mean drift at boundary / mean elsewhere)", ""]
lines += [f"- {k}: {v:.1f}x" for k, v in detect.items()]
lines += ["", "## Re-planning policies", ""] + X.md_table(
    rows, ["migration_s_per_rank", "policy", "param", "comm_s", "migration_s", "total_s", "replans", "ranks_moved"],
    {"migration_s_per_rank": "{:.3f}", "comm_s": "{:.3f}", "migration_s": "{:.3f}", "total_s": "{:.3f}"})
lines += ["", "oracle re-plans on the step's own matrix (future knowledge) — an upper bound, not a policy."]
X.write_summary(d, "I1 — Stability and re-planning", lines)
print(open(d / "summary.md").read())

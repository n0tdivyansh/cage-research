"""T1 — Does mapping above the host level matter? (Paper 2, T-GAPE)

Leaf-spine cluster, 4 hosts per leaf, oversubscription 1:1, 2:1, 4:1.
Strategies: block, flat (CAGE host partition only), hierarchical (host partition, then hosts onto
leaves), top_down (ranks onto leaves, then hosts). Metric: estimated communication time relative
to block, and share of bytes crossing leaves.
Expected: at 1:1 the leaf level barely matters; with oversubscription, leaf-aware mapping starts
to pay. The size of that gap for ML traffic is the T-GAPE result.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, placement as PL, workloads as W
from gape import topology as TP

d = X.outdir("02-tgape", "t1_topology_mapping")
cases = {
    "MoE aff 0.6, 128 ranks (pinned)": PL.pinned_matrix(W.moe(world=128, affinity=0.6)),
    "Stencil 3D 8x4x4 (fixed)": W.stencil((8, 4, 4)).matrix(per_step=True),
    "Megatron 8·4·4, scrambled (fixed)": W.megatron_3d(8, 4, 4).relabel(
        np.random.default_rng(1).permutation(128)).matrix(per_step=True),
}
if X.quick():
    cases.pop("Stencil 3D 8x4x4 (fixed)")
oversubs = [1, 2, 4]
rows = []
for name, M in cases.items():
    for ov in oversubs:
        topo = TP.LeafSpine.for_ranks(M.shape[0], 8, 4, oversubscription=ov)
        base = None
        for s in TP.STRATEGIES:
            host, leaf = TP.map_ranks(M, topo, s)
            r = TP.evaluate(M, host, leaf, topo)
            if s == "block":
                base = r["est_comm_time_s"]
            rows.append({"workload": name, "oversub": ov, "strategy": s,
                         "cross_leaf_frac": r["cross_leaf_frac"], "cross_host_bytes": r["cross_host_bytes"],
                         "est_time_s": r["est_comm_time_s"], "time_vs_block": r["est_comm_time_s"] / base,
                         "uplink_bound": r["max_uplink_time_s"] > r["max_nic_time_s"]})
X.save_json(d, rows)
fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, cases):
    for i, s in enumerate(TP.STRATEGIES):
        r = [x for x in rows if x["workload"] == name and x["strategy"] == s]
        ax.plot([x["oversub"] for x in r], [x["time_vs_block"] for x in r], marker="o", color=X.SERIES[i], label=s)
    ax.set(title=name, xlabel="spine oversubscription", xticks=oversubs)
axes[0].set_ylabel("est. comm time relative to block")
axes[0].legend()
X.save_fig(fig, d, "time_vs_oversubscription")
X.write_summary(d, "T1 — Topology-aware mapping", X.md_table(
    rows, ["workload", "oversub", "strategy", "cross_leaf_frac", "est_time_s", "time_vs_block", "uplink_bound"],
    {"cross_leaf_frac": "{:.3f}", "est_time_s": "{:.4f}", "time_vs_block": "{:.3f}"}))
print(open(d / "summary.md").read())

"""T1 — Does mapping above the host level matter? (Paper 2, T-GAPE)

Leaf-spine cluster, 4 hosts per leaf, oversubscription 1:1, 2:1, 4:1.
Strategies: block, flat (CAGE host partition only), hierarchical (host partition, then hosts onto
leaves), top_down (ranks onto leaves, then hosts). Metric: estimated communication time relative
to block, and share of bytes crossing leaves.
Expected: at 1:1 the leaf level barely matters; with oversubscription, leaf-aware mapping starts
to pay. The size of that gap for ML traffic is the T-GAPE result.

NCCL regime (rule E3, added before running): MoE and Megatron are GPU jobs whose collectives
NCCL re-rings for every placement, so they are also evaluated with placement.nccl_matrix, rings
ordered by (leaf, host) as NCCL derives them from topology. Each strategy takes the better of two
objectives (all traffic in rank order; rank-pinned traffic + collective clique), as recommend_nccl.
Expected: the ordering of strategies holds (hierarchical <= flat under oversubscription), with a
smaller gap for Megatron, because re-ringing already removes part of the cross-leaf ring traffic.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, placement as PL, workloads as W
from gape import topology as TP

d = X.outdir("02-tgape", "t1_topology_mapping")
moe_t = W.moe(world=128, affinity=0.6)
meg_t = W.megatron_3d(8, 4, 4).relabel(np.random.default_rng(1).permutation(128))
cases = {
    "MoE aff 0.6, 128 ranks (pinned)": PL.pinned_matrix(moe_t),
    "Stencil 3D 8x4x4 (fixed)": W.stencil((8, 4, 4)).matrix(per_step=True),
    "Megatron 8·4·4, scrambled (fixed)": meg_t.matrix(per_step=True),
}
nccl_cases = {"MoE aff 0.6, 128 ranks (nccl)": moe_t, "Megatron 8·4·4, scrambled (nccl)": meg_t}
if X.quick():
    cases.pop("Stencil 3D 8x4x4 (fixed)")
oversubs = [1, 2, 4]
rows = []


def add_row(name, ov, s, r, base):
    rows.append({"workload": name, "oversub": ov, "strategy": s,
                 "cross_leaf_frac": r["cross_leaf_frac"], "cross_host_bytes": r["cross_host_bytes"],
                 "est_time_s": r["est_comm_time_s"], "time_vs_block": r["est_comm_time_s"] / base,
                 "max_uplink_ms": r["max_uplink_time_s"] * 1e3, "max_nic_ms": r["max_nic_time_s"] * 1e3,
                 "uplink_bound": r["max_uplink_time_s"] > r["max_nic_time_s"]})


for name, M in cases.items():
    for ov in oversubs:
        topo = TP.LeafSpine.for_ranks(M.shape[0], 8, 4, oversubscription=ov)
        base = None
        for s in TP.STRATEGIES:
            host, leaf = TP.map_ranks(M, topo, s)
            r = TP.evaluate(M, host, leaf, topo)
            if s == "block":
                base = r["est_comm_time_s"]
            add_row(name, ov, s, r, base)


def nccl_eval(t, host, leaf, topo):
    # nccl_matrix orders ring members by the key it is given: leaf first, then host
    return TP.evaluate(PL.nccl_matrix(t, leaf[host] * topo.n_hosts + host), host, leaf, topo)


for name, t in nccl_cases.items():
    colls = {c["op"] for c in t.meta["collectives"]}
    objectives = [t.matrix(per_step=True),
                  t.matrix(per_step=True, ops=[o for o in t.op_names if o not in colls])
                  + PL.collective_clique_matrix(t)]
    for ov in oversubs:
        topo = TP.LeafSpine.for_ranks(t.n_ranks, 8, 4, oversubscription=ov)
        base = None
        for s in TP.STRATEGIES:
            res = [nccl_eval(t, *TP.map_ranks(O, topo, s), topo)
                   for O in (objectives[:1] if s == "block" else objectives)]
            r = min(res, key=lambda x: x["est_comm_time_s"])
            if s == "block":
                base = r["est_comm_time_s"]
            add_row(name, ov, s, r, base)
X.save_json(d, rows)
names = list(dict.fromkeys(r["workload"] for r in rows))
fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, names):
    for i, s in enumerate(TP.STRATEGIES):
        r = [x for x in rows if x["workload"] == name and x["strategy"] == s]
        ax.plot([x["oversub"] for x in r], [x["time_vs_block"] for x in r], marker="o", color=X.SERIES[i], label=s)
    ax.set(title=name, xlabel="spine oversubscription", xticks=oversubs)
axes[0].set_ylabel("est. comm time relative to block")
axes[0].legend()
X.save_fig(fig, d, "time_vs_oversubscription")
X.write_summary(d, "T1 — Topology-aware mapping", X.md_table(
    rows, ["workload", "oversub", "strategy", "cross_leaf_frac", "est_time_s", "time_vs_block", "max_uplink_ms",
     "max_nic_ms", "uplink_bound"],
    {"cross_leaf_frac": "{:.3f}", "est_time_s": "{:.4f}", "time_vs_block": "{:.3f}", "max_uplink_ms": "{:.3f}",
     "max_nic_ms": "{:.3f}"}))
print(open(d / "summary.md").read())

"""E4 — When does communication-aware placement beat the default? (Claim 4, H3 — the headline)

Baseline is BLOCK placement (SLURM block / torchrun default), not round-robin.
Two regimes, reported separately:
  A  'fixed' : all traffic, collectives expanded in rank order and held fixed when ranks move
               (valid for MPI codes whose algorithm order follows ranks)
  B  'nccl'  : rank-pinned traffic as recorded + every library collective re-expanded as a ring
               ordered by host for each candidate placement (NCCL re-derives ring order).
Regime B is the defensible headline for GPU jobs. CAGE placement in B = placement.recommend_nccl
(candidates from two objectives, evaluated under the NCCL model, block included, so never worse).
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, partition as P, placement as PL, workloads as W

d = X.outdir("01-cage", "e4_placement_benefit")
G = 8


def run(name, t):
    out = []
    cl = PL.Cluster.for_ranks(t.n_ranks, G)
    base = {"block": P.block(t.n_ranks, G), "round_robin": P.round_robin(t.n_ranks, G),
            "random": P.random_placement(t.n_ranks, G, 0)}
    for regime in ("fixed", "nccl"):
        if regime == "fixed":
            M = t.matrix(per_step=True)
            pl = dict(base, cage=P.remap_to_reference(P.partition(M, G, "best"), base["block"]))
            res = PL.compare(M, pl, cl, t.matrix("msgs", per_step=True))
        else:
            pl = dict(base, cage=PL.recommend_nccl(t, G)[0])
            res = {k: PL.evaluate(PL.nccl_matrix(t, h), h, cl) for k, h in pl.items()}
            ref = res["block"]
            for k, r in res.items():
                r["cross_bytes_reduction_vs_block"] = 1 - r["cross_host_bytes"] / ref["cross_host_bytes"] if ref["cross_host_bytes"] else 0.0
                r["est_time_reduction_vs_block"] = 1 - r["est_comm_time_s"] / ref["est_comm_time_s"] if ref["est_comm_time_s"] else 0.0
                r["ranks_moved_vs_block"] = int((pl[k] != pl["block"]).sum())
        for k, r in res.items():
            out.append({"workload": name, "regime": regime, "placement": k,
                        "cross_host_frac": r["cross_host_frac"],
                        "cross_reduction_vs_block": r["cross_bytes_reduction_vs_block"],
                        "time_reduction_vs_block": r["est_time_reduction_vs_block"],
                        "ranks_moved": r["ranks_moved_vs_block"]})
    return out


rows = []
cases = {
    "DP ring": W.dp_ring(64),
    "DP hier. leader": W.dp_hier(64),
    "Megatron 8·4·2": W.megatron_3d(8, 4, 2),
    "Megatron, scrambled ranks": W.megatron_3d(8, 4, 2).relabel(np.random.default_rng(0).permutation(64)),
    "Stencil 2D 8x8": W.stencil((8, 8)),
    "Stencil 3D 4x4x4": W.stencil((4, 4, 4)),
    "MoE aff 0": W.moe(affinity=0.0),
    "MoE aff 0.6": W.moe(affinity=0.6),
}
for name, t in cases.items():
    rows += run(name, t)

aff_levels = [0.0, 0.2, 0.4, 0.6, 0.8] if not X.quick() else [0.0, 0.6]
sweep = []
for a in aff_levels:
    for layout in ("interleaved", "block"):
        t = W.moe(affinity=a, group_layout=layout)
        r = [x for x in run(f"MoE aff {a} {layout}", t) if x["regime"] == "nccl" and x["placement"] == "cage"][0]
        sweep.append({"affinity": a, "layout": layout, **r})
X.save_json(d, {"cases": rows, "moe_affinity_sweep": sweep})

pinned = [r for r in rows if r["regime"] == "nccl"]
names = list(dict.fromkeys(r["workload"] for r in pinned))
fig, ax = plt.subplots(figsize=(7, 3.8))
y = np.arange(len(names))
for i, plc in enumerate(["cage", "round_robin"]):
    vals = [next(r["cross_reduction_vs_block"] for r in pinned if r["workload"] == n and r["placement"] == plc) for n in names]
    off = 0.2 if i == 0 else -0.2
    ax.barh(y + off, np.maximum(vals, -1.0), 0.38, color=X.SERIES[i],
            label={"cage": "CAGE placement", "round_robin": "round-robin"}[plc])
    for yy, v in zip(y, vals):
        if v < -1.0:
            ax.text(-0.98, yy + off, f"{v:.1f} (clipped)", va="center", fontsize=7, color="#fcfcfb")
ax.axvline(0, color=X.INK2, lw=1)
ax.set_yticks(y, names); ax.invert_yaxis()
ax.set(xlabel="cross-host bytes reduction vs. block (NCCL re-ringing model)",
       title="Placement benefit over the realistic baseline")
ax.set_xlim(max(-1.0, ax.get_xlim()[0]), 1)
ax.legend(loc="lower left")
X.save_fig(fig, d, "placement_vs_block")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for i, layout in enumerate(("interleaved", "block")):
    s = [x for x in sweep if x["layout"] == layout]
    ax.plot([x["affinity"] for x in s], [x["cross_reduction_vs_block"] for x in s], marker="o",
            color=X.SERIES[i], label=f"data groups {layout} across ranks")
ax.set(xlabel="routing affinity (0 = uniformly shuffled data)", ylabel="cross-host reduction vs. block",
       title="MoE: benefit depends on routing locality")
ax.legend()
X.save_fig(fig, d, "moe_affinity_sweep")
lines = ["## Regime B (NCCL model: collectives re-ringed per placement — defensible for GPU jobs)", ""] + X.md_table(
    [r for r in pinned if r["placement"] in ("cage", "round_robin")],
    ["workload", "placement", "cross_host_frac", "cross_reduction_vs_block", "time_reduction_vs_block", "ranks_moved"],
    {c: "{:.3f}" for c in ["cross_host_frac", "cross_reduction_vs_block", "time_reduction_vs_block"]})
lines += ["", "## Regime A (all traffic, collectives fixed in rank order)", ""] + X.md_table(
    [r for r in rows if r["regime"] == "fixed" and r["placement"] in ("cage", "round_robin")],
    ["workload", "placement", "cross_host_frac", "cross_reduction_vs_block", "time_reduction_vs_block", "ranks_moved"],
    {c: "{:.3f}" for c in ["cross_host_frac", "cross_reduction_vs_block", "time_reduction_vs_block"]})
lines += ["", "## MoE affinity sweep (NCCL model, CAGE vs block)", ""] + X.md_table(
    sweep, ["affinity", "layout", "cross_reduction_vs_block", "time_reduction_vs_block", "ranks_moved"],
    {c: "{:.3f}" for c in ["affinity", "cross_reduction_vs_block", "time_reduction_vs_block"]})
X.write_summary(d, "E4 — Placement benefit vs. block placement", lines)
print(open(d / "summary.md").read())

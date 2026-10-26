"""Single-trace analysis report (what `cage analyze` produces)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics as m
from . import partition as P
from . import placement as PL
from .trace import CommTrace, _jsonable


def analyze(trace: CommTrace, gpus_per_node: int, out: str | Path, n_null: int = 20,
            current_placement: np.ndarray | None = None, betweenness: bool = False,
            plots: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    M = trace.matrix(per_step=True)
    summary = m.summarize(M, n_null=n_null, with_betweenness=betweenness)
    n = trace.n_ranks
    cur = current_placement if current_placement is not None else P.block(n, gpus_per_node)
    cl = PL.Cluster.for_ranks(n, gpus_per_node)
    placement = {}
    rec = P.remap_to_reference(P.partition(M, gpus_per_node, "best"), cur)
    res = PL.compare(M, {"current": cur, "cage": rec}, cl, reference="current")
    placement["fixed"] = {"current": res["current"], "cage": res["cage"], "moves": _moves(cur, rec)}
    if trace.meta.get("collectives"):
        rec = P.remap_to_reference(PL.recommend_nccl(trace, gpus_per_node)[0], cur)
        a = PL.evaluate(PL.nccl_matrix(trace, cur), cur, cl)
        b = PL.evaluate(PL.nccl_matrix(trace, rec), rec, cl)
        b["cross_bytes_reduction_vs_current"] = 1 - b["cross_host_bytes"] / a["cross_host_bytes"] if a["cross_host_bytes"] else 0.0
        b["est_time_reduction_vs_current"] = 1 - b["est_comm_time_s"] / a["est_comm_time_s"] if a["est_comm_time_s"] else 0.0
        placement["nccl"] = {"current": a, "cage": b, "moves": _moves(cur, rec)}
    report = {"trace": {"n_ranks": n, "n_events": len(trace), "n_steps": trace.n_steps,
                        "ops": trace.op_names, "source": trace.meta.get("source", trace.meta.get("name")),
                        "assumptions": trace.meta.get("assumptions", [])},
              "analysis": summary, "placement": placement}
    (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
    (out / "report.md").write_text(_markdown(report))
    if plots:
        from . import viz
        viz.traffic_heatmap(M, out / "traffic_matrix.png")
        viz.strength_bars(M, out / "strength.png")
        if n <= 256:
            viz.graph_layout(M, np.asarray(summary["community_labels"]), out / "graph.png")
    return report


def _moves(cur, rec) -> list[dict]:
    return [{"rank": int(r), "from_host": int(cur[r]), "to_host": int(rec[r])}
            for r in np.nonzero(rec != cur)[0]]


def _verdict(c: dict) -> str:
    if c["n_null"] < 19:
        return (f"inconclusive: {c['n_null']} null samples cannot reach p ≤ 0.05; "
                "rerun with --n-null 19 or more")
    if c["z"] > 3 and c["p_empirical"] <= 0.05:
        return "significant structure"
    return "NO structure beyond what the degree sequence forces"


def _markdown(r: dict) -> str:
    a, t = r["analysis"], r["trace"]
    ld, c = a["load"], a["communities"]
    L = [f"# CAGE report — {t['source']}", "",
         f"{t['n_ranks']} ranks, {t['n_events']} events, {t['n_steps']} steps, ops: {', '.join(t['ops'])}", ""]
    if t["assumptions"]:
        L += ["**Assumptions carried by this trace:**"] + [f"- {x}" for x in t["assumptions"]] + [""]
    L += ["## Load", "",
          f"- traffic per step: {ld['total_bytes']:.4g} B; strength sum {ld['strength_sum']:.4g} B (= 2 × traffic)",
          f"- Gini {ld['gini']:.3f}, CV {ld['cv']:.3f}, busiest 20% of ranks carry {ld['top20_share']:.1%}",
          f"- max / median strength {ld['max_over_median']:.2f}; isolated ranks {ld['isolated_ranks']}", "",
          "## Hotspots (robust z ≥ 3.5)", ""]
    L += [f"- rank {h['rank']}: {h['strength']:.4g} B ({h['x_median']:.1f}× median, z={h['robust_z']:.1f}); "
          f"top partners {[p for p, _ in h['top_partners']]}" for h in a["hotspots"]] or ["- none"]
    L += ["", "## Community structure", "",
          f"- Louvain Q = {c['Q']:.3f} with {c['n_communities']} communities",
          f"- null model Q = {c['null_mean']:.3f} ± {c['null_std']:.3f}  →  z = {c['z']:.1f}, p = {c['p_empirical']:.3f}",
          f"- verdict: {_verdict(c)}",
          "", "## Placement (current vs CAGE recommendation)", ""]
    for regime, p in r["placement"].items():
        cu, ca = p["current"], p["cage"]
        L += [f"**{regime}** traffic: cross-host {cu['cross_host_frac']:.1%} → {ca['cross_host_frac']:.1%} "
              f"({ca['cross_bytes_reduction_vs_current']:.1%} fewer bytes), est. comm time "
              f"{ca['est_time_reduction_vs_current']:.1%} lower (model), {len(p['moves'])} ranks move", ""]
    L += ["`fixed` holds collectives in their recorded rank order (MPI-like). `nccl` re-expands "
          "library collectives as host-ordered rings for each placement — trust it for NCCL jobs.", ""]
    return "\n".join(L)

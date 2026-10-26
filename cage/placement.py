"""Cluster model and placement evaluation.

The cost model is a BANDWIDTH-BOUND ESTIMATE, not a measurement:
  per host h: inter_time = max(egress_h, ingress_h) / (nic_bw * nic_factor_h)
              intra_time = intra_bytes_h / intra_bw
              host_time  = inter_time + intra_time + alpha * messages_h
  estimated communication time per step = max over hosts.
It ignores dependencies, overlap with compute, congestion inside the fabric, and the fact that
NCCL re-derives ring order from topology. Its job is to rank placements, not to predict seconds.
Anything reported from it is a model output (epistemic rule E1).

Important scope limit (state it in the paper): moving ranks while keeping the communication
pattern fixed is valid for patterns pinned to rank identity — MPI point-to-point, pipeline
stages, expert-parallel all-to-all, TP groups. For NCCL collectives the library rebuilds the ring
for the new placement, so placement 'gains' on pure collectives overstate what is achievable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PINNED_KEYWORDS = ("alltoall", "send", "recv", "p2p", "halo", "pp_")


def is_pinned(op: str) -> bool:
    """True if the traffic of this op is fixed by rank identity (moving the rank moves the
    traffic). False for collectives whose algorithm/ring order the library re-derives from the
    placement (NCCL/most MPIs), where a fixed rank-order expansion misstates the post-move cost."""
    return any(k in op.lower() for k in PINNED_KEYWORDS)


def pinned_matrix(trace, per_step: bool = True) -> np.ndarray:
    ops = [o for o in trace.op_names if is_pinned(o)]
    return trace.matrix(ops=ops, per_step=per_step) if ops else np.zeros((trace.n_ranks,) * 2)


def _coll_scale(trace, per_step: bool) -> float:
    total = trace.meta.get("collectives_steps", trace.n_steps) or 1
    present = len(np.unique(trace.step)) if len(trace) else 0
    return 1.0 / total if per_step else present / total


def nccl_matrix(trace, host: np.ndarray, per_step: bool = True) -> np.ndarray:
    """Traffic under a placement when the communication library re-derives ring order from the
    placement (NCCL behaviour): rank-pinned events as recorded + every recorded collective
    re-expanded as a ring ordered by host. Without recorded collectives this equals the full
    trace matrix (nothing is re-routable)."""
    from . import collectives as C
    from .trace import TraceBuilder
    colls = trace.meta.get("collectives", [])
    if not colls:
        return trace.matrix(per_step=per_step)
    rer = {c["op"] for c in colls}
    keep = [o for o in trace.op_names if o not in rer]
    M = trace.matrix(ops=keep, per_step=False) if keep else np.zeros((trace.n_ranks,) * 2)
    if per_step:
        n = len(np.unique(trace.step)) if len(trace) else 1
        M = M / max(n, 1)
    host = np.asarray(host)
    scale = _coll_scale(trace, per_step)
    b = TraceBuilder(trace.n_ranks)
    for c in colls:
        mem = sorted(c["members"], key=lambda r: (host[r], c["members"].index(r)))
        size = c["size"] * c["count"] * scale
        kind = c.get("kind", "allreduce")
        if kind in ("allreduce",) or "allreduce" in kind:
            C.ring_allreduce(b, mem, size)
        elif kind in ("allgather", "reduce_scatter"):
            C.ring_allgather(b, mem, size)
        else:
            C.chain_broadcast(b, mem, size)
    return M + b.build(n_ranks=trace.n_ranks).matrix()


def collective_clique_matrix(trace, per_step: bool = True) -> np.ndarray:
    """Clique expansion of recorded collectives, used only as a partitioning OBJECTIVE.
    A host-ordered ring spanning h hosts costs ~h ring edges; the clique weight 8*e/p^2 per
    pair makes a 2-way even split cost the same as the ring (2 crossing edges of e bytes)."""
    M = np.zeros((trace.n_ranks,) * 2)
    scale = _coll_scale(trace, per_step)
    for c in trace.meta.get("collectives", []):
        m = np.asarray(c["members"]); p = len(m)
        if p < 2:
            continue
        frac = 2 * (p - 1) / p if "allreduce" in c.get("kind", "allreduce") else (p - 1) / p
        e = frac * c["size"] * c["count"] * scale
        M[np.ix_(m, m)] += 4 * e / p ** 2          # 4 each direction -> 8 symmetric
    np.fill_diagonal(M, 0)
    return M


def recommend_nccl(trace, capacity: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """CAGE recommendation under the NCCL model: partition candidates from (a) the rank-order
    matrix and (b) pinned traffic + collective clique objective; evaluate every candidate and
    block under nccl_matrix; return the one with the fewest cross-host bytes."""
    from . import partition as P
    n = trace.n_ranks
    pinned = trace.matrix(per_step=True, ops=[o for o in trace.op_names if o not in
                                              {c["op"] for c in trace.meta.get("collectives", [])}])
    cands = {"block": P.block(n, capacity),
             "fixed_objective": P.partition(trace.matrix(per_step=True), capacity, "best", seed),
             "clique_objective": P.partition(pinned + collective_clique_matrix(trace), capacity, "best", seed)}
    scores = {}
    for k, h in cands.items():
        M = nccl_matrix(trace, h)
        scores[k] = float((M * (h[:, None] != h[None, :])).sum())
    best = min(scores, key=scores.get)
    return P.remap_to_reference(cands[best], cands["block"]), {"chosen": best, "cross_bytes": scores}


@dataclass
class Cluster:
    n_hosts: int
    gpus_per_host: int = 8
    intra_bw: float = 300e9      # bytes/s per host-internal transfer budget (NVLink-class)
    nic_bw: float = 50e9         # bytes/s per host, each direction (400 Gb/s NIC)
    alpha: float = 5e-6          # per-message latency, seconds
    nic_factor: np.ndarray = field(default=None)  # per-host bandwidth multiplier (fault injection)

    def __post_init__(self):
        if self.nic_factor is None:
            self.nic_factor = np.ones(self.n_hosts)

    @classmethod
    def for_ranks(cls, n_ranks: int, gpus_per_host: int = 8, **kw) -> "Cluster":
        return cls(-(-n_ranks // gpus_per_host), gpus_per_host, **kw)


def evaluate(M: np.ndarray, host: np.ndarray, cluster: Cluster,
             msgs: np.ndarray | None = None) -> dict:
    """M: per-step bytes matrix (directed). host: host index per rank."""
    host = np.asarray(host)
    H = cluster.n_hosts
    cross = host[:, None] != host[None, :]
    inter = M * cross
    egress = np.bincount(host, weights=inter.sum(1), minlength=H)
    ingress = np.bincount(host, weights=inter.sum(0), minlength=H)
    intra = np.bincount(host, weights=(M * ~cross).sum(1), minlength=H)
    n_msgs = (np.bincount(host, weights=msgs.sum(1), minlength=H) if msgs is not None
              else np.zeros(H))
    inter_t = np.maximum(egress, ingress) / (cluster.nic_bw * cluster.nic_factor)
    intra_t = intra / cluster.intra_bw
    host_t = inter_t + intra_t + cluster.alpha * n_msgs
    total = M.sum()
    return {"cross_host_bytes": float(inter.sum()),
            "cross_host_frac": float(inter.sum() / total) if total > 0 else 0.0,
            "max_nic_bytes": float(np.maximum(egress, ingress).max()),
            "est_comm_time_s": float(host_t.max()),
            "bottleneck_host": int(np.argmax(host_t)),
            "host_time_s": host_t.tolist()}


def compare(M: np.ndarray, placements: dict[str, np.ndarray], cluster: Cluster,
            msgs: np.ndarray | None = None, reference: str = "block") -> dict:
    res = {name: evaluate(M, lab, cluster, msgs) for name, lab in placements.items()}
    ref = res[reference]
    for name, r in res.items():
        r["cross_bytes_reduction_vs_" + reference] = (
            1 - r["cross_host_bytes"] / ref["cross_host_bytes"] if ref["cross_host_bytes"] > 0 else 0.0)
        r["est_time_reduction_vs_" + reference] = (
            1 - r["est_comm_time_s"] / ref["est_comm_time_s"] if ref["est_comm_time_s"] > 0 else 0.0)
        r["ranks_moved_vs_" + reference] = int((placements[name] != placements[reference]).sum())
    return res


def time_weighted(M: np.ndarray, host: np.ndarray, cluster: Cluster) -> np.ndarray:
    """Convert a bytes matrix into an estimated transfer-time matrix under a placement.
    With real traces, use measured per-event durations instead (trace.matrix('duration'))."""
    host = np.asarray(host)
    cross = host[:, None] != host[None, :]
    nic = cluster.nic_bw * cluster.nic_factor
    eff = np.where(cross, np.minimum(nic[host][:, None], nic[host][None, :]), cluster.intra_bw)
    return M / eff

"""Shared collective expansion for parsers."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import collectives as C
from ..trace import TraceBuilder

# canonical op -> (expander, how the recorded byte count maps to the expander's `size`)
#   'buffer'  : recorded bytes is the full buffer S (allreduce, broadcast, reduce)
#   'per_rank': recorded bytes is one rank's contribution; full buffer = bytes * p (allgather,
#               reduce_scatter)
CANON = {
    "allreduce": "allreduce", "all_reduce": "allreduce",
    "allgather": "allgather", "all_gather": "allgather", "_allgather_base": "allgather",
    "reducescatter": "reduce_scatter", "reduce_scatter": "reduce_scatter",
    "reduce_scatter_block": "reduce_scatter",
    "broadcast": "broadcast", "bcast": "broadcast",
    "reduce": "reduce",
    "alltoall": "alltoall", "all_to_all": "alltoall",
    "barrier": "barrier",
}


def canon(op: str) -> str:
    return CANON.get(op.strip().lower().replace("ncclkernel_", "").replace("mpi_", ""), op.lower())


def expand(b: TraceBuilder, op: str, members: Sequence[int], nbytes: float, step: int = 0,
           algo: str = "ring", root: int | None = None, only_src: int | None = None,
           duration: float | None = None) -> None:
    """Expand one collective call. `members` are global ranks in communicator order.
    `only_src`: keep only edges sent by this rank (used when every rank's file records the
    same collective, to avoid counting it p times)."""
    op = canon(op)
    p = len(members)
    if p < 2 or op == "barrier" or nbytes <= 0:
        return
    tmp = TraceBuilder(b.n_ranks)
    m = list(members)
    if op == "allreduce":
        (C.tree_allreduce if algo == "tree" else C.ring_allreduce)(tmp, m, nbytes, step, op)
    elif op == "allgather":
        C.ring_allgather(tmp, m, nbytes * p, step, op)
    elif op == "reduce_scatter":
        C.ring_reduce_scatter(tmp, m, nbytes * p, step, op)
    elif op in ("broadcast", "reduce"):
        r = m.index(root) if root in m else 0
        order = m[r:] + m[:r]
        (C.chain_broadcast if op == "broadcast" else C.chain_reduce)(tmp, order, nbytes, step, op)
    elif op == "alltoall":
        M = np.full((p, p), nbytes / p)
        np.fill_diagonal(M, 0)
        C.alltoall(tmp, m, M, step, op)
    else:
        return
    if op in ("allreduce", "allgather", "reduce_scatter", "broadcast", "reduce") and \
            (only_src is None or only_src == m[0]):
        full = nbytes * p if op in ("allgather", "reduce_scatter") else nbytes
        b.record_collective(op, m, full, kind=op)
    t = tmp.build(n_ranks=max(b.n_ranks, max(m) + 1))
    keep = np.ones(len(t), bool) if only_src is None else (t.src == only_src)
    if not keep.any():
        return
    for name_id in np.unique(t.op[keep]):
        k = keep & (t.op == name_id)
        dur = None
        if duration is not None:
            dur = np.full(int(k.sum()), duration)
        b.add_arrays(t.src[k], t.dst[k], t.nbytes[k], t.msgs[k], step, t.op_names[name_id], dur)

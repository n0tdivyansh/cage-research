"""Expansion of collective operations into point-to-point edges.

Resolution of open question OQ1 ("how to represent AllReduce in a pairwise graph"):
a collective is expanded into the point-to-point transfers of the ALGORITHM THAT RAN, because that
is the traffic the network actually carries. The semantic intent (which ranks took part) is kept in
the op name and can be recovered. A hypergraph view is available through `participants()`
for analyses that need it, but every graph algorithm in CAGE runs on the expanded pairwise graph.

Byte formulas are the standard bandwidth-optimal ones (Thakur et al. 2005; Patarasuk & Yuan 2009):
  ring allreduce, S bytes, p ranks:   each ring edge carries 2(p-1)/p * S
  ring allgather / reduce-scatter:    each ring edge carries (p-1)/p * S   (S = full buffer)
  binary-tree allreduce:              each tree edge carries S up and S down
  chain broadcast / reduce:           each chain edge carries S
  direct all-to-all:                  i -> j carries the routed bytes M[i, j]

Limitations (state them in the paper):
- NCCL selects ring vs tree vs NVLS/CollNet at runtime by message size and topology, and builds
  ring order from the detected topology. A trace that does not record which algorithm ran must
  assume one; record the assumption in trace.meta['collective_algo'].
- Double-binary-tree (NCCL tree) is modelled as a single binary tree with full S per edge. This
  overstates per-edge bytes by up to 2x relative to NCCL's split across two trees.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .trace import TraceBuilder


def ring_allreduce(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                   op: str = "allreduce", repeat: int = 1) -> None:
    """`repeat` folds `repeat` identical calls into one event (bytes and message counts scale)."""
    p = len(group)
    if p < 2:
        return
    g = np.asarray(group)
    per_edge = 2.0 * (p - 1) / p * size * repeat
    b.add_arrays(g, np.roll(g, -1), np.full(p, per_edge), 2 * (p - 1) * repeat, step, op)


def ring_allgather(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                   op: str = "allgather") -> None:
    p = len(group)
    if p < 2:
        return
    g = np.asarray(group)
    b.add_arrays(g, np.roll(g, -1), np.full(p, (p - 1) / p * size), p - 1, step, op)


def ring_reduce_scatter(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                        op: str = "reduce_scatter") -> None:
    ring_allgather(b, group, size, step, op)


def tree_allreduce(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                   op: str = "allreduce_tree") -> None:
    g = list(group)
    for k in range(1, len(g)):
        parent = g[(k - 1) // 2]
        b.add(g[k], parent, size, 1, step, op)   # reduce up
        b.add(parent, g[k], size, 1, step, op)   # broadcast down


def chain_broadcast(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                    op: str = "broadcast") -> None:
    """Pipelined chain broadcast from group[0]."""
    g = np.asarray(group)
    if len(g) < 2:
        return
    b.add_arrays(g[:-1], g[1:], np.full(len(g) - 1, size), 1, step, op)


def chain_reduce(b: TraceBuilder, group: Sequence[int], size: float, step: int = 0,
                 op: str = "reduce") -> None:
    """Pipelined chain reduce towards group[0]."""
    g = np.asarray(group)
    if len(g) < 2:
        return
    b.add_arrays(g[1:], g[:-1], np.full(len(g) - 1, size), 1, step, op)


def alltoall(b: TraceBuilder, group: Sequence[int], bytes_matrix: np.ndarray, step: int = 0,
             op: str = "alltoall") -> None:
    g = np.asarray(group)
    M = np.asarray(bytes_matrix, dtype=float)
    ii, jj = np.nonzero(M)
    b.add_arrays(g[ii], g[jj], M[ii, jj], 1, step, op)


def hier_leader_allreduce(b: TraceBuilder, hosts: Sequence[Sequence[int]], size: float,
                          step: int = 0, op: str = "allreduce_hier_leader") -> None:
    """Horovod-style hierarchical allreduce: intra-host reduce to a local leader, ring
    allreduce among leaders, intra-host broadcast from the leader. Leaders are hosts[h][0]."""
    leaders = [h[0] for h in hosts if len(h)]
    for h in hosts:
        chain_reduce(b, h, size, step, op)
    ring_allreduce(b, leaders, size, step, op)
    for h in hosts:
        chain_broadcast(b, h, size, step, op)


def hier_rail_allreduce(b: TraceBuilder, hosts: Sequence[Sequence[int]], size: float,
                        step: int = 0, op: str = "allreduce_hier_rail") -> None:
    """Rail-optimised hierarchical allreduce: intra-host ring reduce-scatter, inter-host ring
    allreduce of each 1/g shard among ranks with the same local index, intra-host ring allgather.
    Inter-host load is spread over every GPU instead of one leader."""
    g = min(len(h) for h in hosts)
    for h in hosts:
        ring_reduce_scatter(b, h, size, step, op)
    for local in range(g):
        ring_allreduce(b, [h[local] for h in hosts], size / g, step, op)
    for h in hosts:
        ring_allgather(b, h, size, step, op)


def participants(group: Sequence[int]) -> frozenset[int]:
    """Hyperedge view of a collective (semantic intent, not traffic)."""
    return frozenset(int(x) for x in group)

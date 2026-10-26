"""Paper 3 — C-GAPE: data-access graphs for cache and data placement.

Correction to the original roadmap: the briefing says the data-access graph is "implicit in the
communication graph CAGE extracts". It is not. A communication trace records who sent bytes to
whom, not WHICH data object was moved. C-GAPE needs its own extraction source that records object
identity: embedding-table row/shard IDs, parameter-server keys, dataloader shard reads,
KV-cache blocks. This module therefore has its own trace type (AccessTrace).

Model
  bipartite graph  ranks x objects, weight = bytes read
  each object has a home host (where it lives); a read from another host is a remote fetch
  each host has a cache of `cache_bytes`; cached objects are read locally

Policies compared (all respect the same per-host cache budget)
  rr_home            objects homed round-robin across hosts, no cache
  rr_home+lfu        + each host caches its own most-read objects (graph-oblivious, STRONG baseline)
  graph_home+lfu     objects homed on the host that reads them most, subject to an equal
                     per-host home-storage budget (greedy on the bipartite host x object graph),
                     then the same LFU cache
  graph_home+lfu+rep  + replicate objects read by many hosts (high bipartite degree) first
The research question is whether anything graph-derived beats per-host frequency counting.
If it does not, that is the finding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AccessTrace:
    n_ranks: int
    n_objects: int
    obj_bytes: np.ndarray            # size of each object
    reads: np.ndarray                # reads[r, o] = number of reads of object o by rank r (per step)
    meta: dict = field(default_factory=dict)

    def host_reads(self, host: np.ndarray, n_hosts: int) -> np.ndarray:
        R = np.zeros((n_hosts, self.n_objects))
        np.add.at(R, host, self.reads)
        return R


def embedding_workload(n_ranks: int = 64, n_objects: int = 4096, zipf: float = 1.1,
                       affinity: float = 0.5, n_groups: int = 8, lookups_per_rank: int = 20000,
                       obj_bytes: float = 1 << 20, seed: int = 0) -> AccessTrace:
    """Sharded embedding table. Global popularity is Zipf; with probability `affinity` a lookup
    from group g targets the objects 'owned' by group g (models per-domain feature locality).
    Groups are interleaved across ranks (rank % n_groups) so block hosting does not align."""
    rng = np.random.default_rng(seed)
    pop = np.empty(n_objects)
    pop[rng.permutation(n_objects)] = 1 / np.arange(1, n_objects + 1) ** zipf
    pop /= pop.sum()
    obj_group = rng.integers(0, n_groups, n_objects)
    group = np.arange(n_ranks) % n_groups
    reads = np.zeros((n_ranks, n_objects))
    for r in range(n_ranks):
        local = np.where(obj_group == group[r], pop, 0)
        local /= local.sum()
        reads[r] = rng.multinomial(lookups_per_rank, (1 - affinity) * pop + affinity * local)
    return AccessTrace(n_ranks, n_objects, np.full(n_objects, obj_bytes), reads,
                       {"name": "embedding", "group": group, "obj_group": obj_group,
                        "assumptions": ["per-group feature locality is a parameter",
                                        "uniform object size"]})


def _greedy_home(Rb: np.ndarray, obj_bytes: np.ndarray, budget: float) -> np.ndarray:
    """Assign each object a home host, preferring the host that reads it most, with equal
    per-host storage budget. Objects processed in descending total demand."""
    H, O = Rb.shape
    home = np.full(O, -1)
    used = np.zeros(H)
    for o in np.argsort(-Rb.sum(0)):
        for h in np.argsort(-Rb[:, o]):
            if used[h] + obj_bytes[o] <= budget:
                home[o] = h; used[h] += obj_bytes[o]; break
        if home[o] < 0:
            h = int(np.argmin(used)); home[o] = h; used[h] += obj_bytes[o]
    return home


def evaluate(trace: AccessTrace, host: np.ndarray, n_hosts: int, policy: str,
             cache_bytes: float) -> dict:
    R = trace.host_reads(host, n_hosts)          # host x object read counts
    Rb = R * trace.obj_bytes[None, :]            # bytes
    O = trace.n_objects
    if policy.startswith("rr_home"):
        home = np.arange(O) % n_hosts
    else:
        home = _greedy_home(Rb, trace.obj_bytes, trace.obj_bytes.sum() / n_hosts * 1.02)
    local = np.zeros((n_hosts, O), dtype=bool)
    local[home, np.arange(O)] = True
    if "lfu" in policy:
        degree = (R > 0).sum(0)
        for h in range(n_hosts):
            score = Rb[h].copy()
            if "rep" in policy:
                score = score * (1 + np.log1p(degree))   # favour widely shared objects
            score[local[h]] = -1
            used = 0.0
            for o in np.argsort(-score):
                if score[o] <= 0 or used + trace.obj_bytes[o] > cache_bytes:
                    continue
                local[h, o] = True; used += trace.obj_bytes[o]
    remote = Rb * ~local
    inbound = remote.sum(1)
    serve = np.bincount(home, weights=remote.sum(0), minlength=n_hosts)
    return {"policy": policy, "remote_bytes": float(remote.sum()),
            "remote_frac": float(remote.sum() / Rb.sum()),
            "max_host_inbound": float(inbound.max()), "max_host_serve": float(serve.max())}


POLICIES = ("rr_home", "rr_home+lfu", "graph_home+lfu", "graph_home+lfu+rep")

"""Paper 2 — T-GAPE: topology-aware mapping.

Physical model: two-tier leaf-spine (the common shape of GPU clusters).
  GPU -> host (NVLink, intra_bw) -> leaf switch (one NIC link per host, nic_bw per direction)
  leaf -> spine (one aggregated uplink per leaf, bw = hosts_per_leaf * nic_bw / oversubscription)
Routing: shortest path; the spine is treated as one non-blocking crossbar (ECMP ideal).
Link time = bytes on link / link bandwidth; estimated step comm time = max link time + intra term.

Mapping strategies compared
  block          ranks in order onto hosts in order; hosts in order onto leaves
  flat           CAGE host partition (Paper 1), host labels matched to block placement to
                 minimise moves, hosts onto leaves in that order -> oblivious above the host
  hierarchical   CAGE host partition, then partition the HOST traffic graph onto leaves
                 (bottom-up two-level mapping)
  top_down       partition ranks onto leaves first, then hosts inside each leaf
Classic predecessors to cite: Hoefler & Snir ICS'11 (generic topology mapping), TreeMatch
(Jeannot et al.), Scotch/LibTopoMap. T-GAPE's open question is not whether hierarchical mapping
helps (known) but how much it helps for modern ML traffic under realistic oversubscription.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cage import partition as P


@dataclass
class LeafSpine:
    n_leaves: int
    hosts_per_leaf: int = 4
    gpus_per_host: int = 8
    intra_bw: float = 300e9
    nic_bw: float = 50e9
    oversubscription: float = 1.0

    @property
    def n_hosts(self) -> int:
        return self.n_leaves * self.hosts_per_leaf

    @property
    def n_gpus(self) -> int:
        return self.n_hosts * self.gpus_per_host

    @property
    def uplink_bw(self) -> float:
        return self.hosts_per_leaf * self.nic_bw / self.oversubscription

    @classmethod
    def for_ranks(cls, n_ranks: int, gpus_per_host: int = 8, hosts_per_leaf: int = 4, **kw):
        hosts = -(-n_ranks // gpus_per_host)
        return cls(-(-hosts // hosts_per_leaf), hosts_per_leaf, gpus_per_host, **kw)


def host_matrix(M: np.ndarray, host: np.ndarray, n_hosts: int) -> np.ndarray:
    H = np.zeros((n_hosts, n_hosts))
    np.add.at(H, (host[:, None].repeat(len(host), 1), host[None, :].repeat(len(host), 0)), M)
    return H


def evaluate(M: np.ndarray, host: np.ndarray, leaf_of_host: np.ndarray, topo: LeafSpine) -> dict:
    host = np.asarray(host)
    leaf_of_host = np.asarray(leaf_of_host)
    H = host_matrix(M, host, topo.n_hosts)
    intra = np.diag(H).copy()
    Hx = H.copy(); np.fill_diagonal(Hx, 0)
    nic_t = np.maximum(Hx.sum(1), Hx.sum(0)) / topo.nic_bw
    L = np.zeros((topo.n_leaves, topo.n_leaves))
    np.add.at(L, (leaf_of_host[:, None].repeat(topo.n_hosts, 1),
                  leaf_of_host[None, :].repeat(topo.n_hosts, 0)), Hx)
    np.fill_diagonal(L, 0)
    up_t = np.maximum(L.sum(1), L.sum(0)) / topo.uplink_bw
    leaf = leaf_of_host[host]
    same_host = host[:, None] == host[None, :]
    same_leaf = leaf[:, None] == leaf[None, :]
    hops = np.where(same_host, 0, np.where(same_leaf, 2, 4))
    total = M.sum()
    return {"cross_host_bytes": float(Hx.sum()), "cross_leaf_bytes": float(L.sum()),
            "cross_leaf_frac": float(L.sum() / total) if total else 0.0,
            "hop_bytes": float((M * hops).sum()),
            "max_nic_time_s": float(nic_t.max()), "max_uplink_time_s": float(up_t.max()),
            "intra_time_s": float((intra / topo.intra_bw).max()),
            "est_comm_time_s": float(max(nic_t.max(), up_t.max()) + (intra / topo.intra_bw).max())}


def map_ranks(M: np.ndarray, topo: LeafSpine, strategy: str = "hierarchical",
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (host_of_rank, leaf_of_host)."""
    n = M.shape[0]
    g, hpl = topo.gpus_per_host, topo.hosts_per_leaf
    identity_leaves = np.arange(topo.n_hosts) // hpl
    if strategy == "block":
        return P.block(n, g), identity_leaves
    host = P.partition(M, g, method="best", seed=seed)
    if strategy == "flat":
        # host labels renamed to keep as many ranks as possible where block put them — what a
        # topology-oblivious operator would do; host-to-leaf order then follows that naming
        return P.remap_to_reference(host, P.block(n, g)), identity_leaves
    if strategy == "hierarchical":
        H = host_matrix(M, host, topo.n_hosts)
        np.fill_diagonal(H, 0)
        return host, P.partition(H, hpl, method="best", seed=seed)
    if strategy == "top_down":
        leaf_of_rank = P.partition(M, g * hpl, method="best", seed=seed)
        host = np.empty(n, dtype=int)
        for lf in range(topo.n_leaves):
            members = np.nonzero(leaf_of_rank == lf)[0]
            if len(members) == 0:
                continue
            sub = P.partition(M[np.ix_(members, members)], g, method="best", seed=seed)
            host[members] = lf * hpl + sub
        return host, identity_leaves
    raise ValueError(strategy)


STRATEGIES = ("block", "flat", "hierarchical", "top_down")

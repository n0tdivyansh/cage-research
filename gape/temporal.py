"""Paper 4 — I-GAPE: iteration-aware analysis.

Question: how stable is the communication graph across iterations, and when can a placement
computed earlier be reused instead of recomputed?

Similarity between two traffic matrices A, B (directed, bytes):
  rel_l1     |A-B|_1 / (|A|_1 + |B|_1)            in [0,1], 0 = identical
  cosine     1 - <A,B>/(|A||B|)                     in [0,1]
  spectral   L2 distance of sorted normalised-Laplacian eigenvalues of the undirected views
             (permutation invariant: detects SHAPE change, blind to WHICH ranks changed)
  partition  1 - NMI of Louvain communities
Placement-relevant drift is about which ranks talk to which, so rel_l1/cosine (rank-aligned) are
the triggers; spectral distance is reported to show it misses relabelling-type drift.

Re-planning policies simulated over a step sequence:
  static_block   never plan (block placement)
  once           plan after a warm-up window, never again (= Paper 1 CAGE used offline)
  periodic(k)    re-plan every k steps from the last window
  triggered(th)  re-plan when rel_l1(current window, window at last plan) > th
  oracle         re-plan every step on that step's own matrix (unachievable upper bound)
Every re-plan pays migration_cost_s * (ranks moved) — moving a rank moves its state.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import laplacian

from cage import metrics as m
from cage import partition as P
from cage import placement as PL


def rel_l1(A: np.ndarray, B: np.ndarray) -> float:
    s = np.abs(A).sum() + np.abs(B).sum()
    return float(np.abs(A - B).sum() / s) if s else 0.0


def cosine_dist(A: np.ndarray, B: np.ndarray) -> float:
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    return float(1 - (A * B).sum() / (na * nb)) if na and nb else 1.0


def spectral_dist(A: np.ndarray, B: np.ndarray) -> float:
    ea = np.sort(np.linalg.eigvalsh(laplacian(A + A.T, normed=True)))
    eb = np.sort(np.linalg.eigvalsh(laplacian(B + B.T, normed=True)))
    return float(np.linalg.norm(ea - eb) / np.sqrt(len(ea)))


def partition_dist(A: np.ndarray, B: np.ndarray, seed: int = 0) -> float:
    la, _ = m.louvain(A, seed=seed)
    lb, _ = m.louvain(B, seed=seed)
    return 1 - m.nmi(la, lb)


def drift_series(mats: list[np.ndarray], ref: str = "previous") -> dict[str, list[float]]:
    """ref='previous': step t vs t-1. ref='first': step t vs step 0."""
    out = {"rel_l1": [], "cosine": [], "spectral": [], "partition": []}
    for t in range(1, len(mats)):
        A = mats[t - 1] if ref == "previous" else mats[0]
        B = mats[t]
        out["rel_l1"].append(rel_l1(A, B)); out["cosine"].append(cosine_dist(A, B))
        out["spectral"].append(spectral_dist(A, B)); out["partition"].append(partition_dist(A, B))
    return out


def simulate_policy(mats: list[np.ndarray], capacity: int, cluster: PL.Cluster, policy: str,
                    param: float | None = None, window: int = 5, migration_cost_s: float = 0.0,
                    seed: int = 0) -> dict:
    n = mats[0].shape[0]
    host = P.block(n, capacity)
    comm_t, replans, moved = [], 0, 0
    ref_window = None
    for t, M in enumerate(mats):
        win = sum(mats[max(0, t - window):t]) if t > 0 else None
        plan = False
        if policy == "once":
            plan = t == window
        elif policy == "periodic":
            plan = t >= window and (t - window) % int(param) == 0
        elif policy == "triggered":
            if t == window:
                plan = True
            elif t > window and ref_window is not None:
                plan = rel_l1(win / window, ref_window) > param
        elif policy == "oracle":
            plan = True
        elif policy != "static_block":
            raise ValueError(policy)
        if plan:
            basis = M if policy == "oracle" else win / window
            new = P.partition(basis, capacity, method="best", seed=seed)
            new = P.remap_to_reference(new, host)
            moved += int((new != host).sum())
            host = new
            replans += 1
            ref_window = basis if policy != "oracle" else M
        comm_t.append(PL.evaluate(M, host, cluster)["est_comm_time_s"])
    comm = float(np.sum(comm_t))
    return {"policy": policy, "param": param, "comm_time_s": comm, "replans": replans,
            "ranks_moved": moved, "migration_s": moved * migration_cost_s,
            "total_s": comm + moved * migration_cost_s, "per_step": comm_t}

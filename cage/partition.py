"""Capacity-constrained partitioning of the communication graph onto hosts.

Hosts have a fixed number of GPU slots, so a placement must put EXACTLY `capacity` ranks on each
host. METIS only bounds imbalance (ufactor), so its output is repaired to exact capacity and then
improved with pairwise-swap refinement (Kernighan–Lin style gains), which keeps capacity exact.

Baselines (what a reviewer will expect):
  block        ranks [0..g) on host 0, [g..2g) on host 1, ...  — SLURM block / torchrun default
  round_robin  rank r on host r % H                              — SLURM cyclic (a weak baseline)
  random       uniform random exact-capacity assignment
Report gains against `block` first. Gains against round_robin are not a headline result.
"""
from __future__ import annotations

import numpy as np

try:
    import pymetis
    HAVE_METIS = True
except ImportError:  # pragma: no cover
    HAVE_METIS = False


# ------------------------------------------------------------------------------ baselines
def block(n: int, capacity: int) -> np.ndarray:
    return np.arange(n) // capacity


def round_robin(n: int, capacity: int) -> np.ndarray:
    return np.arange(n) % (-(-n // capacity))


def random_placement(n: int, capacity: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(block(n, capacity))


# ---------------------------------------------------------------------------- objective
def cut_weight(W: np.ndarray, labels: np.ndarray) -> float:
    """Sum of symmetric weight on edges crossing parts (each undirected pair counted once)."""
    diff = labels[:, None] != labels[None, :]
    return float(np.triu(W * diff, 1).sum())


# ---------------------------------------------------------------------------- algorithms
def _metis(W: np.ndarray, n_parts: int, seed: int) -> np.ndarray:
    n = W.shape[0]
    mx = W.max()
    if mx <= 0:
        return block(n, -(-n // n_parts))
    Wi = np.where(W > 0, np.maximum(1, np.round(W / mx * 1e6)), 0).astype(np.int64)
    xadj, adjncy, ew = [0], [], []
    for i in range(n):
        nb = np.nonzero(Wi[i])[0]
        nb = nb[nb != i]
        adjncy.extend(nb.tolist()); ew.extend(Wi[i, nb].tolist()); xadj.append(len(adjncy))
    opts = pymetis.Options(seed=seed, ufactor=1, ncuts=8)
    if hasattr(pymetis, "CSRAdjacency"):
        res = pymetis.part_graph(n_parts, pymetis.CSRAdjacency(xadj, adjncy), eweights=ew, options=opts)
    else:  # older pymetis
        res = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy, eweights=ew, options=opts)
    parts = res.vertex_part if hasattr(res, "vertex_part") else res[1]
    return np.asarray(parts, dtype=int)


def enforce_capacity(W: np.ndarray, labels: np.ndarray, n_parts: int, capacity: int) -> np.ndarray:
    """Greedy repair: move ranks out of overfull parts into the underfull part they are most
    connected to, choosing the move that loses least cut quality."""
    lab = labels.copy()
    D = np.stack([W[:, lab == p].sum(1) for p in range(n_parts)], axis=1)
    sizes = np.bincount(lab, minlength=n_parts)
    while (sizes > capacity).any():
        over = np.nonzero(sizes > capacity)[0]
        under = np.nonzero(sizes < capacity)[0]
        best = None
        for p in over:
            members = np.nonzero(lab == p)[0]
            gains = D[np.ix_(members, under)] - D[members, p][:, None]
            i, j = np.unravel_index(np.argmax(gains), gains.shape)
            if best is None or gains[i, j] > best[0]:
                best = (gains[i, j], members[i], p, under[j])
        _, r, a, c = best
        lab[r] = c; sizes[a] -= 1; sizes[c] += 1
        D[:, a] -= W[:, r]; D[:, c] += W[:, r]
    return lab


def refine_swaps(W: np.ndarray, labels: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """Best-improvement pairwise swaps between parts until no swap reduces the cut.
    gain(i<->j) = D[i,b]-D[i,a] + D[j,a]-D[j,b] - 2W[i,j]  for i in a, j in b."""
    lab = labels.copy()
    n = len(lab)
    k = lab.max() + 1
    D = np.stack([W[:, lab == p].sum(1) for p in range(k)], axis=1)
    max_iter = max_iter or 4 * n
    for _ in range(max_iter):
        A = D[:, lab]                      # A[i, j] = D[i, lab[j]]
        s = D[np.arange(n), lab]
        G = A - s[:, None] + A.T - s[None, :] - 2 * W
        G[lab[:, None] == lab[None, :]] = -np.inf
        i, j = np.unravel_index(np.argmax(G), G.shape)
        if not G[i, j] > 1e-9 * max(W.max(), 1e-30):
            break
        a, b = lab[i], lab[j]
        D[:, a] += W[:, j] - W[:, i]
        D[:, b] += W[:, i] - W[:, j]
        lab[i], lab[j] = b, a
    return lab


def partition(M: np.ndarray, capacity: int, method: str = "metis", seed: int = 0,
              refine: bool = True) -> np.ndarray:
    """Return host label per rank with exactly `capacity` ranks per host (last host may be
    smaller when n is not a multiple). method: 'metis' | 'refine_block' (no METIS: swap
    refinement from block placement) | 'refine_random' | 'best' (lowest cut of all three —
    this is what CAGE recommends; the refinement from block guarantees it is never worse
    than block placement)."""
    W = M + M.T
    n = W.shape[0]
    n_parts = -(-n // capacity)
    if n_parts == 1:
        return np.zeros(n, dtype=int)
    if method == "best":
        cands = [partition(M, capacity, m, seed, refine) for m in
                 (["metis"] if HAVE_METIS else []) + ["refine_block", "refine_random"]]
        return min(cands, key=lambda l: cut_weight(W, l))
    if method == "metis" and HAVE_METIS:
        lab = _metis(W, n_parts, seed)
        lab = enforce_capacity(W, lab, n_parts, capacity)
    elif method in ("metis", "refine_block"):
        lab = block(n, capacity)
    elif method == "refine_random":
        lab = random_placement(n, capacity, seed)
    else:
        raise ValueError(method)
    if refine:
        lab = refine_swaps(W, lab)
    return lab


def remap_to_reference(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rename parts so that as many ranks as possible keep their reference host
    (Hungarian matching). Minimises the number of ranks an operator must move."""
    from scipy.optimize import linear_sum_assignment
    k = max(labels.max(), reference.max()) + 1
    T = np.zeros((k, k))
    np.add.at(T, (labels, reference), 1)
    r, c = linear_sum_assignment(-T)
    mapping = np.empty(k, dtype=int); mapping[r] = c
    return mapping[labels]

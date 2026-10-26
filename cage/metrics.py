"""Graph metrics on traffic matrices.

Conventions
- M is the directed traffic matrix (M[i,j] = bytes i -> j). W = M + M.T is the undirected view.
- Weighted degree (strength) of rank i = sent + received = W[i].sum().
  sum_i strength_i == 2 * total traffic  (each byte has a sender and a receiver).
- Modularity is never reported without a null model. Sparse and regular graphs (rings, grids,
  random sparse graphs) have high Q with no meaningful community structure (Guimera et al. 2004).
- Betweenness on an application-level communication graph is EXPERIMENTAL: messages do not
  route through intermediate ranks, so shortest paths in this graph are not physical paths.
  It is computed with distance = 1/weight (heavy edges are short), never with raw weight.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- load / hotspots
def strength(M: np.ndarray) -> dict[str, np.ndarray]:
    out = M.sum(axis=1)
    inn = M.sum(axis=0)
    return {"out": out, "in": inn, "total": out + inn}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.std() / x.mean()) if x.mean() > 0 else 0.0


def top_share(x: np.ndarray, frac: float = 0.2) -> float:
    """Share of total held by the top `frac` of ranks."""
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(round(frac * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else 0.0


def hotspots(M: np.ndarray, z_thresh: float = 3.5, top_partners: int = 5) -> list[dict]:
    """Ranks whose strength is an outlier by robust z-score (median / MAD).
    Robust z avoids the masking effect where a few large hubs inflate the std."""
    s = strength(M)["total"]
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826
    if mad == 0:
        mad = s.std() if s.std() > 0 else 1.0
    z = (s - med) / mad
    W = M + M.T
    out = []
    for r in np.argsort(-z):
        if z[r] < z_thresh:
            break
        partners = np.argsort(-W[r])[:top_partners]
        out.append({"rank": int(r), "strength": float(s[r]), "robust_z": float(z[r]),
                    "x_median": float(s[r] / med) if med > 0 else float("inf"),
                    "sent": float(M[r].sum()), "received": float(M[:, r].sum()),
                    "top_partners": [(int(p), float(W[r, p])) for p in partners if W[r, p] > 0]})
    return out


def load_summary(M: np.ndarray) -> dict:
    s = strength(M)["total"]
    return {"total_bytes": float(M.sum()), "strength_sum": float(s.sum()),
            "strength_mean": float(s.mean()), "strength_median": float(np.median(s)),
            "strength_max": float(s.max()), "max_over_median": float(s.max() / np.median(s))
            if np.median(s) > 0 else float("inf"),
            "gini": gini(s), "cv": cv(s), "top10_share": top_share(s, 0.1),
            "top20_share": top_share(s, 0.2), "isolated_ranks": int((s == 0).sum())}


# ------------------------------------------------------------------------------ structure
def undirected(M: np.ndarray) -> nx.Graph:
    W = M + M.T
    G = nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    ii, jj = np.nonzero(np.triu(W, 1))
    G.add_weighted_edges_from(zip(ii.tolist(), jj.tolist(), W[ii, jj].tolist()))
    return G


def louvain(M: np.ndarray, seed: int = 0, resolution: float = 1.0) -> tuple[np.ndarray, float]:
    G = undirected(M)
    if G.number_of_edges() == 0:
        return np.arange(M.shape[0]), 0.0
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    labels = np.empty(M.shape[0], dtype=int)
    for c, nodes in enumerate(comms):
        labels[list(nodes)] = c
    Q = nx.community.modularity(G, comms, weight="weight", resolution=resolution)
    return labels, float(Q)


def modularity_of(M: np.ndarray, labels: np.ndarray) -> float:
    G = undirected(M)
    comms = [set(np.nonzero(labels == c)[0].tolist()) for c in np.unique(labels)]
    return float(nx.community.modularity(G, comms, weight="weight"))


def _null_sample(W: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving rewiring (double-edge swaps) carrying edge weights with the edges.
    Preserves the degree sequence exactly and the weight distribution exactly; strengths
    approximately. Falls back to weight shuffling when the graph is too dense to rewire."""
    n = W.shape[0]
    ii, jj = np.nonzero(np.triu(W, 1))
    w = W[ii, jj].copy()
    m = len(w)
    if m < 2:
        return W.copy()
    density = m / (n * (n - 1) / 2)
    if density > 0.8:
        P = np.zeros_like(W)
        P[ii, jj] = rng.permutation(w)
        return P + P.T
    edges = set(zip(ii.tolist(), jj.tolist()))
    a, c = ii.copy(), jj.copy()
    for _ in range(swaps_per_edge * m):
        x, y = rng.integers(0, m, 2)
        if x == y:
            continue
        u, v, s, t = a[x], c[x], a[y], c[y]
        if rng.random() < 0.5:
            s, t = t, s
        if len({u, v, s, t}) < 4:
            continue
        e1, e2 = (min(u, t), max(u, t)), (min(s, v), max(s, v))
        if e1 in edges or e2 in edges:
            continue
        edges.discard((min(u, v), max(u, v))); edges.discard((min(s, t), max(s, t)))
        edges.add(e1); edges.add(e2)
        a[x], c[x] = e1; a[y], c[y] = e2
    P = np.zeros_like(W)
    P[a, c] = w
    return P + P.T


def modularity_significance(M: np.ndarray, n_null: int = 20, seed: int = 0) -> dict:
    """Louvain Q of the observed graph vs. Louvain Q of degree-preserving null graphs.

    Report z and the empirical p-value, not raw Q. A ring of 32 has Q ~0.7 and still no
    community structure beyond what its degree sequence forces.
    """
    rng = np.random.default_rng(seed)
    labels, Q = louvain(M, seed=seed)
    W = M + M.T
    null_q = []
    for i in range(n_null):
        P = _null_sample(W, rng)
        _, q = louvain(np.triu(P), seed=seed + i + 1)
        null_q.append(q)
    null_q = np.asarray(null_q)
    sd = null_q.std(ddof=1) if n_null > 1 else 0.0
    return {"Q": Q, "n_communities": int(len(np.unique(labels))), "labels": labels,
            "null_mean": float(null_q.mean()), "null_std": float(sd),
            "z": float((Q - null_q.mean()) / sd) if sd > 0 else float("inf") if Q > null_q.mean() else 0.0,
            "p_empirical": float((1 + (null_q >= Q).sum()) / (1 + n_null)),
            "n_null": n_null}


def betweenness(M: np.ndarray, k: int | None = None, seed: int = 0) -> np.ndarray:
    """EXPERIMENTAL. Normalised betweenness with distance = 1/weight on the undirected view.
    `k` samples pivots (Brandes approximation) for large graphs."""
    G = undirected(M)
    for _, _, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]
    bc = nx.betweenness_centrality(G, k=k, weight="distance", normalized=True, seed=seed)
    return np.array([bc[i] for i in range(M.shape[0])])


# ------------------------------------------------------------------------ label agreement
def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    T = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(T, (ai, bi), 1)
    return T


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic normalisation)."""
    T = _contingency(a, b)
    n = T.sum()
    pa, pb, pab = T.sum(1) / n, T.sum(0) / n, T / n
    nz = pab > 0
    mi = (pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    ha, hb = h(pa), h(pb)
    if ha + hb == 0:
        return 1.0
    return float(2 * mi / (ha + hb))


def ari(a: np.ndarray, b: np.ndarray) -> float:
    T = _contingency(a, b)
    c2 = lambda x: x * (x - 1) / 2
    n = T.sum()
    s_ij, s_a, s_b = c2(T).sum(), c2(T.sum(1)).sum(), c2(T.sum(0)).sum()
    exp = s_a * s_b / c2(n)
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


def rank_agreement(x: np.ndarray, y: np.ndarray, top: int = 10) -> dict:
    tau = stats.kendalltau(x, y).statistic
    tx, ty = set(np.argsort(-x)[:top]), set(np.argsort(-y)[:top])
    return {"kendall_tau": float(tau), f"top{top}_jaccard": len(tx & ty) / len(tx | ty)}


def summarize(M: np.ndarray, n_null: int = 20, seed: int = 0, with_betweenness: bool = False) -> dict:
    sig = modularity_significance(M, n_null=n_null, seed=seed)
    out = {"load": load_summary(M), "hotspots": hotspots(M),
           "communities": {k: v for k, v in sig.items() if k != "labels"},
           "community_labels": sig["labels"].tolist(),
           "density": float((M + M.T > 0).sum() / (M.shape[0] * (M.shape[0] - 1)))}
    if with_betweenness:
        out["betweenness_experimental"] = betweenness(M).tolist()
    return out

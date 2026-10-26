"""Invariants the paper's numbers depend on. If one of these fails, no result is trustworthy."""
import numpy as np
import pytest

from cage import collectives as C, metrics as m, partition as P, placement as PL, workloads as W
from cage.trace import CommTrace, TraceBuilder, concat


def test_ring_allreduce_bytes_per_rank():
    b = TraceBuilder(8)
    C.ring_allreduce(b, range(8), 1000.0)
    M = b.build().matrix()
    np.testing.assert_allclose(M.sum(1), 2 * 7 / 8 * 1000)          # bandwidth-optimal volume
    assert (M > 0).sum() == 8                                          # one edge per ring hop


def test_allgather_and_reduce_scatter_volume():
    for fn in (C.ring_allgather, C.ring_reduce_scatter):
        b = TraceBuilder(4)
        fn(b, range(4), 400.0)
        np.testing.assert_allclose(b.build().matrix().sum(1), 3 / 4 * 400)


def test_rail_spreads_inter_host_load_vs_leader():
    hosts = [list(range(h * 4, h * 4 + 4)) for h in range(4)]
    host = np.arange(16) // 4
    cross = host[:, None] != host[None, :]
    out = {}
    for name, fn in (("leader", C.hier_leader_allreduce), ("rail", C.hier_rail_allreduce)):
        b = TraceBuilder(16)
        fn(b, hosts, 1e6)
        M = b.build().matrix()
        out[name] = (M * cross).sum(1).max()
    assert out["rail"] < out["leader"]


def test_strength_sums_to_twice_traffic():
    for t in (W.megatron_3d(4, 2, 2), W.moe(world=16, steps=2), W.stencil((4, 4))):
        M = t.matrix(per_step=True)
        assert np.isclose(m.strength(M)["total"].sum(), 2 * M.sum())


def test_csv_roundtrip(tmp_path):
    t = W.megatron_3d(4, 2, 2, steps=2)
    t.to_csv(tmp_path / "t.csv")
    u = CommTrace.from_csv(tmp_path / "t.csv")
    np.testing.assert_allclose(t.matrix(), u.matrix())
    assert u.n_steps == 2 and u.n_ranks == t.n_ranks
    np.testing.assert_array_equal(u.meta["labels"]["stage"], t.meta["labels"]["stage"])


def test_relabel_moves_traffic_and_labels():
    t = W.planted_partition(16, 4, seed=1)
    perm = np.random.default_rng(0).permutation(16)
    u = t.relabel(perm)
    M, N = t.matrix(), u.matrix()
    np.testing.assert_allclose(N[np.ix_(perm, perm)], M)
    np.testing.assert_array_equal(u.meta["labels"]["block"][perm], t.meta["labels"]["block"])


def test_concat_offsets_steps():
    a, b = W.dp_ring(8, steps=2), W.dp_ring(8, steps=3)
    c = concat([a, b])
    assert c.n_steps == 5 and np.isclose(c.total_bytes, a.total_bytes + b.total_bytes)


def test_gini_bounds():
    assert m.gini(np.ones(10)) == pytest.approx(0)
    x = np.zeros(10); x[0] = 1
    assert m.gini(x) == pytest.approx(0.9)


def test_nmi_ari_identity_and_permutation():
    a = np.repeat(np.arange(4), 5)
    assert m.nmi(a, a) == pytest.approx(1) and m.ari(a, a) == pytest.approx(1)
    assert m.nmi(a, (a + 1) % 4) == pytest.approx(1)


def test_null_model_preserves_degrees():
    rng = np.random.default_rng(0)
    A = np.triu(rng.random((30, 30)) < 0.15, 1).astype(float) * rng.random((30, 30))
    W_ = A + A.T
    P_ = m._null_sample(W_, rng)
    np.testing.assert_array_equal((P_ > 0).sum(1), (W_ > 0).sum(1))
    np.testing.assert_allclose(np.sort(P_[np.triu_indices(30, 1)]), np.sort(W_[np.triu_indices(30, 1)]))


def test_ring_has_high_q_but_no_significance():
    s = m.modularity_significance(W.dp_ring(64).matrix(per_step=True), n_null=8)
    assert s["Q"] > 0.5 and s["z"] < 3


def test_sbm_is_significant():
    s = m.modularity_significance(W.planted_partition(64, 8, 0.6, 0.01).matrix(), n_null=8)
    assert s["z"] > 5


@pytest.mark.parametrize("method", ["metis", "refine_block", "refine_random", "best"])
def test_partition_exact_capacity(method):
    M = W.moe(world=32, steps=1, affinity=0.5).matrix()
    lab = P.partition(M, 8, method)
    assert np.bincount(lab).tolist() == [8, 8, 8, 8]


def test_best_partition_never_worse_than_block():
    for t in (W.dp_ring(32), W.moe(world=32, steps=1), W.stencil((4, 8)), W.megatron_3d(4, 4, 2)):
        M = t.matrix()
        W_ = M + M.T
        assert P.cut_weight(W_, P.partition(M, 8, "best")) <= P.cut_weight(W_, P.block(t.n_ranks, 8)) + 1e-6


def test_partition_recovers_scrambled_planted_blocks():
    t = W.planted_partition(64, 8, 0.7, 0.01, seed=2).relabel(np.random.default_rng(1).permutation(64))
    lab = P.partition(t.matrix(), 8, "best")
    assert m.ari(lab, t.meta["labels"]["block"]) > 0.95


def test_remap_minimises_moves():
    ref = P.block(16, 4)
    perm_labels = np.array([2, 3, 0, 1])[ref]               # same grouping, renamed parts
    assert (P.remap_to_reference(perm_labels, ref) == ref).all()


def test_placement_eval_and_slow_nic():
    t = W.megatron_3d(4, 2, 4)
    M = t.matrix(per_step=True)
    host = P.block(t.n_ranks, 8)
    cl = PL.Cluster.for_ranks(t.n_ranks, 8)
    base = PL.evaluate(M, host, cl)
    assert 0 <= base["cross_host_frac"] <= 1
    cl.nic_factor[1] = 0.1
    slow = PL.evaluate(M, host, cl)
    assert slow["est_comm_time_s"] >= base["est_comm_time_s"]
    assert slow["cross_host_bytes"] == base["cross_host_bytes"]   # bytes cannot see the fault


def test_pinned_matrix_excludes_collectives():
    t = W.moe(world=16, steps=1)
    Mp = PL.pinned_matrix(t, per_step=False)
    Ma = t.matrix(ops=["ep_alltoall"])
    np.testing.assert_allclose(Mp, Ma)

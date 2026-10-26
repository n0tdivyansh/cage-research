import numpy as np

from cage import partition as P, placement as PL, workloads as W
from gape import cache as CA, engine as E, temporal as T, topology as TP


def test_topology_strategies_respect_capacity():
    M = PL.pinned_matrix(W.moe(world=64, affinity=0.6, steps=1))
    topo = TP.LeafSpine.for_ranks(64, 8, 2, oversubscription=2)
    for s in TP.STRATEGIES:
        host, leaf = TP.map_ranks(M, topo, s)
        assert np.bincount(host, minlength=8).tolist() == [8] * 8
        assert np.bincount(leaf, minlength=4).tolist() == [2] * 4


def test_topology_eval_bytes_consistent():
    M = W.stencil((8, 8)).matrix(per_step=True)
    topo = TP.LeafSpine.for_ranks(64, 8, 4)
    host, leaf = TP.map_ranks(M, topo, "block")
    r = TP.evaluate(M, host, leaf, topo)
    flat = PL.evaluate(M, host, PL.Cluster.for_ranks(64, 8))
    assert np.isclose(r["cross_host_bytes"], flat["cross_host_bytes"])
    assert r["cross_leaf_bytes"] <= r["cross_host_bytes"]


def test_drift_detects_relabel_but_spectral_does_not_much():
    t = W.moe(world=32, affinity=0.6, steps=1)
    A = PL.pinned_matrix(t)
    B = PL.pinned_matrix(t.relabel(np.random.default_rng(0).permutation(32)))
    assert T.rel_l1(A, A) == 0
    assert T.rel_l1(A, B) > 0.2
    assert T.spectral_dist(A, B) < 1e-6          # same graph up to relabelling


def test_policy_simulation_runs():
    t = W.moe_phases(n_phases=2, steps_per_phase=8, world=32)
    mats = [PL.pinned_matrix(t.subset(steps=[s]), per_step=False) for s in range(t.n_steps)]
    cl = PL.Cluster.for_ranks(32, 8)
    a = T.simulate_policy(mats, 8, cl, "static_block")
    b = T.simulate_policy(mats, 8, cl, "triggered", 0.2, window=3)
    assert b["comm_time_s"] <= a["comm_time_s"] and b["replans"] >= 1


def test_cache_policies_ordering():
    tr = CA.embedding_workload(n_ranks=32, n_objects=512, lookups_per_rank=2000, seed=0)
    host = P.block(32, 8)
    r = {p: CA.evaluate(tr, host, 4, p, 0.05 * tr.obj_bytes.sum())["remote_frac"] for p in CA.POLICIES}
    assert r["rr_home+lfu"] <= r["rr_home"]


def test_engine_modes():
    t = W.moe_phases(n_phases=2, steps_per_phase=10, world=32)
    mats = [PL.pinned_matrix(t.subset(steps=[s]), per_step=False) for s in range(t.n_steps)]
    topo = TP.LeafSpine.for_ranks(32, 8, 2)
    res = {mode: E.run(mats, topo, E.EngineConfig(window=3), mode) for mode in ("block", "once", "gape")}
    assert res["block"]["replans"] == 0 and res["once"]["replans"] == 1
    assert res["gape"]["comm_time_s"] <= res["block"]["comm_time_s"]


def test_topology_eval_uplink_hand_checked():
    # 3 leaves x 2 hosts x 1 GPU; uplink_bw = 2 hosts * 10 B/s / 4 = 5 B/s
    topo = TP.LeafSpine(n_leaves=3, hosts_per_leaf=2, gpus_per_host=1, nic_bw=10.0, oversubscription=4)
    assert topo.uplink_bw == 5.0
    M = np.zeros((6, 6))
    M[0, 1], M[0, 2], M[1, 4], M[3, 5] = 40, 100, 60, 20
    host = np.arange(6)
    r = TP.evaluate(M, host, np.array([0, 0, 1, 1, 2, 2]), topo)
    # leaf egress [160, 20, 0], ingress [0, 100, 80] -> busiest uplink 160 B / 5 B/s
    assert r["cross_leaf_bytes"] == 180 and r["cross_host_bytes"] == 220
    assert np.isclose(r["cross_leaf_frac"], 180 / 220)
    assert np.isclose(r["max_uplink_time_s"], 32.0)
    assert np.isclose(r["max_nic_time_s"], 14.0)                  # host 0 egress 140 B / 10 B/s
    assert np.isclose(r["est_comm_time_s"], 32.0)                 # uplink-bound
    assert r["hop_bytes"] == 2 * 40 + 4 * (100 + 60 + 20)
    # co-locating hosts 0 and 2 on a leaf: leaf egress [40, 80, 0], ingress [0, 40, 80]
    r2 = TP.evaluate(M, host, np.array([0, 1, 0, 1, 2, 2]), topo)
    assert r2["cross_leaf_bytes"] == 120 and np.isclose(r2["max_uplink_time_s"], 16.0)
    # without oversubscription the same mapping is NIC-bound
    r3 = TP.evaluate(M, host, np.array([0, 0, 1, 1, 2, 2]),
                     TP.LeafSpine(3, 2, 1, nic_bw=10.0, oversubscription=1))
    assert np.isclose(r3["max_uplink_time_s"], 8.0) and np.isclose(r3["est_comm_time_s"], 14.0)

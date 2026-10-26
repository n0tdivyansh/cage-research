"""E5 — Does CAGE find real bottlenecks, or only what bytes can show? (Claim 3, OQ4 option C)

Two injected anomalies on the same workloads:
  (a) VOLUME: one rank sends 3x its normal pinned traffic. A bytes-weighted metric finds this
      by construction — TAUTOLOGICAL, reported only as a sanity check.
  (b) SLOW NIC: one host's NIC runs at 25% bandwidth. Bytes do not change; the step gets slower.
      Bytes-weighted analysis cannot see it. Time-weighted analysis (per-edge transfer time,
      which a real trace gets from per-event durations) can.
The paper should report (b) as the validity test and (a) as a unit test.
"""
import numpy as np

from cage import expkit as X, metrics as m, partition as P, placement as PL, workloads as W

d = X.outdir("01-cage", "e5_detection_validity")
G = 8
rows = []
for name, t in {"MoE aff 0": W.moe(), "Megatron 8·4·2": W.megatron_3d(8, 4, 2),
                "Stencil 2D 8x8": W.stencil((8, 8))}.items():
    M = t.matrix(per_step=True)
    n = t.n_ranks
    host = P.block(n, G)
    H = n // G
    # (a) volume injection
    hits = []
    for r in range(0, n, max(1, n // (8 if X.quick() else 32))):
        Mi = M.copy(); Mi[r] *= 3
        s = m.strength(Mi)["total"]
        hits.append(int(np.argsort(-s).tolist().index(r)) + 1)
    rows.append({"workload": name, "anomaly": "volume x3 (tautological)", "metric": "bytes strength",
                 "hit_rate@1": float(np.mean(np.array(hits) == 1)), "mean_rank_found": float(np.mean(hits))})
    # (b) slow NIC
    hb, ht, hc = [], [], []
    for h in range(H):
        cl = PL.Cluster.for_ranks(n, G)
        cl.nic_factor = np.ones(H); cl.nic_factor[h] = 0.25
        truth = PL.evaluate(M, host, cl)["bottleneck_host"]
        sb = np.bincount(host, weights=m.strength(M)["total"], minlength=H)
        Tm = PL.time_weighted(M, host, cl)
        st = np.bincount(host, weights=m.strength(Tm)["total"], minlength=H)
        hb.append(int(np.argmax(sb)) == truth); ht.append(int(np.argmax(st)) == truth); hc.append(truth == h)
    rows.append({"workload": name, "anomaly": "NIC at 25% bandwidth", "metric": "bytes strength",
                 "hit_rate@1": float(np.mean(hb)), "mean_rank_found": float("nan")})
    rows.append({"workload": name, "anomaly": "NIC at 25% bandwidth", "metric": "time-weighted strength",
                 "hit_rate@1": float(np.mean(ht)), "mean_rank_found": float("nan")})
    rows.append({"workload": name, "anomaly": "NIC at 25% bandwidth", "metric": "(check) slowed host is true bottleneck",
                 "hit_rate@1": float(np.mean(hc)), "mean_rank_found": float("nan")})
X.save_json(d, rows)
lines = X.md_table(rows, ["workload", "anomaly", "metric", "hit_rate@1", "mean_rank_found"],
                   {"hit_rate@1": "{:.2f}", "mean_rank_found": "{:.1f}"})
lines += ["", "Chance level for the NIC test is 1/#hosts = 0.125.",
          "Interpretation: a bytes-only graph cannot attribute slowdowns that do not change volume. "
          "Real extraction must record per-event durations (the PMPI shim, Chakra and Nsight do)."]
X.write_summary(d, "E5 — Detection validity", lines)
print(open(d / "summary.md").read())

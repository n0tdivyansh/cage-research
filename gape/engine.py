"""Paper 5 — GAPE: a windowed, graph-aware placement engine (SIMULATED).

GAPE composes the earlier papers into one control loop:
  observe   per-step communication matrices (CAGE extraction) and, optionally, data accesses
  detect    drift between the current window and the window the plan was built on (I-GAPE)
  plan      capacity-constrained, topology-aware rank mapping (CAGE partition + T-GAPE)
  cache     data homes + caches from the access graph under the new placement (C-GAPE)
            -- NOT YET IN THE LOOP: open task for Paper 5 (see papers/05-gape/README.md)
  act      migrate only if predicted savings over the expected horizon exceed migration cost

Everything here runs against the analytic cost models of placement.py / topology.py.
No claim about wall-clock speedups can come from this module; it decides whether a real
implementation is worth building and which knobs matter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cage import partition as P

from . import temporal as T
from . import topology as TP


@dataclass
class EngineConfig:
    window: int = 5
    drift_threshold: float = 0.15
    horizon_steps: int = 20            # steps a new plan is expected to stay valid
    migration_cost_s: float = 0.5      # seconds per rank moved
    topology_aware: bool = True
    cost_gate: bool = True             # only migrate when predicted savings > migration cost


def run(mats: list[np.ndarray], topo: TP.LeafSpine, cfg: EngineConfig, mode: str = "gape",
        seed: int = 0) -> dict:
    """mode: 'block' (never plan) | 'once' (plan after warm-up only) | 'gape' (adaptive)."""
    n = mats[0].shape[0]
    g = topo.gpus_per_host
    host = P.block(n, g)
    leaf_of_host = np.arange(topo.n_hosts) // topo.hosts_per_leaf
    ref = None
    log, comm, moved, replans, skipped = [], 0.0, 0, 0, 0
    strategy = "hierarchical" if cfg.topology_aware else "flat"
    for t, M in enumerate(mats):
        if t >= cfg.window and mode != "block":
            win = sum(mats[t - cfg.window:t]) / cfg.window
            want = (t == cfg.window) if mode == "once" else (
                ref is None or T.rel_l1(win, ref) > cfg.drift_threshold)
            if want:
                new_host, new_leaf = TP.map_ranks(win, topo, strategy, seed)
                new_host = P.remap_to_reference(new_host, host)
                n_move = int((new_host != host).sum())
                cur = TP.evaluate(win, host, leaf_of_host, topo)["est_comm_time_s"]
                nxt = TP.evaluate(win, new_host, new_leaf, topo)["est_comm_time_s"]
                gain = (cur - nxt) * cfg.horizon_steps
                if not cfg.cost_gate or mode == "once" or gain > n_move * cfg.migration_cost_s:
                    host, leaf_of_host = new_host, new_leaf
                    moved += n_move; replans += 1
                    log.append({"step": t, "moved": n_move, "pred_gain_s": gain})
                    ref = win
                else:
                    # keep the old reference so the decision is re-evaluated next step, when the
                    # window holds more of the new regime and the predicted gain is larger
                    skipped += 1
        comm += TP.evaluate(M, host, leaf_of_host, topo)["est_comm_time_s"]
    mig = moved * cfg.migration_cost_s
    return {"mode": mode, "comm_time_s": comm, "migration_s": mig, "total_s": comm + mig,
            "replans": replans, "skipped_replans": skipped, "ranks_moved": moved, "log": log}

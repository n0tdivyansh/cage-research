"""Parsers against hand-written fixtures with hand-computed expected matrices.
These prove the parsers match the fixtures. They do NOT prove the fixtures match real tool
output — verify against real files and add them here (see CLAUDE.md, task list)."""
from pathlib import Path

import numpy as np

from cage.extract import load

FX = Path(__file__).resolve().parents[1] / "data" / "fixtures"
RING4 = 2 * 3 / 4 * 4096   # ring allreduce of 4096 B over 4 ranks, per ring edge


def edges(t):
    M = t.matrix()
    return {(int(i), int(j)): float(M[i, j]) for i, j in zip(*np.nonzero(M))}


def test_nccl_log():
    t = load(FX / "nccl" / "nccl.log", "nccl")
    assert t.n_ranks == 4
    assert edges(t) == {(0, 1): RING4 + 1024, (1, 2): RING4, (2, 3): RING4 + 200, (3, 0): RING4}


def test_chakra():
    t = load(FX / "chakra", "chakra")
    assert edges(t) == {(0, 1): RING4, (1, 2): RING4, (2, 3): RING4 + 200, (3, 0): RING4}
    assert t.duration is not None


def test_pmpi():
    t = load(FX / "pmpi", "pmpi")
    assert t.n_steps == 2
    e = edges(t)
    assert e[(0, 1)] == 2 * RING4 + 2000 and e[(3, 2)] == 512 and e[(1, 2)] == 2 * RING4 + 2000


def test_csv_autodetect(tmp_path):
    from cage import workloads as W
    W.stencil((4, 4)).to_csv(tmp_path / "s.csv")
    assert load(tmp_path / "s.csv").n_ranks == 16

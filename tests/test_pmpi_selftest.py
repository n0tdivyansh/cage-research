"""Validates the C PMPI shim end to end. Skipped unless `make test` in tools/pmpi_shim set
CAGE_PMPI_SELFTEST to the directory the self-test wrote (needs a real MPI installation)."""
import os

import numpy as np
import pytest

from cage.extract import load

DIR = os.environ.get("CAGE_PMPI_SELFTEST")
pytestmark = pytest.mark.skipif(not DIR, reason="run `make test` in tools/pmpi_shim (needs MPI)")


def test_selftest_trace():
    t = load(DIR, "pmpi")
    assert t.n_ranks == 4 and t.n_steps == 3
    MiB = 1 << 20
    M = t.matrix() / 3                                    # per step
    ring = 2 * 3 / 4 * 8 * MiB                            # allreduce 8 MiB, ring of 4
    for r in range(4):
        expected = MiB + ring + (MiB if r == 0 else 0)    # sendrecv + allreduce (+ bcast 0->1)
        assert M[r, (r + 1) % 4] == pytest.approx(expected)
    assert np.count_nonzero(M) == 4

"""Trace extraction (Layer 1).

Supported inputs
  csv     CAGE event CSV (src,dst,bytes,msgs,step,op,duration_s)            — tested
  pmpi    output directory of tools/pmpi_shim (cage_trace.<rank>.csv files) — parser tested on
          fixtures; the C shim itself is UNTESTED until compiled against a real MPI
  chakra  Chakra execution traces converted to JSON (one file per rank)      — parser tested on a
          hand-written fixture; VERIFY against real chakra_jsonizer output before trusting
  nccl    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL logs                   — parser tested on a
          hand-written fixture; VERIFY against your NCCL version's log format

Why not PMPI for PyTorch: DDP/FSDP on GPUs call NCCL, not MPI. PMPI sees nothing.
"""
from __future__ import annotations

from pathlib import Path

from ..trace import CommTrace


def load(path: str | Path, fmt: str = "auto", **kw) -> CommTrace:
    path = Path(path)
    if fmt == "auto":
        if path.is_dir():
            fmt = "pmpi" if list(path.glob("cage_trace.*.csv")) else "chakra"
        elif path.suffix == ".csv":
            fmt = "csv"
        elif path.suffix in (".log", ".txt"):
            fmt = "nccl"
        elif path.suffix == ".json":
            fmt = "chakra"
        else:
            raise ValueError(f"cannot infer format of {path}; pass fmt=")
    if fmt == "csv":
        return CommTrace.from_csv(path, kw.get("n_ranks"))
    if fmt == "pmpi":
        from .pmpi import load_pmpi
        return load_pmpi(path, **kw)
    if fmt == "chakra":
        from .chakra import load_chakra
        return load_chakra(path, **kw)
    if fmt == "nccl":
        from .nccl_log import load_nccl_log
        return load_nccl_log(path, **kw)
    raise ValueError(fmt)

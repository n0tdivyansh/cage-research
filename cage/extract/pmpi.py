"""Parser for tools/pmpi_shim output.

Each rank writes cage_trace.<world_rank>.csv with header
  kind,op,src,dst,bytes,step,t0,t1,root,members
kind = p2p : recorded on the SENDER only (src = this rank, dst = world rank of peer)
kind = coll: recorded once per call by communicator rank 0; members = ';'-joined world ranks in
             communicator order; bytes = the per-call count*typesize the shim saw
             (full buffer for allreduce/bcast/reduce, per-rank chunk for allgather/
             reduce_scatter_block/alltoall-per-peer*p).
step comes from MPI_Pcontrol(1) calls at iteration boundaries (0 if the app never calls it).
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..trace import CommTrace, TraceBuilder
from ._expand import expand


def load_pmpi(directory: str | Path, algo: str = "ring", n_ranks: int | None = None) -> CommTrace:
    files = sorted(Path(directory).glob("cage_trace.*.csv"),
                   key=lambda p: int(p.name.split(".")[1]))
    if not files:
        raise FileNotFoundError(f"no cage_trace.*.csv in {directory}")
    n = n_ranks or len(files)
    b = TraceBuilder(n)
    for f in files:
        with f.open() as fh:
            for row in csv.DictReader(fh):
                step = int(row.get("step") or 0)
                t0, t1 = row.get("t0"), row.get("t1")
                dur = float(t1) - float(t0) if t0 and t1 else None
                nbytes = float(row["bytes"])
                if row["kind"] == "p2p":
                    b.add(int(row["src"]), int(row["dst"]), nbytes, 1, step, row["op"].lower(), dur)
                else:
                    members = [int(x) for x in row["members"].split(";") if x != ""]
                    root = int(row["root"]) if row.get("root") not in (None, "", "-1") else None
                    expand(b, row["op"], members, nbytes, step, algo, root, duration=dur)
    t = b.build(n_ranks=n)
    t.meta.update(name=Path(directory).name, source="pmpi", collective_algo=algo)
    return t

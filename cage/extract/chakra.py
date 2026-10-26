"""Parser for Chakra execution traces in JSON form (one file per rank).

STATUS: tested only on the hand-written fixture data/fixtures/chakra/. Field names follow the
Chakra protobuf schema as understood when this was written (node types COMM_SEND_NODE=5,
COMM_RECV_NODE=6, COMM_COLL_NODE=7; attrs comm_type, comm_size, comm_dst, comm_src, pg_name).
Convert real traces with `chakra_jsonizer` and check field names before trusting results.
If they differ, fix the constants below and add the real file as a test fixture.

Expected JSON shape per rank file:
  {"rank": 3,                               # optional; else parsed from file name (..._3.json)
   "nodes": [{"id": 12, "name": "...", "type": "COMM_COLL_NODE" | 7,
              "duration_micros": 105,       # optional
              "attr": [{"name": "comm_type", "int64_val": 0},
                       {"name": "comm_size", "int64_val": 4194304},
                       {"name": "pg_name", "string_val": "0"}]}]}

Process-group membership is not reliably inside ETs. Pass pg_members={"pg_name": [ranks...]}
(world group is the default for any unknown pg).
Each rank file records its own view of each collective; edges are kept only where that rank is
the sender, so collectives are not counted p times.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..trace import CommTrace, TraceBuilder
from ._expand import expand

NODE_TYPES = {5: "COMM_SEND_NODE", 6: "COMM_RECV_NODE", 7: "COMM_COLL_NODE"}
COLL_TYPES = {0: "allreduce", 1: "reduce", 2: "allgather", 3: "gather", 4: "scatter",
              5: "broadcast", 6: "alltoall", 7: "reduce_scatter", 8: "reduce_scatter", 9: "barrier"}


def _attrs(node: dict) -> dict:
    out = {}
    for a in node.get("attr", []):
        for k, v in a.items():
            if k != "name":
                out[a["name"]] = v
    return out


def load_chakra(path: str | Path, pg_members: dict | None = None, algo: str = "ring",
                n_ranks: int | None = None, step_regex: str = r"ProfilerStep#(\d+)") -> CommTrace:
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    docs = []
    for f in files:
        d = json.loads(f.read_text())
        r = d.get("rank")
        if r is None:
            m = re.search(r"(\d+)(?!.*\d)", f.stem)
            r = int(m.group(1)) if m else 0
        docs.append((int(r), d))
    n = n_ranks or (max(r for r, _ in docs) + 1)
    world = list(range(n))
    pg_members = {str(k): list(v) for k, v in (pg_members or {}).items()}
    b = TraceBuilder(n)
    step_re = re.compile(step_regex)
    for rank, d in docs:
        step = 0
        for node in d.get("nodes", []):
            ntype = node.get("type")
            ntype = NODE_TYPES.get(ntype, ntype) if isinstance(ntype, int) else ntype
            mstep = step_re.search(node.get("name", ""))
            if mstep:
                step = int(mstep.group(1))
            a = _attrs(node)
            dur = node.get("duration_micros")
            dur = dur * 1e-6 if dur is not None else None
            size = float(a.get("comm_size", 0))
            if ntype == "COMM_SEND_NODE":
                b.add(rank, int(a["comm_dst"]), size, 1, step, "send", dur)
            elif ntype == "COMM_COLL_NODE":
                ct = a.get("comm_type")
                op = COLL_TYPES.get(ct, str(ct)) if isinstance(ct, int) else str(ct)
                members = pg_members.get(str(a.get("pg_name", "")), world)
                expand(b, op, members, size, step, algo, a.get("root"), only_src=rank, duration=dur)
            # COMM_RECV_NODE ignored: every transfer is recorded on its sender
    t = b.build(n_ranks=n)
    t.meta.update(name=path.stem, source="chakra", collective_algo=algo)
    return t

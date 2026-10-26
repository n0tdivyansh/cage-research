"""Parser for NCCL debug logs.

Collect with:  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL NCCL_DEBUG_FILE=nccl.%h.%p.log

STATUS: tested on the hand-written fixture data/fixtures/nccl/. NCCL log wording changes across
versions; check the regexes against your logs first.

Lines used
  init:  "<host>:<pid>:<tid> [<dev>] NCCL INFO comm 0x55.. rank 3 nranks 8 cudaDev 3 ...
          commId 0x1a2b - Init COMPLETE"
  coll:  "<host>:<pid>:<tid> [<dev>] NCCL INFO AllReduce: opCount 5 sendbuff 0x.. recvbuff 0x..
          count 1048576 datatype 7 op 0 root 0 comm 0x55.. [nranks=8] stream 0x.."
  p2p :  "... NCCL INFO Send: opCount 9 ... count 4096 datatype 9 op 0 root 5 comm 0x.." (root=peer)

Mapping to global ranks: each process (host, pid) is assigned its rank in the largest
communicator it joined (the world). Communicator members are ordered by their rank in that
communicator. Each collective (commId, opCount) is expanded once.
Steps are not marked in NCCL logs; pass `ops_per_step` if you know the number of collectives
per iteration on the world communicator, else everything is step 0.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..trace import CommTrace, TraceBuilder
from ._expand import expand

DTYPE_BYTES = {0: 1, 1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 1, 11: 1}
PREFIX = r"(?P<host>[^\s:]+):(?P<pid>\d+):\d+\s+\[\d+\]\s+NCCL INFO\s+"
INIT_RE = re.compile(PREFIX + r"comm (?P<comm>0x[0-9a-f]+) rank (?P<rank>\d+) nranks (?P<n>\d+)"
                     r".*?commId (?P<cid>0x[0-9a-f]+)", re.I)
COLL_RE = re.compile(PREFIX + r"(?P<op>AllReduce|AllGather|ReduceScatter|Broadcast|Reduce|Send|Recv)"
                     r": opCount (?P<opc>[0-9a-fx]+).*?count (?P<count>\d+) datatype (?P<dt>\d+)"
                     r".*?root (?P<root>-?\d+) comm (?P<comm>0x[0-9a-f]+)", re.I)


def load_nccl_log(path: str | Path, algo: str = "ring", ops_per_step: int | None = None,
                  n_ranks: int | None = None) -> CommTrace:
    path = Path(path)
    files = sorted(path.glob("*.log")) if path.is_dir() else [path]
    lines = [ln for f in files for ln in f.read_text(errors="replace").splitlines()]
    comms = {}                                 # (proc, comm_ptr) -> (cid, comm_rank, nranks)
    members = defaultdict(dict)                # cid -> {comm_rank: proc}
    for ln in lines:
        m = INIT_RE.search(ln)
        if m:
            proc = (m["host"], m["pid"])
            comms[(proc, m["comm"])] = (m["cid"], int(m["rank"]), int(m["n"]))
            members[m["cid"]][int(m["rank"])] = proc
    if not comms:
        raise ValueError("no 'Init COMPLETE' lines found; set NCCL_DEBUG_SUBSYS=INIT,COLL")
    world_cid = max(members, key=lambda c: len(members[c]))
    proc_rank = {proc: r for r, proc in members[world_cid].items()}
    n = n_ranks or len(proc_rank)
    b = TraceBuilder(n)
    seen = set()
    world_ops = 0
    for ln in lines:
        m = COLL_RE.search(ln)
        if not m:
            continue
        proc = (m["host"], m["pid"])
        key = (proc, m["comm"])
        if key not in comms or proc not in proc_rank:
            continue
        cid, crank, nr = comms[key]
        op = m["op"].lower()
        nbytes = int(m["count"]) * DTYPE_BYTES.get(int(m["dt"]), 1)
        glob_members = [proc_rank[members[cid][r]] for r in sorted(members[cid])]
        if op == "send":
            peer = glob_members[int(m["root"])]
            step = world_ops // ops_per_step if ops_per_step else 0
            b.add(proc_rank[proc], peer, nbytes, 1, step, "send")
            continue
        if op == "recv":
            continue
        ident = (cid, m["opc"])
        if ident in seen:
            continue
        seen.add(ident)
        if cid == world_cid:
            world_ops += 1
        step = (world_ops - 1) // ops_per_step if ops_per_step else 0
        root = int(m["root"])
        root_g = glob_members[root] if op in ("broadcast", "reduce") and 0 <= root < len(glob_members) else None
        expand(b, op, glob_members, nbytes, step, algo, root_g)
    t = b.build(n_ranks=n)
    t.meta.update(name=path.stem, source="nccl_log", collective_algo=algo,
                  assumptions=[f"NCCL algorithm assumed '{algo}' (logs at INFO do not say which ran)"])
    return t

"""Communication trace data model.

Design decisions (see docs/design_decisions.md):
- The primary record is an EVENT TABLE, not a graph. Every event keeps its step index, so the
  static aggregate graph (Paper 1) and per-step snapshots (I-GAPE, Paper 4) come from the same data.
- Collectives are expanded to point-to-point edges by the algorithm that actually ran
  (see collectives.py). The originating op name is kept per event, so a different expansion can be
  substituted later without re-collecting traces.
- Optional per-event duration (seconds) is kept when the source provides it. Byte volume is not the
  same as time; time-weighted analysis needs durations.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass
class CommTrace:
    n_ranks: int
    src: np.ndarray
    dst: np.ndarray
    nbytes: np.ndarray
    msgs: np.ndarray
    step: np.ndarray
    op: np.ndarray                      # index into op_names
    op_names: list[str]
    duration: np.ndarray | None = None  # seconds per event, NaN where unknown
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.src.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.step.max()) + 1 if len(self) else 0

    @property
    def total_bytes(self) -> float:
        return float(self.nbytes.sum())

    def _mask(self, steps: Sequence[int] | None, ops: Sequence[str] | None) -> np.ndarray:
        m = np.ones(len(self), dtype=bool)
        if steps is not None:
            m &= np.isin(self.step, np.asarray(list(steps)))
        if ops is not None:
            ids = [self.op_names.index(o) for o in ops if o in self.op_names]
            m &= np.isin(self.op, np.asarray(ids, dtype=self.op.dtype))
        return m

    # ------------------------------------------------------------- aggregation
    def matrix(self, weight: str = "bytes", steps: Sequence[int] | None = None,
               ops: Sequence[str] | None = None, per_step: bool = False) -> np.ndarray:
        """Dense directed traffic matrix M[i, j] = traffic i -> j.

        weight: 'bytes' | 'msgs' | 'duration'
        per_step: divide by the number of steps included (average per step).
        """
        m = self._mask(steps, ops)
        if weight == "bytes":
            w = self.nbytes[m]
        elif weight == "msgs":
            w = self.msgs[m].astype(float)
        elif weight == "duration":
            if self.duration is None:
                raise ValueError("trace has no durations")
            w = np.nan_to_num(self.duration[m])
        else:
            raise ValueError(weight)
        M = sp.coo_matrix((w, (self.src[m], self.dst[m])),
                          shape=(self.n_ranks, self.n_ranks)).toarray()
        if per_step:
            n = len(np.unique(self.step[m])) if m.any() else 1
            M = M / max(n, 1)
        return M

    def step_matrices(self, weight: str = "bytes") -> Iterable[tuple[int, np.ndarray]]:
        for s in np.unique(self.step):
            yield int(s), self.matrix(weight, steps=[int(s)])

    def graph(self, steps: Sequence[int] | None = None, per_step: bool = True) -> nx.DiGraph:
        """Directed NetworkX graph. Node set always contains every rank (isolated ranks are kept).

        Edge attrs: bytes, msgs. Node attrs: anything in meta['node_attrs'][rank].
        Extension points for later papers: edge attr 'link_class' (T-GAPE), node attr
        'data_objects' (C-GAPE). CAGE does not populate them.
        """
        B = self.matrix("bytes", steps=steps, per_step=per_step)
        C = self.matrix("msgs", steps=steps, per_step=per_step)
        G = nx.DiGraph()
        attrs = self.meta.get("node_attrs", {})
        for r in range(self.n_ranks):
            G.add_node(r, **attrs.get(r, attrs.get(str(r), {})))
        ii, jj = np.nonzero(B)
        for i, j in zip(ii.tolist(), jj.tolist()):
            G.add_edge(i, j, bytes=float(B[i, j]), msgs=float(C[i, j]))
        G.graph.update({k: v for k, v in self.meta.items() if k != "node_attrs"})
        return G

    # -------------------------------------------------------------- transforms
    def relabel(self, perm: np.ndarray) -> "CommTrace":
        """Rename rank r -> perm[r]. Used to model a launcher whose rank order does not
        match the communication structure (e.g. MoE expert assignment)."""
        perm = np.asarray(perm)
        meta = dict(self.meta)
        if "labels" in meta:
            new_labels = {}
            for k, arr in meta["labels"].items():
                arr = np.asarray(arr)
                out = np.empty_like(arr)
                out[perm] = arr
                new_labels[k] = out
            meta["labels"] = new_labels
        if "collectives" in meta:
            meta["collectives"] = [{**c, "members": [int(perm[x]) for x in c["members"]]}
                                   for c in meta["collectives"]]
        return CommTrace(self.n_ranks, perm[self.src], perm[self.dst], self.nbytes.copy(),
                         self.msgs.copy(), self.step.copy(), self.op.copy(), list(self.op_names),
                         None if self.duration is None else self.duration.copy(), meta)

    def subset(self, steps: Sequence[int] | None = None, ops: Sequence[str] | None = None) -> "CommTrace":
        m = self._mask(steps, ops)
        return CommTrace(self.n_ranks, self.src[m], self.dst[m], self.nbytes[m], self.msgs[m],
                         self.step[m], self.op[m], list(self.op_names),
                         None if self.duration is None else self.duration[m], dict(self.meta))

    # ---------------------------------------------------------------------- io
    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["src", "dst", "bytes", "msgs", "step", "op", "duration_s"])
            dur = self.duration if self.duration is not None else np.full(len(self), np.nan)
            for i in range(len(self)):
                d = "" if np.isnan(dur[i]) else f"{dur[i]:.9g}"
                w.writerow([int(self.src[i]), int(self.dst[i]), f"{self.nbytes[i]:.17g}",
                            int(self.msgs[i]), int(self.step[i]), self.op_names[self.op[i]], d])
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({"n_ranks": self.n_ranks, **_jsonable(self.meta)}, indent=2))

    @classmethod
    def from_csv(cls, path: str | Path, n_ranks: int | None = None) -> "CommTrace":
        path = Path(path)
        b = TraceBuilder(n_ranks or 0)
        with path.open() as f:
            for row in csv.DictReader(f):
                d = row.get("duration_s") or ""
                b.add(int(row["src"]), int(row["dst"]), float(row["bytes"]),
                      msgs=int(float(row.get("msgs") or 1)), step=int(float(row.get("step") or 0)),
                      op=row.get("op") or "p2p", duration=float(d) if d else None)
        meta = {}
        mp = path.with_suffix(".meta.json")
        if mp.exists():
            meta = json.loads(mp.read_text())
            n_ranks = n_ranks or meta.pop("n_ranks", None)
            if "labels" in meta:
                meta["labels"] = {k: np.asarray(v) for k, v in meta["labels"].items()}
        t = b.build(n_ranks=n_ranks)
        t.meta.update(meta)
        return t


def concat(traces: Sequence[CommTrace]) -> CommTrace:
    """Concatenate traces in time: step indices of later traces are offset. Ops are re-indexed."""
    b = TraceBuilder(max(t.n_ranks for t in traces))
    offset = 0
    for t in traces:
        for oid, name in enumerate(t.op_names):
            k = t.op == oid
            if k.any():
                b.add_arrays(t.src[k], t.dst[k], t.nbytes[k], t.msgs[k], t.step[k] + offset, name,
                             None if t.duration is None else t.duration[k])
        offset += t.n_steps
    out = b.build()
    out.meta.update({k: v for k, v in traces[0].meta.items() if not k.startswith("collectives")})
    colls = [c for t in traces for c in t.meta.get("collectives", [])]
    if colls:
        out.meta["collectives"] = colls
        out.meta["collectives_steps"] = sum(t.n_steps for t in traces)
    out.meta["segments"] = [t.n_steps for t in traces]
    return out


class TraceBuilder:
    """Accumulates events; call build() once."""

    def __init__(self, n_ranks: int = 0):
        self.n_ranks = n_ranks
        self._src: list[np.ndarray] = []
        self._dst: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        self._m: list[np.ndarray] = []
        self._s: list[np.ndarray] = []
        self._o: list[np.ndarray] = []
        self._d: list[np.ndarray] = []
        self.op_names: list[str] = []
        self.meta: dict = {}
        self._coll: dict = {}

    def record_collective(self, op: str, members, size: float, kind: str = "allreduce",
                          repeat: int = 1) -> None:
        """Remember a library collective (e.g. NCCL ring) whose algorithm order the library
        re-derives from placement. Its expanded events are still added normally; this record lets
        placement.nccl_matrix() re-expand it for a candidate placement. Stored in
        meta['collectives'] as {op, kind, members, size, count}."""
        key = (op, kind, tuple(int(m) for m in members), float(size))
        self._coll[key] = self._coll.get(key, 0) + repeat

    def _op(self, name: str) -> int:
        if name not in self.op_names:
            self.op_names.append(name)
        return self.op_names.index(name)

    def add(self, src: int, dst: int, nbytes: float, msgs: int = 1, step: int = 0,
            op: str = "p2p", duration: float | None = None) -> None:
        if src == dst or nbytes <= 0:
            return
        self.add_arrays([src], [dst], [nbytes], [msgs], step, op,
                        None if duration is None else [duration])

    def add_arrays(self, src, dst, nbytes, msgs, step: int | Sequence[int], op: str,
                   duration=None) -> None:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        nb = np.asarray(nbytes, dtype=float)
        ms = np.broadcast_to(np.asarray(msgs, dtype=np.int64), src.shape)
        keep = (src != dst) & (nb > 0)
        if not keep.any():
            return
        st = np.broadcast_to(np.asarray(step, dtype=np.int64), src.shape)
        du = (np.full(src.shape, np.nan) if duration is None
              else np.broadcast_to(np.asarray(duration, dtype=float), src.shape))
        oid = self._op(op)
        self._src.append(src[keep]); self._dst.append(dst[keep]); self._b.append(nb[keep])
        self._m.append(ms[keep].copy()); self._s.append(st[keep].copy())
        self._o.append(np.full(int(keep.sum()), oid, dtype=np.int16)); self._d.append(du[keep].copy())

    def build(self, n_ranks: int | None = None) -> CommTrace:
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
        src, dst = cat(self._src, np.int64), cat(self._dst, np.int64)
        n = n_ranks or self.n_ranks or (int(max(src.max(), dst.max())) + 1 if len(src) else 0)
        dur = cat(self._d, float)
        has_dur = bool(len(dur)) and not np.all(np.isnan(dur))
        meta = dict(self.meta)
        if self._coll:
            meta["collectives"] = [{"op": k[0], "kind": k[1], "members": list(k[2]), "size": k[3],
                                    "count": c} for k, c in self._coll.items()]
            meta["collectives_steps"] = int(cat(self._s, np.int64).max()) + 1 if len(src) else 1
        return CommTrace(n, src, dst, cat(self._b, float), cat(self._m, np.int64),
                         cat(self._s, np.int64), cat(self._o, np.int16), list(self.op_names),
                         dur if has_dur else None, meta)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o

# Paper 3 — C-GAPE: data-access graphs for caching and data placement

**Correction to the roadmap.** The data-access graph is *not* implicit in the communication graph —
a comm trace has no object identity. C-GAPE needs its own extraction (embedding row/shard IDs,
parameter-server keys, dataloader shard reads, KV-cache blocks). `gape/cache.py::AccessTrace` is the model.

**Question.** Does anything derived from the rank × object access graph beat per-host frequency
caching (LFU) under the same cache budget?

## Findings (C1, scenario values)
- With block rank placement, graph-aware data homes improve remote-fetch share by ≤1.5 points over
  rr_home+LFU at every locality level. Not worth a paper alone.
- With rank placement chosen from the co-access graph *and* graph-aware homes, remote fetches fall from
  0.22 to 0.02 (affinity 0.9, cache 5% of table) and from 0.21 to 0.10 at affinity 0.5.
- The replication heuristic (+rep) changes nothing — LFU already captures shared hot objects.
- Thesis for C-GAPE: graph-aware caching pays only *jointly* with placement — which feeds GAPE.

## Open tasks
- Real access traces: DLRM / TorchRec embedding lookups, or a parameter-server key log.
- Object-size heterogeneity, cache eviction dynamics over time (current model is a static plan).
- Locality is a parameter here; measure whether real workloads have rank-level access locality.

"""CAGE — Communication Analysis Graph Engine (Paper 1 of the GAPE roadmap).

Pipeline:  trace (events)  ->  traffic matrix / graph  ->  metrics  ->  placement analysis.

Modules
-------
trace        CommTrace event table; aggregation to matrices / NetworkX graphs; per-step snapshots
collectives  Expansion of collectives (allreduce, allgather, alltoall, ...) into point-to-point edges
workloads    Synthetic workload generators with known ground truth
metrics      Load concentration, hotspots, Louvain + null-model modularity, betweenness (experimental)
partition    Capacity-constrained partitioning (METIS + exact-capacity repair + swap refinement) and baselines
placement    Cluster model, placement evaluation, bandwidth cost model
extract      Parsers for edge CSV, Chakra execution traces (JSON), NCCL debug logs, CAGE PMPI shim output
report       JSON / Markdown report for a single trace
viz          Plots
"""

__version__ = "0.1.0"

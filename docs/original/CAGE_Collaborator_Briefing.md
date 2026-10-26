<!-- Original document, converted from CAGE_Collaborator_Briefing.docx (claude.ai Project "Research").
     Kept verbatim as the historical baseline. Several claims and numbers in it are wrong or
     unsupported — see docs/review_findings.md. Do not copy numbers from here into the paper. -->

**CAGE**

Communication Analysis Graph Engine

What It Is. What Problem It Solves. How It Solves It.

A Deep-Dive Briefing for Research Collaborators

Paper 1 of the GAPE Five-Paper Research Roadmap

# A Note to the Collaborator

This document exists for one purpose: to give you the complete picture of CAGE — what it is, why it matters, what we are going to build, and how we are going to build it — so that you and I can work on it together from a shared foundation.

You are joining a research program positioned at the intersection of distributed systems and graph analysis. These two fields rarely talk to each other as directly as they should. The core thesis is this: the communication behavior of distributed systems is inherently graph-structured, and the systems research community has left significant performance gains on the table by failing to treat it that way. CAGE is the first paper making that argument, and the first system proving it.

This is not a finished paper. It is a research program in motion — a set of problems, hypotheses, and a prototype we are building to test them. By the end of this document you should have genuine answers to:

- What specific, concrete problem are we solving, and for whom?
- What does CAGE actually do, technically, from end to end?
- How does each piece of CAGE address a specific pain point?
- What are we claiming that is novel, and how will we prove it?
- What is still open, and where does your thinking shape the direction?

Read this actively. Where you see gaps, have better framings, or disagree — those are the most valuable conversations we can have.

# 1. The Problem CAGE Solves

## 1.1 How Distributed Systems Actually Run

Picture a company training a large language model on a cluster of 512 GPUs spread across 64 machines. Each GPU processes a different slice of training data, computes local gradients, and then — critically — those gradients must be synchronized across all 512 workers before the next training step can begin. This synchronization is called AllReduce. It is pure communication: every GPU is sending its gradient tensor to every other GPU through the network.

On a well-configured cluster, AllReduce takes somewhere between 5% and 40% of total training time, depending on model size, batch size, and network hardware. At production scale — training jobs that run for days or weeks — if communication overhead is even 10% higher than necessary, that is hours of wasted GPU time per run and thousands of dollars per experiment. Multiply across dozens of experiments and you have a meaningful fraction of a research budget evaporating into avoidable network inefficiency.

Now ask: why is communication slow? The engineers running this cluster cannot easily answer that question. They know the job is slower than expected. They might know from coarse profiling that AllReduce is the culprit. But they cannot answer:

- Which specific workers are sending or receiving the most data?
- Are there workers that are bottlenecking the entire AllReduce because they handle disproportionate traffic?
- Would remapping which GPU runs which part of the job reduce cross-machine traffic?
- Do the 512 workers naturally form communication communities that should be co-located on the same machines?
- Is the communication load evenly distributed, or do a few processes carry the entire burden?

> **The Core Problem in One Sentence** The communication behavior of distributed systems has rich, analyzable structure — hotspots, communities, bottleneck bridges, load imbalances — but no existing tool exposes that structure in a systematic, graph-theoretic way. The result: operators are flying blind. They know something is slow. They don't know what, or why, or how to fix it in a principled way.

## 1.2 Why Existing Tools Fall Short

This is not a problem that has been neglected. There are profiling tools, monitoring systems, and performance analyzers for distributed systems. The issue is that none of them treat communication as a graph. They measure communication — but they discard the structural information.

| Tool | What It Measures | What Structure It Misses |
|---|---|---|
| mpiP / Score-P | Aggregate MPI call count and time per rank over the whole job | No graph of who communicates with whom. No community detection. No betweenness centrality. No placement comparison. |
| Nsight Systems / nvprof | GPU-level timeline: compute vs. communication phases per step | Which specific communication relationships cause slowdown. No task-level communication graph. |
| InfiniBand fabric counters | Bytes and messages per network port per second | Which application-level processes map to which ports. No task-to-task view. |
| PyTorch Profiler | Operator-level execution time including AllReduce duration | No breakdown by which ranks are slow. No communication topology. |
| SLURM / Kubernetes | Compute utilization per node; job queue state | Communication topology completely invisible. No communication-aware placement. |

Notice the pattern. Every existing tool either measures at the wrong granularity (fabric-level, not task-level), gives aggregate statistics that throw away structural information, or focuses entirely on compute while treating communication as a side note. None of them ask the graph question: 'What is the topology of communication in this system, and what does that topology tell us about performance?'

CAGE asks that question. CAGE builds the graph. CAGE analyzes it.

## 1.3 The Problem Is Universal, Not Specific to ML Training

Distributed ML training is the motivating example because it is high-profile and communication-intensive. But the same problem exists across every class of distributed computing:

- Distributed databases (Cassandra, CockroachDB, Spanner): Queries spanning multiple shards generate cross-node communication. If related data is on distant nodes, every cross-shard read pays a network round-trip. No tool systematically asks: 'Which shards communicate most, and are they co-located?'
- HPC simulations (molecular dynamics, climate modeling, CFD): Processes simulating adjacent spatial regions exchange boundary data every timestep. If neighboring-region processes are on distant nodes, every timestep pays a penalty. The communication graph has obvious structure — it mirrors the spatial adjacency graph — but schedulers ignore it.
- Stream processing (Apache Flink, Kafka Streams): Operators that are upstream/downstream of each other generate continuous directed data flow. Co-locating them eliminates serialization and network overhead. The operator graph is literally a communication graph, but it is not analyzed as one.
- Microservice architectures: Services that call each other frequently but are deployed across distant availability zones pay accumulated round-trip costs that compound into user-facing latency. No tool surfaces the community structure of service call graphs and suggests co-location.

In every case, the underlying structure is the same: there is a communication graph with hotspots, communities, and bottlenecks, and nobody is systematically looking at it. CAGE provides the methodology and tooling to change that.

# 2. What CAGE Is — The System We Are Building

## 2.1 The Central Idea

CAGE observes a running distributed system, captures how its processes communicate, represents those communication patterns as a weighted graph, and then applies graph-theoretic algorithms to that graph to surface structured, actionable insights. It makes the invisible architecture of distributed communication visible and analyzable.

More precisely: CAGE takes a distributed system's communication trace as input, constructs G = (V, E, W) where V is the set of processes, E is the set of communication relationships, and W is the weight function assigning communication volume to each edge — and then answers graph-theoretic questions about that graph that translate directly into performance insights and optimization recommendations.

## 2.2 The Three Layers of CAGE

> **Layer 1 — Extraction** Instruments or observes the distributed system and captures raw communication events. Input: A running distributed system — MPI job, distributed ML training, stream processor, etc. Output: A structured edge list: (source_process, dest_process, bytes, message_count, timestamp) for every communication relationship observed. Concretely: CAGE intercepts MPI calls at the PMPI profiling layer, or reads PyTorch DDP communication logs, or processes network-level traces — depending on what the system exposes. Research dimension: What is the right extraction granularity? Process-level gives a clean graph. Thread-level gives finer resolution but more noise. Network-level avoids application instrumentation but loses process-to-process semantics. What fidelity do we need to preserve the structure that community detection and centrality analysis rely on?

> **Layer 2 — Graph Construction and Analysis** Builds the weighted graph from the edge list and runs a battery of graph algorithms. Input: Raw edge list from Layer 1. Output: A fully analyzed graph with: weighted degree centrality per node, betweenness centrality per node, Louvain community membership per node, METIS partition assignment per node, load balance statistics for the full graph. Algorithms: NetworkX for graph construction and centrality; python-louvain for community detection; python-metis for partitioning. Research dimension: Which graph metrics are most predictive of actual performance bottlenecks? How does community structure quality (modularity Q) correlate with the benefit of communication-aware placement?

> **Layer 3 — Insight Generation and Recommendations** Translates graph-theoretic findings into concrete, operator-actionable recommendations. Input: Analyzed graph with all computed metrics from Layer 2. Output: Ranked hotspot process list with remediation suggestions; community-to-node co-location map; METIS-optimal placement assignment; load balance assessment with quantified Gini coefficient; bottleneck bridge node report. Research dimension: How do we close the loop? Can CAGE's placement recommendations be fed directly into a scheduler? What is the measured performance delta when they are applied?

## 2.3 What CAGE Concretely Produces

| Artifact | What It Contains and Why It Matters |
|---|---|
| Hotspot Report | Ranked list of processes by weighted degree centrality. Top 10% flagged as hotspots. For each: total bytes sent, total bytes received, top 5 communication partners, hotspot severity score. Immediately identifies which processes to investigate or rebalance. |
| Bridge Node Report | Processes with high betweenness centrality — coordination hubs whose overload disrupts global communication flow. Ranked with paths that flow through each bridge node visualized. Critical for fault tolerance planning and targeted optimization. |
| Community Map | Partition of all processes into communication communities via Louvain. For each community: member list, intra- vs. inter-community traffic ratios, modularity contribution. Co-location recommendations follow directly. |
| Placement Recommendation | METIS k-way partition assignment for all processes. Side-by-side comparison: edge cut cost under current placement vs. METIS-optimal, expressed as % reduction in cross-node traffic. This is the clearest performance number CAGE produces. |
| Load Balance Assessment | Gini coefficient and coefficient of variation of communication load. Degree distribution histogram. Under- and over-loaded process identification. A single score characterizing communication balance across the whole job. |
| Graph Visualization | Force-directed layout of the communication graph. Node size ∝ weighted degree centrality. Edge width ∝ communication volume. Node color = community membership. Makes the structure instantly visible to any reader. |

## 2.4 The Prototype: What We Will Actually Build

The CAGE prototype is a Python pipeline with five components:

- graph_extractor.py — Parses MPI trace files (mpiP format) or distributed training logs (PyTorch DDP JSON). Constructs a weighted NetworkX DiGraph. Nodes are process ranks. Edge weights: bytes_total, message_count, optional timestamps.
- cage_analyzer.py — Core analysis engine. Computes weighted degree centrality, betweenness centrality (using Brandes algorithm for efficiency), Louvain community detection, and METIS k-way partitioning. Outputs structured JSON report.
- placement_evaluator.py — Loads physical topology (rack / node / GPU hierarchy). Computes edge cut cost for current placement vs. METIS-optimal placement. Reports the delta and the specific process remappings required.
- visualizer.py — Generates graph visualizations using matplotlib + NetworkX drawing. Nodes sized by centrality, colored by community, edges weighted by volume. Produces both full graph and community-level summary.
- cage_cli.py — Command-line interface tying all components together. Usage: cage_cli.py --trace job.mpip --partitions 8. Output: full report directory with JSON analysis, PNG visualizations, and plain-English summary.

# 3. Problem-to-Solution: How CAGE Addresses Each Pain Point

## 3.1 Pain Point: 'I Don't Know Which Process Is My Bottleneck'

> **The Situation Without CAGE** You have a 128-worker distributed training job. The AllReduce step is slower than expected. Your profiling tool tells you: 'AllReduce accounts for 38% of step time across all ranks.' That is an aggregate number. It tells you communication is slow. It does not tell you which of the 128 workers is responsible. Is it one worker sending too much? One receiving too much? A handful of workers involved in too many communication relationships simultaneously? You don't know. You cannot fix what you cannot see.

> **How CAGE Solves It: Weighted Degree Centrality** CAGE computes weighted degree centrality for every process: the sum of all edge weights (bytes sent + received) for that node. This immediately surfaces processes handling disproportionate communication volume. Expected finding (Hypothesis H2): The top 15-20% of processes will handle 70-80% of total communication volume — a classic power-law concentration. This means a small number of targeted interventions can dramatically reduce overall communication overhead. CAGE output example: 'Worker 37: weighted degree = 2.4 GB/step (8.3x median)', 'Worker 37 communicates with 47 other workers (top 5 by volume: ranks 0, 12, 24, 36, 48)', 'Recommended: investigate data distribution skew or consider gradient compression for this rank.' This transforms 'AllReduce is slow' into a specific, named, quantified bottleneck with a direction for remediation.

## 3.2 Pain Point: 'I Don't Know What Would Break If One Node Goes Down'

> **The Situation Without CAGE** In large distributed jobs, some processes are critical coordination hubs — they relay data between groups of other processes. If one of these hubs is overloaded or fails, it doesn't just affect its direct neighbors. Communication flow across large portions of the system fragments. Operators have no way to identify these critical bridge processes ahead of time. Failures are reactive, not anticipated.

> **How CAGE Solves It: Betweenness Centrality** Betweenness centrality measures how often a node lies on the shortest communication path between all other pairs of nodes. A process with high betweenness is a 'communication bridge': many other process pairs route data through it, even if those processes don't directly call it. CAGE produces a bridge node ranking with two use cases: 1. Fault tolerance: bridge nodes should be monitored carefully, replicated if possible, or given dedicated resources. 2. Optimization: reducing load on high-betweenness nodes has outsized system-wide impact because they are on the critical path for so many communication flows. CAGE output example: 'Process 12: betweenness centrality = 0.34 (99th percentile)', 'Lies on shortest path for 34% of all process pairs', 'If overloaded, affects communication for approximately 4,300 process-pair interactions', 'Classification: Critical Coordination Hub. Recommend priority resource allocation.'

## 3.3 Pain Point: 'My Scheduler Places Tasks Without Knowing Communication Patterns'

> **The Situation Without CAGE** SLURM, Kubernetes, and PyTorch's DDP launcher all assign tasks based on compute availability — not communication affinity. The result: two workers that exchange 2 GB of gradient data every step might be placed on opposite ends of the rack, connected by a congested inter-rack InfiniBand link, while two workers that barely communicate are co-located on the same machine sharing a fast NVLink bus. This is entirely avoidable — but only if you know the communication structure before you place the job. Current tooling provides no mechanism to answer: 'What would the ideal placement look like, and how far is our current placement from it?'

> **How CAGE Solves It: METIS Graph Partitioning** CAGE runs METIS — the gold standard for graph partitioning — on the communication graph to compute the optimal assignment of processes to physical nodes. METIS divides the process graph into k balanced partitions (one per physical node) such that the total weight of edges crossing partition boundaries is minimized. Minimizing cross-partition edge weight = minimizing inter-node network traffic. CAGE then computes the concrete delta between current and optimal placement: Current placement cross-node traffic: X GB/step; METIS-optimal cross-node traffic: Y GB/step; Improvement: (X-Y)/X * 100%. Expected finding (Hypothesis H3): For workloads with strong community structure (Q > 0.4), METIS placement will reduce cross-node traffic by 20-40% compared to naive round-robin placement. This is the headline performance result of the CAGE paper.

## 3.4 Pain Point: 'I Don't Know Which Tasks Should Share a Machine'

> **The Situation Without CAGE** Even without running METIS, operators want intuition about which processes are 'natural neighbors' — communicating so densely that separating them is expensive. This is especially important on heterogeneous clusters: some nodes have fast local interconnects (NVLink, PCIe, shared memory) while others are connected only by the slower fabric. Ideally, you would place dense communication communities on nodes with fast local links, and reserve inter-node fabric for the sparse cross-community traffic. But without knowing the community structure of the communication graph, this optimization is guesswork.

> **How CAGE Solves It: Louvain Community Detection** CAGE runs the Louvain algorithm on the communication graph to detect communities — groups of processes with dense internal communication and sparse inter-community links. Modularity Q measures community structure quality: Q > 0.3: meaningful communities exist — co-location will help; Q > 0.6: strong communities — co-location could provide major benefit; Q < 0.1: near-random communication — co-location unlikely to help. A key insight: detected communities often correspond to logical structure the operator didn't explicitly model — pipeline stages, model parallel shards, data partition boundaries. CAGE surfaces these implicit groupings automatically. CAGE output example: 'Detected 4 communities (Q = 0.51 — strong structure)', 'Community A: workers 0-7, intra-community traffic = 78% of their total', 'Community A maps naturally to Machine 1 (8 GPUs). Recommend co-location.'

## 3.5 Pain Point: 'I Can't Tell If Communication Load Is Balanced'

> **The Situation Without CAGE** Compute load balance is measured everywhere: GPU utilization, CPU usage, memory pressure. If one GPU is at 90% and another at 10%, a scheduler can rebalance. Communication load balance is almost never measured. Some processes may handle 10x the network traffic of others — not because of more computation, but because of where they sit in the communication graph. This communication imbalance can completely dominate job latency even when compute is perfectly balanced. The slowest communication step determines when the next training step can begin — every worker waits for the last one.

> **How CAGE Solves It: Communication Load Distribution Analysis** CAGE computes the full weighted degree distribution and characterizes it statistically: Gini coefficient: 0 = perfectly equal load distribution; 1 = all traffic through one process. Coefficient of variation: standard deviation / mean of weighted degrees. Power-law test: does the degree distribution follow a power law (exponent)? Outlier identification: which specific processes are >2 standard deviations above mean? This produces a single interpretable score for communication load balance across the entire job, plus specific process-level diagnostics. CAGE output example: 'Communication Gini coefficient: 0.62 (moderate imbalance)', 'Top 15% of processes handle 78% of total traffic', 'Degree distribution follows power law with exponent 2.1', 'Processes 0, 8, 16, 24 are >3 sigma above mean — investigate AllReduce master assignment.'

# 4. The Research Argument — What Is Novel and How We Prove It

## 4.1 The Thesis

A research paper must make a claim that is non-obvious, meaningful, and empirically demonstrable. CAGE's claim is:

> **The CAGE Research Thesis** Communication patterns in real distributed systems exhibit graph-theoretic structure — specifically, community structure with high modularity, power-law degree distributions, and high-betweenness bridge nodes — that (a) is not captured or exposed by existing profiling and monitoring tools, (b) directly corresponds to measurable performance bottlenecks and inefficiencies, and (c) can be exploited by standard graph algorithms to generate placement and scheduling recommendations that meaningfully and quantifiably reduce cross-node communication overhead.

Each clause is a separate claim that must be demonstrated empirically. Let's unpack each one.

## 4.2 Claim 1: Real Communication Graphs Have Non-Trivial Structure (Q > 0.3)

It might seem obvious that a distributed training job has structure — it has pipeline stages, model layers, data shards. But it is not obvious that this structure manifests as measurable graph-theoretic properties that algorithms can detect and exploit.

Consider the null hypothesis: AllReduce is implemented as a symmetric ring or tree topology, which is close to random in its graph structure. Community detection would yield Q ≈ 0. Degree distribution would be near-uniform. METIS would offer no improvement over random placement. If this were true, CAGE's analysis would find nothing interesting.

Our expectation: real workloads are not uniform all-to-all communication. Pipeline parallelism creates directed chains. Hierarchical AllReduce creates hub-and-spoke patterns. Data parallelism with gradient compression creates asymmetric communication loads. These produce non-trivial graph structure. But this is a hypothesis — CAGE will measure it directly.

## 4.3 Claim 2: Existing Tools Miss This Structure

This is the 'gap' claim. The argument is categorical, not a matter of degree: no existing tool constructs a graph representation of communication behavior and applies graph algorithms to it.

mpiP tells you 'rank 5 spent 12% of its time in MPI_Send.' That is an aggregate performance statistic. CAGE tells you 'rank 5 has betweenness centrality 0.31, lies on the communication path between 28% of all process pairs, and is a critical bridge node in the parameter server subgraph.' These are categorically different levels of insight. The graph-level insight is not derivable from the aggregate statistics — you cannot compute betweenness centrality from total send times.

## 4.4 Claim 3: The Structure Corresponds to Real Bottlenecks (Validation)

This is the hardest claim to demonstrate and the most important. We need to show that the processes CAGE identifies as hotspots (high weighted degree centrality) actually correspond to processes that are empirically measured as bottlenecks in real performance experiments.

Validation strategy: (a) Inject artificial bottlenecks into a controlled job — deliberately overload a specific rank with extra gradient data — and verify CAGE surfaces that rank as a hotspot. (b) Run on real benchmarks (NAS Parallel Benchmarks, HPCG) with known structural properties and compare CAGE's community detection to the known communication patterns of those benchmarks. (c) Run on production-style distributed training traces and correlate CAGE's hotspot rankings with empirically observed slow steps.

## 4.5 Claim 4: CAGE Recommendations Improve Performance (The Headline Result)

The strongest claim: apply CAGE's METIS-optimal placement to a real job and measure the resulting performance improvement. We expect:

- 15-40% reduction in cross-node network traffic for workloads with Q > 0.4
- Measurable reduction in AllReduce step time when placement is optimized
- Larger improvement for jobs with stronger community structure (higher Q)

This closes the research loop: CAGE is not just an analysis tool — it is an optimization tool with demonstrable, quantified performance impact.

# 5. A Concrete Walkthrough — CAGE on a Real Scenario

## 5.1 The Scenario

32-worker data-parallel training job across 4 physical machines (8 GPUs per machine). The workers use hierarchical AllReduce: within each machine, the 8 GPUs do a local AllReduce over fast NVLink. Then one 'master' GPU per machine handles inter-machine synchronization over the slower InfiniBand fabric. The job is running slower than expected. The team suspects network congestion but doesn't know where.

## 5.2 Extraction Phase

CAGE instruments the job via PMPI (the standard MPI profiling hook). Every MPI_Send and MPI_Recv call is recorded: caller rank, partner rank, bytes, timestamp. After one training step: approximately 4,200 communication events. After aggregation: 87 unique communication relationships (edges) among 32 processes (nodes).

## 5.3 Graph Construction — Initial Observations

| | |
|---|---|
| Nodes (processes) | 32 |
| Unique edges | 87 (graph density 0.18 — moderately sparse) |
| Total comm. volume | ~12.4 GB per training step |
| Mean weighted degree | 387 MB per process per step |
| Max weighted degree | 2.1 GB (process 0) — 5.4x the mean |
| Min weighted degree | 38 MB (several isolated workers) |

Immediately: process 0 is an outlier at 5.4x the mean. Something is happening there.

## 5.4 Analysis Results

### Hotspot Detection (Weighted Degree Centrality)

Weighted degree centrality surfaces four outlier processes: 0, 8, 16, and 24 — with degrees of 2.1 GB, 1.9 GB, 1.8 GB, and 1.8 GB respectively, against a median of 310 MB. These four are the inter-machine AllReduce masters. Process 0 is more loaded than the others — suggesting it is doing additional intra-machine coordination.

### Community Detection (Louvain)

Louvain finds 4 communities with modularity Q = 0.51 — strong community structure. The communities map exactly to the 4 physical machines: workers 0-7, 8-15, 16-23, 24-31. Intra-community (intra-machine) traffic accounts for 78% of total volume. The 22% inter-community traffic is the expensive cross-fabric communication.

### Bridge Node Analysis (Betweenness Centrality)

Process 0 has betweenness centrality of 0.41 — placing it in the 99th percentile of all 32 processes. It lies on the shortest communication path for 41% of all process-pair combinations. This is not just a hotspot — it is a structural bottleneck. The hierarchical AllReduce implementation has made process 0 a hub that all cross-machine coordination routes through.

### METIS Placement Analysis

CAGE evaluates current placement against METIS-optimal. The current placement is reasonably good (it happens to align with the community structure) but CAGE identifies that workers 3 and 19 are frequent cross-community communicators (they share a data shard boundary) currently on different machines. Swapping worker 19 to machine 1 would reduce inter-machine traffic by 8%.

> **The Insight CAGE Found That Nothing Else Would Show** Without CAGE: 'The job is slow. AllReduce takes 38% of step time. Unknown root cause.' With CAGE: 'Process 0 has betweenness centrality 0.41 — it is on the critical path for 41% of all process-pair communication.', 'The hierarchical AllReduce is creating an artificial hub at rank 0.', 'This is not a placement problem — it is a communication topology design problem.', 'Recommendation: switch from master-based to ring-AllReduce topology across machines.', 'Expected improvement: distribute the 2.1 GB/step currently on rank 0 evenly across all 4 masters — reducing peak load by approximately 75%.' The first observation leads to guessing. The second leads to a specific, graph-theoretically grounded architectural fix.

# 6. CAGE in the Five-Paper Roadmap

## 6.1 Why CAGE Comes First

The research program has five papers. CAGE is first not because it is the simplest but because every subsequent paper requires what CAGE provides. T-GAPE, C-GAPE, and I-GAPE are each extensions of the same core idea — optimizing distributed systems using graph-theoretic analysis — but each needs a working communication graph extraction and analysis engine before it can ask its specific question.

T-GAPE asks: 'How should we map the communication graph onto the physical network topology?' That question is unanswerable without the communication graph. C-GAPE asks: 'Which data should be co-cached based on access patterns?' The access pattern graph is implicit in the communication graph CAGE extracts. I-GAPE asks: 'How stable is the communication graph across training iterations?' It needs sequences of CAGE-extracted graphs to compare.

CAGE is the substrate. Getting it right — the data structures, the extraction methods, the analysis pipeline, the codebase — pays compounding dividends across the entire program.

## 6.2 What Each Paper Inherits from CAGE

| Paper | System | New Research Question | Direct CAGE Dependency |
|---|---|---|---|
| Paper 2 | T-GAPE | How does the communication graph map onto physical network topology? What cross-layer placement minimizes topology-aware routing cost? | Weighted communication graph G. METIS partition baseline. Centrality metrics as inputs to topology overlay. |
| Paper 3 | C-GAPE | Which data objects are accessed together most? How to place data in cache hierarchy to minimize fetch latency? | Data dependency graph implicit in communication edges. Community detection methods for grouping data with its consumers. |
| Paper 4 | I-GAPE | How stable is the communication graph across iterations? When can we safely reuse a previous schedule? | Sequences of CAGE-extracted graphs. Graph similarity metrics. Structural fingerprinting methods built on CAGE's representation. |
| Paper 5 | GAPE | Can we build a unified execution engine making placement, caching, and scheduling decisions jointly in real time using communication graph state? | Everything. GAPE is the synthesis of all four prior papers. CAGE's graph representation is GAPE's core data structure. |

## 6.3 Designing CAGE with Downstream Papers in Mind

Because CAGE is the foundation, design decisions we make now will constrain or enable what is possible in Papers 2-5. Three specific implications:

- Graph data model: The NetworkX DiGraph we use in CAGE must be extensible. T-GAPE will need topology annotations on edges (rack-level, machine-level, GPU-level). C-GAPE will need data-access annotations on nodes. We should design the graph schema with these extensions in mind, even if CAGE itself doesn't populate them.
- Temporal representation: CAGE will likely start with static aggregate graphs. But I-GAPE needs temporal sequences — graphs across iterations. We should ensure CAGE's extraction layer can emit per-step snapshots, even if Paper 1 only analyzes the aggregate. A small design choice now avoids a full rewrite later.
- Code quality and modularity: The CAGE codebase is the codebase that T-GAPE, C-GAPE, I-GAPE, and GAPE all fork or extend. Clean interfaces, documented modules, and reproducible experiments in CAGE compound across the entire program's lifetime.

# 7. Open Questions — Where Your Input Shapes the Research

> **OQ 1: How to Represent Collective Operations (AllReduce, Broadcast)?** AllReduce is many-to-many. It doesn't fit cleanly into a pairwise edge model. Option A: Decompose into the point-to-point messages of the ring or tree implementation. Preserves pairwise graph. But models the implementation, not the semantic intent. Option B: Hypergraph — one hyperedge connecting all participants. Semantically accurate. Requires hypergraph algorithms (different from standard graph algorithms). Option C: Synthetic 'AllReduce node' with edges to all participants. Standard graph structure but an artificial abstraction. The choice affects which graph algorithms are applicable. Centrality and community detection are standard on pairwise graphs; less standard on hypergraphs. Question: Which representation best preserves the structure we care about for hotspot detection and placement optimization?

> **OQ 2: Static Aggregate vs. Dynamic / Per-Step Graph?** We can represent communication as: (a) one aggregate graph over the entire job, (b) a sequence of per-step graphs, or (c) a dynamic graph with evolving edge weights. Static aggregate is simplest and captures overall structure. Loses information about how communication changes across training phases (early training is often more exploratory; convergence phases often more stable). Per-step sequence is richer but 10x-100x more data and requires temporal graph analysis. For CAGE (Paper 1), we will likely start with static snapshots. But I-GAPE (Paper 4) needs the temporal representation. Question: How do we design CAGE's data model so upgrading from static to temporal doesn't require a full rewrite?

> **OQ 3: Extraction Overhead vs. Fidelity** Higher-fidelity extraction — capturing every message with size and timestamp — gives richer graphs but introduces runtime overhead. At 512 workers doing 1000 steps: potentially hundreds of millions of events. Storing and analyzing all of them is expensive. Sampling: capture every k-th step. Aggregation: store per-epoch summaries. Approximation: use fabric counters instead of MPI-level instrumentation. Question: What is the minimum extraction fidelity that preserves the graph structure CAGE needs to detect communities and hotspots accurately? This is an empirical question — the answer will significantly influence CAGE's practical deployability.

> **OQ 4: How Do We Validate Ground Truth?** How do we know CAGE's hotspot detection is correct? We need ground truth to validate against. Option A: Synthetic graphs with planted partition model communities and injected hotspots — ground truth known by construction. Option B: Real benchmarks with well-characterized communication patterns (NAS Parallel Benchmarks, HPCG). Option C: Inject artificial bottlenecks into a real job — overload a specific rank deliberately — and verify CAGE detects it. We likely need all three, in order. The design of these validation experiments will significantly affect how convincing the paper is. Question: What specific injection methodology for Option C is both realistic enough to be convincing and controlled enough to be unambiguous?

# 8. How We Work Together

## 8.1 What This Research Needs

CAGE sits at an uncommon intersection. Making it work well requires depth in both distributed systems (MPI, GPU clusters, AllReduce implementations, network topology) and graph analysis (community detection, centrality algorithms, partitioning methods). Neither of us needs to be an expert in both from day one — but together we need to cover the ground between us.

The research program is structured to build both kinds of knowledge in parallel. Phase 1 (now) is graph theory fundamentals. Phase 2 is distributed systems fundamentals. The prototype work in Phase 3 requires both, and that is when the two threads of knowledge converge into actual code.

## 8.2 The Phase Structure

| | |
|---|---|
| **Phase 1 — Now** | Graph theory foundations via CS224W (Stanford, Jure Leskovec): lectures 1.1-1.3, 2.1-2.3, 4.1-4.2, 13.1-13.3. Hands-on NetworkX implementation of a mock distributed training communication graph. Selective reading in Easley & Kleinberg (chapters 2, 3, 13). First experiments with METIS partitioning. |
| **Phase 2 — Next** | Distributed systems foundations: MPI programming model and profiling tools, distributed ML training mechanics (DDP, Horovod, pipeline parallelism), network topology and InfiniBand, trace collection and analysis. |
| **Phase 3 — Prototype** | End-to-end CAGE prototype. Real extraction from MPI traces and/or DDP logs. Full analysis pipeline. Experiments on NAS Parallel Benchmarks and mock distributed training. First performance measurements. |
| **Phase 4 — Paper** | Formalize research claims. Design and run all validation experiments. Write up findings. Prepare visualizations and evaluation for publication. |

## 8.3 Where to Start Together

The single most valuable thing we can do right now — before any algorithms — is build shared intuition about what communication graphs actually look like in practice.

Before running Louvain or METIS, we should both have a visceral, visual sense of: what does a 32-worker training job's communication graph actually look like when drawn? Is it sparse or dense? Are the clusters obvious to the eye? What do the degree distributions look like — normal, skewed, power-law? Does the graph have obvious hubs?

This motivates starting with the NetworkX mock graph experiment: build a synthetic 32-worker data-parallel training communication graph in Python (with realistic parameters — AllReduce ring topology, some injected imbalance), draw it, and examine its properties by hand before running any automated analysis. Theory is easier to internalize when you have already seen what the data looks like.

From that foundation, the CS224W lectures and the algorithm implementations will all feel grounded rather than abstract — because we will have already seen the kinds of graphs we are trying to analyze.

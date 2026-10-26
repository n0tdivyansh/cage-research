<!-- Original roadmap document, converted from CAGE_Overview.docx (claude.ai Project "Research").
     Kept verbatim as the historical baseline. Where it conflicts with docs/review_findings.md or
     docs/design_decisions.md, those win. -->

**CAGE**

Communication Analysis Graph Engine

Research Overview Document

Paper 1 of the GAPE Research Roadmap

Distributed Systems × Graph Analysis

# 1. Executive Overview

CAGE — the Communication Analysis Graph Engine — is a foundational research system that extracts, represents, and analyzes communication patterns in distributed computing environments as graphs. By modeling how processes, tasks, and nodes exchange data, CAGE enables principled identification of performance bottlenecks, load imbalances, and suboptimal task placements that would otherwise remain invisible to traditional profiling tools.

CAGE is the first paper in a five-paper research roadmap culminating in GAPE (Graph Analyzing and Processing Engine) — a unified, graph-aware distributed execution engine. Every subsequent paper in the roadmap (T-GAPE, C-GAPE, I-GAPE) directly builds on the graph extraction and analysis infrastructure established by CAGE. Getting CAGE right is therefore not just a paper deliverable — it is the architectural foundation upon which the entire research program stands.

> **Why CAGE Matters First** • Distributed system inefficiencies are fundamentally communication problems. • Communication patterns are inherently graph-structured: nodes are processes/tasks, edges are data flows. • Without a rigorous way to extract and analyze these graphs, topology-aware, cache-aware, and iteration-aware optimizations have no substrate to operate on. • CAGE provides that substrate.

# 2. Problem Statement

## 2.1 The Core Challenge

Modern distributed systems — training large ML models, running HPC simulations, executing distributed databases — involve thousands of communicating processes. Their performance is dominated not by raw computation, but by communication: when processes send data, how much, how often, and to whom.

Yet most distributed execution engines treat communication as a side-effect rather than a first-class citizen. Task schedulers assign work based on compute availability, not communication affinity. This leads to:

- Hotspot processes that send or receive far more data than peers, becoming bottlenecks.
- Load imbalance where communication costs are unevenly distributed across nodes.
- Cache-hostile placements where tasks that share data are mapped to distant nodes.
- Wasted bandwidth from unnecessary cross-rack or cross-datacenter communication.

## 2.2 The Missing Abstraction

The root cause is the absence of a systematic, graph-theoretic representation of communication behavior. Without this abstraction:

- Operators cannot easily visualize or reason about communication structure.
- Schedulers lack the information needed to make communication-aware placement decisions.
- Researchers cannot empirically characterize how communication patterns affect performance.

CAGE fills exactly this gap. It provides the vocabulary, tooling, and analytical methods to treat distributed system communication as a graph — and to extract actionable insights from that graph.

# 3. Core Concepts & Graph Foundations

## 3.1 What Is a Communication Graph?

A communication graph G = (V, E, W) is a weighted directed or undirected graph where:

| | |
|---|---|
| **V (vertices)** | Processes, tasks, workers, or compute nodes in the distributed system |
| **E (edges)** | Communication links — an edge (u, v) exists if node u sends data to node v |
| **W (weights)** | Edge weights represent communication volume: bytes transferred, message count, or bandwidth consumed |
| **Direction** | Directed if communication is asymmetric (A sends to B but not vice versa); undirected if symmetric |

This representation is powerful because it translates an operational question — 'who is talking to whom, and how much?' — into a mathematical object with centuries of analytical machinery available.

## 3.2 Key Graph Metrics CAGE Computes

### Degree Centrality

The degree of a node is the number of edges connected to it. In a weighted communication graph, weighted degree centrality measures the total communication volume a process handles. High weighted degree centrality signals a potential communication hotspot — a process that is a nexus of data flow and likely a bottleneck.

In-degree captures received data volume; out-degree captures sent data volume. Their asymmetry reveals directional imbalances.

### Betweenness Centrality

Betweenness centrality of a node v is the fraction of all shortest paths (between all pairs of nodes) that pass through v. A high betweenness node is a 'bridge' in the communication topology — removing or overloading it disproportionately disrupts the entire system.

In distributed systems, high-betweenness nodes are critical coordination points. Identifying them allows targeted optimization: replication, load shedding, or topological redesign.

### Community Detection

Community detection algorithms (e.g., Louvain, Girvan-Newman) partition the graph into densely-connected subgraphs with sparse inter-community edges. In communication graphs, communities correspond to groups of processes that communicate heavily with each other but rarely with outsiders.

This directly maps to placement opportunities: co-locate community members on the same rack, node, or NUMA domain to minimize expensive cross-boundary communication.

### Graph Partitioning (METIS)

Graph partitioning divides V into k balanced partitions such that the total weight of cut edges (edges crossing partition boundaries) is minimized. METIS is the canonical algorithm for this, using multilevel recursive bisection.

For CAGE, METIS-based partitioning directly informs task-to-node assignment: processes in the same partition should be co-located. Minimizing edge cuts minimizes inter-node communication overhead.

## 3.3 Graph Properties of Interest

| Property | What It Measures | Distributed System Implication |
|---|---|---|
| Density | Ratio of actual edges to possible edges | Dense graphs = all-to-all communication (e.g., AllReduce in ML training) |
| Diameter | Longest shortest path in the graph | High diameter = long communication chains, high latency sensitivity |
| Clustering Coefficient | How tightly-knit each node's neighbors are | High clustering = good locality, amenable to hierarchical placement |
| Power-law degree dist. | Few nodes with very high degree | Common in practice; signals hub-and-spoke communication patterns |
| Modularity (Q) | Quality of community structure | High Q = clear partitioning opportunities exist |

# 4. CAGE System Design

## 4.1 High-Level Architecture

CAGE is structured as a three-stage pipeline:

> **CAGE Pipeline** Stage 1 — Extraction: Instrument the distributed system to capture communication events (sends, receives, bytes transferred). Build the raw communication graph from this trace data. Stage 2 — Analysis: Apply graph algorithms to the extracted graph: compute centrality metrics, detect communities, run METIS partitioning, identify structural anomalies. Stage 3 — Insight Generation: Translate graph-theoretic findings into actionable recommendations for task placement, load rebalancing, and scheduling policy adjustments.

## 4.2 Graph Extraction

The extraction layer intercepts or observes inter-process communication. Concretely, CAGE can extract communication graphs from:

- MPI traces (mpiP, Score-P, TAU profiling frameworks) — capturing point-to-point and collective communication in HPC workloads.
- Distributed ML training logs (PyTorch DDP, Horovod, DeepSpeed) — capturing gradient synchronization and parameter server communication.
- Network-level traces (PCAP, RDMA counters, InfiniBand fabric statistics) — capturing actual packet-level communication.
- Application-level instrumentation — manually annotated send/receive operations for fine-grained control.

The extracted data is structured as an edge list with weights: (source_process_id, destination_process_id, bytes_transferred, message_count, timestamps). This is then ingested into NetworkX for analysis.

## 4.3 Analysis Layer

The analysis layer operates on the NetworkX graph object. The key analyses CAGE performs are:

| Analysis | Algorithm | Output |
|---|---|---|
| Hotspot Detection | Weighted degree centrality | Ranked list of highest-traffic processes; hotspot threshold percentile |
| Bridge Detection | Betweenness centrality | Critical coordination nodes whose failure or overload would fragment communication |
| Community Structure | Louvain method (greedy modularity) | Set of process communities; modularity score Q; inter/intra community traffic ratios |
| Optimal Partitioning | METIS (multilevel k-way) | k-partition assignment; edge cut cost; partition balance ratio |
| Load Balance | Degree distribution analysis | Gini coefficient of communication load; coefficient of variation across nodes |
| Structural Anomalies | Degree sequence, connectivity | Isolated nodes, disconnected components, unusually high/low communicators |

## 4.4 Prototype Implementation

The CAGE prototype is implemented in Python using NetworkX as the graph analysis backbone and METIS (via python-metis bindings) for partitioning. The prototype operates on mock distributed training communication graphs for initial validation, with the intent to graduate to real MPI/DDP traces.

Key implementation components:

- graph_extractor.py — parses trace data, constructs weighted NetworkX DiGraph
- cage_analyzer.py — implements the full analysis pipeline; outputs JSON reports
- visualizer.py — generates graph visualizations with community coloring and centrality-scaled node sizes
- metis_partitioner.py — wraps python-metis; computes optimal k-way partitions and reports cut costs

# 5. Expected Findings & Hypotheses

## 5.1 Primary Research Questions

- Do communication graphs exhibit community structure, and if so, how strongly (measured by modularity Q)?
- What fraction of processes account for the majority of communication volume (hotspot concentration)?
- How does current task placement compare to METIS-optimal placement in terms of edge cut cost?
- What is the relationship between betweenness centrality and observed system bottlenecks?
- How does communication graph structure vary across different workload types (training vs. inference vs. HPC simulation)?

## 5.2 Working Hypotheses

> **Hypothesis H1 — Community Structure Exists** Most real distributed workloads will exhibit significant community structure (Q > 0.3) because processes are typically organized into logical groups (pipeline stages, model layers, data shards) that communicate more within groups than across them. Implication: METIS partitioning aligned with community structure should substantially reduce cross-node traffic.

> **Hypothesis H2 — Hotspot Concentration (80/20 Rule)** A small fraction (~20%) of processes will account for the majority (~80%) of communication volume. This is consistent with power-law degree distributions observed in real networks. Implication: Targeted optimization of a small number of high-centrality nodes can yield disproportionate performance gains.

> **Hypothesis H3 — Placement Suboptimality** Naive round-robin or load-balanced task placement will have significantly higher edge cut costs than METIS-optimal placement, even when both achieve compute load balance. Implication: Communication-aware placement is necessary and will be quantifiably better.

# 6. CAGE in the GAPE Research Roadmap

## 6.1 The Five-Paper Arc

| Paper | System | Core Idea | Depends On |
|---|---|---|---|
| Paper 1 | CAGE | Extract and analyze communication graphs; identify hotspots, communities, optimal partitions | — |
| Paper 2 | T-GAPE | Map workload communication graphs onto physical network topology for placement/routing optimization | CAGE graph extraction |
| Paper 3 | C-GAPE | Analyze data access and dependency graphs to guide caching and data placement decisions | CAGE graph analysis methods |
| Paper 4 | I-GAPE | Detect stable, repeating graph structures across iterations; reuse scheduling decisions | CAGE + T-GAPE baseline |
| Paper 5 | GAPE | Unified graph-aware distributed execution engine synthesizing all prior insights | All prior papers |

## 6.2 What CAGE Provides to Each Downstream Paper

- T-GAPE consumes: the weighted communication graph G and METIS partition assignments. It then cross-references these against physical topology graphs to optimize placement.
- C-GAPE consumes: the data access patterns implicit in communication edges. It treats each send as evidence of data dependency and uses CAGE's community detection to group data with its primary consumers.
- I-GAPE consumes: sequences of CAGE-extracted graphs across training iterations. It computes graph similarity metrics (edit distance, spectral distance) to detect stability and trigger schedule reuse.
- GAPE synthesizes all four preceding analyses into a single runtime that makes placement, caching, and scheduling decisions jointly, using CAGE's graph model as the unified state representation.

# 7. Technical Prerequisites & Learning Roadmap

## 7.1 Graph Theory Foundations (Phase 1)

| Topic | Core Concepts | CS224W Coverage |
|---|---|---|
| Graph Basics | Types (directed, weighted, bipartite), adjacency representations, graph properties | Lectures 1.1, 1.2, 1.3 |
| Node Properties | Degree, degree distribution, clustering coefficient, local structure | Lectures 2.1, 2.2, 2.3 |
| Centrality Metrics | Degree, betweenness, closeness, PageRank, eigenvector centrality | Lectures 4.1, 4.2 |
| Community Detection | Modularity, Louvain algorithm, spectral clustering, graph cuts | Lectures 13.1, 13.2, 13.3 |
| Graph Partitioning | Minimum cut, balanced partitioning, METIS algorithm mechanics | Supplementary reading |

## 7.2 Distributed Systems Foundations (Phase 2)

- Communication primitives: MPI point-to-point (send/recv), collective operations (AllReduce, Broadcast, Scatter/Gather), and their performance characteristics.
- Distributed ML training: data parallelism, model parallelism, pipeline parallelism, and the communication patterns each generates.
- Network topology: fat-tree, torus, dragonfly topologies; bisection bandwidth; routing policies.
- Performance analysis: profiling tools (mpiP, nsight), roofline model, communication-to-computation ratio analysis.

## 7.3 Recommended Resources

| | |
|---|---|
| **CS224W (Stanford)** | Primary graph theory resource. Watch lectures 1.1–1.3, 2.1–2.3, 4.1–4.2, 13.1–13.3 first. GNN content (later lectures) is deferred to the GAPE phase. |
| **NetworkX (Python)** | Hands-on graph implementation. Build a mock distributed training communication graph and compute all centrality metrics and community structures. |
| **METIS** | Graph partitioning algorithm. Experiment with python-metis bindings on the NetworkX mock graph. Measure edge cut cost vs. naive partitioning. |
| **Easley & Kleinberg** | Reference text: Networks, Crowds, and Markets. Chapters 2 (Graph Theory), 3 (Strong/Weak Ties), and 13 (Structure of the Web) are directly relevant. |
| **MPI documentation** | For Phase 2: understand MPI_Send, MPI_Recv, MPI_Allreduce signatures and semantics. Focus on communication overhead, buffering, and collective algorithms. |

# 8. Experimental Design

## 8.1 Prototype Validation Experiments

### Experiment 1: Synthetic Graph Benchmark

Generate synthetic communication graphs with known properties (planted partition model for communities, power-law degree distributions for hotspots). Verify CAGE correctly recovers ground-truth communities, correctly ranks hotspot nodes, and produces METIS partitions with near-optimal edge cut costs.

### Experiment 2: Mock Distributed Training Graph

Construct a realistic mock communication graph simulating a distributed training job: 64 workers performing data-parallel training with AllReduce gradient synchronization. Analyze hotspot concentration, community structure (should correspond to pipeline stages or model shards), and compare METIS-suggested placement to round-robin placement.

### Experiment 3: Real MPI Trace Analysis

Obtain MPI communication traces from standard HPC benchmarks (NAS Parallel Benchmarks, HPCG). Apply CAGE to extract and analyze the communication graph. Compare CAGE-identified communities and hotspots to known structural properties of each benchmark.

## 8.2 Evaluation Metrics

| | |
|---|---|
| **Edge Cut Reduction** | % reduction in inter-partition communication under METIS placement vs. baseline (round-robin, bin-packing) |
| **Hotspot Accuracy** | Correlation between CAGE-identified high-centrality nodes and empirically observed bottlenecks |
| **Community Quality (Q)** | Modularity score of detected communities; comparison to ground truth in synthetic experiments |
| **Analysis Overhead** | Time and memory cost of CAGE analysis pipeline as a function of graph size (│V│, │E│) |
| **Coverage** | % of total communication volume captured by CAGE's top-k hotspot nodes |

# 9. Open Research Questions

> **Q1: Dynamic vs. Static Communication Graphs** Communication patterns in distributed training evolve across iterations (early training vs. convergence phase). Should CAGE produce a single aggregate graph, a sequence of snapshots, or a dynamic graph representation? This question directly feeds into I-GAPE (iteration-aware scheduling), which needs to reason about graph stability.

> **Q2: Graph Extraction Granularity** At what granularity should communication be captured? Process-level (coarse), thread-level (medium), or operation-level (fine)? Finer granularity gives richer graphs but introduces overhead and noise. What is the right trade-off for CAGE's use case?

> **Q3: Collective Communication Representation** AllReduce, Broadcast, and Gather operations involve many-to-many communication patterns. How should these be represented in a pairwise edge graph? As a hypergraph? As a ring/tree overlay structure? The choice of representation affects which graph algorithms are applicable and what insights can be extracted.

> **Q4: Online vs. Offline Analysis** Should CAGE operate offline (post-hoc trace analysis) or online (real-time graph updates and analysis during execution)? Online operation enables adaptive scheduling but introduces latency constraints on the analysis pipeline.

# 10. Glossary of Key Terms

| | |
|---|---|
| **AllReduce** | An MPI collective operation where all processes contribute values that are reduced (e.g., summed) and the result is distributed to all — the primary communication primitive in data-parallel ML training. |
| **Betweenness Centrality** | A graph metric measuring how often a node lies on the shortest path between other node pairs. High betweenness = critical bridge node. |
| **Communication Graph** | A weighted graph G = (V, E, W) where vertices are processes and edges represent data exchanges, weighted by volume or frequency. |
| **Community Detection** | Graph algorithms (e.g., Louvain) that partition vertices into groups based on edge density. Dense intra-community, sparse inter-community edges. |
| **Degree Centrality** | The number of edges connected to a node (or their total weight in weighted graphs). Measures communication volume handled by a process. |
| **Edge Cut** | In graph partitioning, the set of edges crossing partition boundaries. Minimizing edge cut = minimizing cross-partition communication. |
| **GAPE** | Graph Analyzing and Processing Engine — the unified long-term research vision synthesizing CAGE, T-GAPE, C-GAPE, and I-GAPE. |
| **Graph Partitioning** | Dividing graph vertices into k balanced groups minimizing inter-group edges. METIS is the standard algorithm. |
| **Hotspot** | A process or node with disproportionately high communication volume (high weighted degree centrality), likely a performance bottleneck. |
| **Louvain Algorithm** | A community detection algorithm that greedily maximizes modularity Q by merging nodes into communities. Fast and widely used. |
| **METIS** | A multilevel recursive bisection algorithm for graph partitioning. Produces high-quality balanced partitions with low edge cut costs. |
| **Modularity (Q)** | A scalar [−1, 1] measuring community structure quality. Q > 0.3 is generally considered meaningful community structure. |
| **NetworkX** | A Python library for graph creation, manipulation, and algorithm execution. CAGE's primary graph processing backend. |
| **T-GAPE / C-GAPE / I-GAPE** | Topology-Aware, Cache-Aware, and Iteration-Aware variants of GAPE — Papers 2, 3, and 4 in the roadmap, each building on CAGE. |
| **Weighted Degree Centrality** | Sum of edge weights for a node. In communication graphs: total bytes sent/received. Directly measures communication load. |

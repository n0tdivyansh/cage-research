# Paper 1 (CAGE): viva prep

Written 2026-11-02 against `draft/cage.pdf` and `results/01-cage/**` (full run 2026-10-26).
All numbers in the paper were checked against the results files and they match. `python -m pytest`: 31 passed, 1 skipped.

---

## 0. The 60-second pitch (memorise this)

> People have placed communicating processes close together since Bokhari in 1981, and MPIPP, TreeMatch
> and Scotch already build and partition communication graphs. We don't claim that's new. We ask a narrower
> question: for modern training jobs, when does graph-aware placement beat the placement the launcher
> already gives you? Three things make the obvious experiment misleading. Raw modularity is high even
> for graphs with no structure. Round-robin is a strawman, so we compare against block placement. And NCCL
> rebuilds its rings when ranks move. We built a pipeline that handles all three. On synthetic models of
> DP, Megatron, MoE and stencils, the answer is: placement pays only when the traffic has structure that
> disagrees with rank order. For DP and default-order Megatron it pays nothing. Every number comes from a
> model, not hardware, and we list what real traces must confirm.

Say "scenario values" before an examiner does. Don't wait to be caught.

---

## 1. The three attacks most likely to hurt, with answers

### A1. "Your decision rule doesn't need the graph analysis. It's just 'run METIS and see if it beats block.'"
**This is the paper's weakest logical point. Prepare it first.** Your own results say:
- Significance is **not sufficient**. Megatron in default order has z = 10.2 and 0% gain.
- Significance is **not necessary**. MoE with a = 0.6 has z = 2.5 (fails z > 3) and a −56.3% gain.
- Section V-D says: "it is not a gate that placement must pass."

So the title question ("when does structure pay off") gets the answer "when the partition disagrees with
rank order", and you compute that answer by running the partitioner, not by running the community test.

**Answer:** "Correct, and we say so in V-D. The graph analysis has two separate jobs. The partition-vs-block
comparison under the NCCL model decides placement. The null-model test is a guard against *reporting*
structure that isn't there, which E2 shows a fixed Q threshold gets wrong. The finding that significance
doesn't predict benefit is a result, not an oversight: it tells practitioners which of the two numbers to
look at." Don't argue that modularity predicts placement benefit. Your data says it doesn't.

### A2. "The MoE z-values depend on your null model." ← **found during this prep; not in the paper**
For graphs denser than 80%, `_null_sample` (`cage/metrics.py:108`) falls back to **shuffling weights**.
Every MoE graph has density 1.00, so every MoE z in Tables II and III uses that fallback. Shuffling destroys
the strength (weighted degree) sequence, and weighted modularity's null term is built from strengths. I
re-ran the MoE rows against a null that keeps every strength exactly (4-cycle weight transfers, 20 samples;
script `papers/01-cage/null_check_dense.py`, ~1 min):

| MoE graph | Q | shuffle null (paper) | strength-preserving null |
|---|---|---|---|
| Zipf 1.0, no affinity | 0.224 | z = −0.1 | z = +18.4 |
| Zipf 1.5 | 0.216 | z = −5.0 | z = +9.8 |
| Zipf 1.0, a = 0.6 | 0.279 | z = +2.5 | z = +22.5 |

(Strength CV observed vs shuffled: 0.60 vs 0.25, 1.14 vs 0.30, 0.42 vs 0.28. The shuffle null removes hubs.)

What this means:
- The sign of z flips depending on the null. The E3 line "Q is *below* the null mean at Zipf 1.5 (z = −5.0)"
  and its hub explanation is probably an artifact of a null that has no hubs.
- The V-D line "the significance test can miss real structure (MoE a = 0.6)" is **null-dependent**. Under
  the stricter null the test does fire. It also fires on no-affinity MoE, which has no planted groups. The
  2 GB ring allreduce inside the MoE trace is probably what reads as structure.
- Neither null is clearly right for a complete weighted graph. That is the honest position.

**Before the viva, either** (a) add a sentence to Threats to Validity saying MoE z-values come from a
weight-shuffling null that doesn't preserve strengths and that their sign is null-dependent, **or**
(b) be ready to say it out loud. Don't defend z = −5.0 as a property of MoE traffic.

**Spoken answer:** "For dense graphs, degree-preserving swaps are impossible, so we fall back to weight
shuffling. That keeps the weight distribution but not the strength sequence. The MoE z-values are therefore
the least reliable numbers in the paper. A strength-preserving null changes their sign. The placement result
for MoE doesn't depend on this, because it comes from the partition comparison, not the significance test."
(This last sentence is true, and it connects back to A1.)

### A3. "There's no real data. Isn't this a simulation paper with a circular design?"
- E4's affinity sweep recovers a parameter you put into the generator. The paper concedes this ("in part it
  reads back the generator's parameter").
- E5's slow-NIC result (bytes can't see it) holds by construction, because bytes don't change.
- E6 (1% of steps is enough) holds by construction for a stationary generator.

**Answer:** "Yes, and we label every number a scenario value. The synthetic setup is what lets us know the
ground truth, which no real trace gives you. The contribution is not the size of the gains. It's (i) the
negative results: no gain for DP and default Megatron, and Q > 0.3 misfires; (ii) the regime distinction,
where fixed vs NCCL changes MoE from −13% to −56%; and (iii) the protocol. Section VI names the three
experiments that would confirm it." Then name them: NCCL/Chakra logs from a real multi-GPU job, NPB under
the PMPI shim, and ASTRA-sim/SimGrid replay.

Know which results are **non-trivial despite synthetic inputs**:
- A 64-ring has Q = 0.74 with z = −1.1. This is a property of modularity, not of your generator.
- Round-robin ties CAGE on MoE only because interleaving by r mod 8 happens to line up. Admit you chose the interleaving.
- Volume ×3 injection is *not* found in skewed MoE (0.06). Hot experts swamp it. This was not designed in.
- Fixed vs NCCL regimes disagree by 4× on MoE (−13.3% vs −56.3%).

---

## 2. Question bank by section

### Introduction / novelty
- **"What's new, given MPIPP/TreeMatch/Scotch?"** Not the partitioner. What's new: the question (vs the
  launcher default), the NCCL re-ringing regime, null-model discipline, and the negative results. Don't
  say "first tool to build communication graphs" (review_findings §2).
- **"Why is round-robin a strawman?"** Nobody launches DP or Megatron that way. SLURM block and torchrun
  put consecutive ranks together. Against RR you'd report +700% / +1657% "wins" that aren't real.
- **"Why 8 GPUs/host, 64 ranks?"** 8 is DGX/HGX-class. 64 is 8 hosts, large enough for nulls and small
  enough for 20 Louvain runs per graph. Larger scale is in E7 (timing only).

### Method
- **Ring allreduce per-edge bytes 2(p−1)S/p.** Reduce-scatter plus all-gather, each (p−1) steps of S/p.
- **Eq. 2, swap gain.** Moving i from a to b gains D_ib − D_ia. Moving j gains D_ja − D_jb. W_ij is counted
  in both D_ib and D_ja, but the i–j edge stays cut after the swap, so subtract 2W_ij. This is Kernighan–Lin.
- **Clique weight 8e/p².** An even split of a p-member clique cuts (p/2)² pairs, so (p²/4)·w = 2e gives
  w = 8e/p², which matches a ring cut in two places. *Weakness:* when NCCL re-rings by host, a ring spanning
  m hosts costs about m·e, which is not a clique cut for uneven splits. That's why candidates are re-scored
  under the true NCCL model. Say that.
- **"Never worse than block."** Under the cut objective, because `refine_block` starts from block and only
  makes improving swaps. In the NCCL regime, block itself is a candidate scored under the NCCL model.
- **p = (1 + #)/(1 + N), N = 20.** The minimum attainable p is 1/21 ≈ 0.048. Many rows sit at exactly that
  floor, so p carries no extra information there. z does the work. *Expect:* "20 samples gives a noisy σ."
  Answer: z values are 6–17 for the significant cases, so the conclusions survive σ error. The borderline
  cases (MoE 2.5, sparse random 2.1) are exactly where you should hedge.
- **Hungarian relabelling.** Part labels are arbitrary. Hungarian matching on the overlap with the current
  placement minimises ranks moved.
- **Cost model (Eq. 3).** It's max over hosts of NIC time + intra time + latency. It ignores overlap,
  dependencies and congestion, so it ranks placements and doesn't predict seconds. Modelled time gains are
  an upper bound.
- **"Why bf16 / 2 GB gradients?"** These are generator parameters. Results are about relative bytes, so the
  constants only scale.

### Results
- **E2.** Ring Q = 0.74 vs null 0.756: the null graphs are *more* modular than the ring. Rings, lattices and
  sparse random graphs all reach high Q (Guimerà et al. 2004).
- **E1.** The detection limit is between μ = 0.6 (z = 3.3) and 0.75 (z = 0.6). With 8 equal blocks, μ = 7/8
  ≈ 0.875 means no structure at all, so 0.6–0.75 is near the theoretical detectability edge. ARI falls
  faster than NMI because ARI penalises any imbalanced split under exact capacity.
- **E3.** H2 (80/20) is refuted everywhere. The maximum top-20% share is 0.42. DP Gini is 0 by design.
- **E4, MoE a = 0.** Bytes −0.1%, time −14.9%, 46 ranks moved. That's load balancing between NICs, not
  structure. *Expect:* "You'd move 46 of 64 ranks for a model-only gain?" Answer: yes, which is why the
  paper calls it a side effect and not a headline. A real deployment would need a moved-ranks penalty
  (Paper 5's migration gate).
- **E4, fixed vs NCCL.** The fixed regime pins the dense ring in rank order, so every scattering placement
  pays for it. NCCL would re-ring. GPU jobs therefore use the NCCL headline (repo rule E3).
- **E5.** NIC at 25%: bytes are at chance (0.12 vs 1/8) and time-weighted strength is 1.00/1.00/0.75. The
  stencil's 0.75 is lower because neighbours of the slowed host also accumulate time. Check
  `e5_detection_validity` if asked for the exact mechanism. Don't improvise one.
- **E6.** Kendall τ is 0.945 at 1% of steps and top-10 overlap is 0.818 from 10% onward. Placement quality
  stays at 1.0 because the generator is stationary. Say "stationary" every time.
- **E7.** At 1024 ranks, swap refinement takes 10.7 s, against Louvain 0.06 s and METIS 0.21 s. Dense O(n²)
  per iteration. The fix is boundary-only swaps or incremental gains. *Spotted oddity:* betweenness looks
  faster at 512 (0.109 s) than at 256 (0.190 s) because exact betweenness runs up to 256 and k = 64 pivot
  sampling above. Know this if someone reads the table.

### Threats / ethics (Research Methodology course: expect at least one)
- **"How do we know expectations were written before running (rule E6)?"** Be careful. The git history
  doesn't show it: commit `ff3325a` adds the experiment scripts and their results together. Say the
  docstrings record expectations and misses are reported (H2, the MoE significance miss). Don't claim
  verifiable pre-registration.
- **Reproducibility.** Fixed seeds, `cage run --paper 1`, every number traced to `results/**`, and tests
  enforcing invariants (e.g. Σ strength = 2 Σ bytes).
- **Authorship.** Who did what? Agree on this with Aniruddha and Ashutosh *before* the viva, so the three
  of you give the same answer.
- **AI assistance.** If the course or venue requires disclosure of AI tools used for code or writing,
  check what you've declared. Examiners in a research-ethics course may ask directly.

---

## 3. Numbers to have cold

| | |
|---|---|
| Ring-64 | Q 0.74, null 0.756, z −1.1 |
| Megatron 8·4·2 | Q 0.82, z 10.2, 0% gain in default order |
| Scrambled Megatron | −92.7% bytes, −78.7% time, 48 moved |
| Stencil 2D / 3D | −42.9% / −25.0% bytes |
| MoE a = 0.6 | −56.3% (NCCL) vs −13.3% (fixed) |
| Affinity sweep | −18.8 / −37.5 / −56.3 / −75.1% at a = 0.2/0.4/0.6/0.8 |
| RR blow-ups | +700% hier-DP, +1657% Megatron |
| NIC test | bytes 0.12 (chance 0.125); time-weighted 1.00/1.00/0.75 |
| E1 limit | z 3.3 at μ 0.6 → 0.6 at μ 0.75 |
| E7 | 10.7 s refinement at 1024 ranks (i5-13420H) |

## 4. Don't say
- "CAGE speeds up training by X%." No hardware run exists.
- "Modularity shows MoE has no structure." That depends on the null (A2).
- "Betweenness identifies bottlenecks." It's experimental with no physical meaning (rule E4).
- "No existing tool builds communication graphs."
- "The parsers work on NCCL logs." They've been tested on hand-written fixtures only.

## 5. If you have a day before the viva
1. Add one sentence on the dense-graph null to Section VI (A2). This is the only substantive gap I found.
2. Rehearse A1 out loud. It is the question a sharp examiner asks.
3. If any 2-GPU box is available (Kaggle T4×2), run one NCCL job with `NCCL_DEBUG=INFO` through
   `cage analyze --format nccl`. "We parsed one real log" is worth more than any synthetic sweep in
   the room.

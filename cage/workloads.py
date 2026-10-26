"""Synthetic workload generators.

Every generator returns a CommTrace whose meta carries:
  name, gpus_per_node, description, assumptions (list of modelling assumptions to disclose),
  labels: dict[str, np.ndarray]   ground-truth groupings per rank (stage, tp_group, data_group, ...)

These are MODELS of real workloads, not measurements. Results computed on them are scenario
values and must be labelled as such in the paper (epistemic rule E1 in CLAUDE.md).
Rank layout follows the default of the corresponding launcher (block placement, Megatron order)
so that "current placement" baselines are realistic, not strawmen.
"""
from __future__ import annotations

import numpy as np

from . import collectives as C
from .trace import CommTrace, TraceBuilder

BF16 = 2


def _hosts(world: int, gpn: int) -> list[list[int]]:
    return [list(range(h * gpn, min((h + 1) * gpn, world))) for h in range(-(-world // gpn))]


def _finish(b: TraceBuilder, world: int, name: str, gpn: int, desc: str,
            assumptions: list[str], labels: dict) -> CommTrace:
    t = b.build(n_ranks=world)
    t.meta.update(name=name, gpus_per_node=gpn, description=desc, assumptions=assumptions,
                  labels={k: np.asarray(v) for k, v in labels.items()})
    return t


# ----------------------------------------------------------------------------- data parallel
def dp_ring(world: int = 32, gpn: int = 8, grad_bytes: float = 2e9, steps: int = 4) -> CommTrace:
    b = TraceBuilder(world)
    ring = list(range(world))
    for s in range(steps):
        C.ring_allreduce(b, ring, grad_bytes, s)
        b.record_collective("allreduce", ring, grad_bytes)
    return _finish(b, world, "dp_ring", gpn,
                   f"Data parallel, flat ring allreduce of {grad_bytes:.2g} B gradients per step",
                   ["ring order = rank order (NCCL would build the ring from detected topology)"],
                   {"host": np.arange(world) // gpn})


def dp_hier(world: int = 32, gpn: int = 8, grad_bytes: float = 2e9, steps: int = 4,
            variant: str = "leader") -> CommTrace:
    b = TraceBuilder(world)
    hosts = _hosts(world, gpn)
    fn = C.hier_leader_allreduce if variant == "leader" else C.hier_rail_allreduce
    for s in range(steps):
        fn(b, hosts, grad_bytes, s)
    return _finish(b, world, f"dp_hier_{variant}", gpn,
                   f"Data parallel, hierarchical ({variant}) allreduce of {grad_bytes:.2g} B per step",
                   ["communication pattern is defined relative to hosts: communities equal hosts by "
                    "construction (circularity: do not report community recovery here as a finding)"],
                   {"host": np.arange(world) // gpn})


# ------------------------------------------------------------------------- 3D (Megatron)
def megatron_3d(tp: int = 8, pp: int = 4, dp: int = 4, gpn: int = 8, layers: int = 32,
                hidden: int = 8192, seq: int = 4096, micro_bs: int = 1, microbatches: int = 8,
                steps: int = 2) -> CommTrace:
    """Megatron-LM rank order: tp fastest, then dp, then pp.
    rank = pp_i*(dp*tp) + dp_i*tp + tp_i
    TP: 4 activation allreduces per layer per microbatch (2 fwd, 2 bwd), ring within TP group.
    PP: activation fwd + activation-grad bwd between adjacent stages, per microbatch.
    DP: ring allreduce of the rank's parameter shard (~12*L*h^2 params total, bf16 grads).
    """
    world = tp * pp * dp
    b = TraceBuilder(world)
    rank = lambda p_i, d_i, t_i: p_i * dp * tp + d_i * tp + t_i
    act = seq * micro_bs * hidden * BF16
    lps = layers // pp
    params = 12 * layers * hidden ** 2
    grad_shard = params / (tp * pp) * BF16
    stage = np.zeros(world, int); tpg = np.zeros(world, int); dpg = np.zeros(world, int)
    for p_i in range(pp):
        for d_i in range(dp):
            for t_i in range(tp):
                r = rank(p_i, d_i, t_i)
                stage[r] = p_i; tpg[r] = p_i * dp + d_i; dpg[r] = p_i * tp + t_i
    for s in range(steps):
        for p_i in range(pp):
            for d_i in range(dp):
                g = [rank(p_i, d_i, t) for t in range(tp)]
                C.ring_allreduce(b, g, act, s, op="tp_allreduce", repeat=4 * lps * microbatches)
                b.record_collective("tp_allreduce", g, act, repeat=4 * lps * microbatches)
        for p_i in range(pp - 1):
            for d_i in range(dp):
                for t_i in range(tp):
                    a, c = rank(p_i, d_i, t_i), rank(p_i + 1, d_i, t_i)
                    # Megatron sends the full activation from every TP rank unless
                    # scatter-gather optimisation is on; model the scatter-gather variant (act/tp).
                    b.add(a, c, act / tp * microbatches, microbatches, s, "pp_send")
                    b.add(c, a, act / tp * microbatches, microbatches, s, "pp_send")
        for p_i in range(pp):
            for t_i in range(tp):
                g = [rank(p_i, d, t_i) for d in range(dp)]
                C.ring_allreduce(b, g, grad_shard, s, op="dp_allreduce")
                b.record_collective("dp_allreduce", g, grad_shard)
    return _finish(b, world, "megatron_3d", gpn,
                   f"3D parallel TP={tp} PP={pp} DP={dp}, L={layers}, h={hidden}, seq={seq}",
                   ["no sequence parallelism, no distributed optimizer",
                    "pipeline uses scatter-gather (act/tp per TP rank)",
                    "rings in rank order inside each group"],
                   {"stage": stage, "tp_group": tpg, "dp_group": dpg,
                    "host": np.arange(world) // gpn})


# ------------------------------------------------------------------------------- MoE / EP
def moe(world: int = 64, gpn: int = 8, experts_per_rank: int = 2, tokens_per_rank: int = 8192,
        hidden: int = 4096, top_k: int = 2, moe_layers: int = 16, zipf: float = 1.0,
        affinity: float = 0.0, n_groups: int = 8, group_layout: str = "interleaved",
        step_noise: float = 0.15, dense_grad_bytes: float = 2e9, steps: int = 8,
        seed: int = 0) -> CommTrace:
    """Expert-parallel MoE across all ranks (EP = world).

    Routing model: each token picks top_k experts. Expert popularity follows a Zipf law over a
    random expert order (skew `zipf`, 0 = uniform), perturbed per step by log-normal noise.
    `affinity` in [0,1]: fraction of routing probability that a token from data group g places on
    experts hosted inside group g. 0 = no locality (the default and the honest null when data is
    shuffled uniformly across DP ranks). >0 models domain-sharded data feeding domain-specialised
    experts. Whether real systems show rank-level affinity is an EMPIRICAL QUESTION for real
    traces; this knob exists so experiments can show what happens on either side of it.

    group_layout: 'interleaved' -> group(r) = r % n_groups (groups straddle hosts under block
    placement); 'block' -> group(r) = r // (world/n_groups) (groups align with hosts).
    """
    rng = np.random.default_rng(seed)
    b = TraceBuilder(world)
    E = world * experts_per_rank
    host_of_expert = np.arange(E) // experts_per_rank
    pop = np.empty(E)
    pop[rng.permutation(E)] = 1.0 / np.arange(1, E + 1) ** zipf
    pop /= pop.sum()
    if group_layout == "interleaved":
        group = np.arange(world) % n_groups
    else:
        group = np.arange(world) // (world // n_groups)
    pref = np.zeros((n_groups, E))
    for g in range(n_groups):
        ex = np.isin(host_of_expert, np.nonzero(group == g)[0])
        pref[g, ex] = pop[ex] / pop[ex].sum()
    tok_bytes = hidden * BF16
    for s in range(steps):
        noisy = pop * np.exp(step_noise * rng.standard_normal(E))
        noisy /= noisy.sum()
        M = np.zeros((world, world))
        for r in range(world):
            probs = (1 - affinity) * noisy + affinity * pref[group[r]]
            counts = rng.multinomial(tokens_per_rank * top_k, probs / probs.sum())
            per_host = np.bincount(host_of_expert, weights=counts, minlength=world)
            M[r] += per_host * tok_bytes
        # dispatch + combine in fwd, the same two in bwd -> 2x each direction
        traffic = 2 * moe_layers * (M + M.T)
        np.fill_diagonal(traffic, 0)
        C.alltoall(b, range(world), traffic, s, op="ep_alltoall")
        C.ring_allreduce(b, list(range(world)), dense_grad_bytes, s, op="dp_allreduce")
        b.record_collective("dp_allreduce", range(world), dense_grad_bytes)
    return _finish(b, world, "moe", gpn,
                   f"MoE EP={world}, {E} experts, top-{top_k}, zipf={zipf}, affinity={affinity}, "
                   f"groups={n_groups} ({group_layout})",
                   ["capacity factor / token dropping not modelled",
                    "routing affinity is a parameter, not an observed property",
                    "dense gradients use a flat ring allreduce"],
                   {"data_group": group, "host": np.arange(world) // gpn,
                    "expert_load": np.bincount(host_of_expert, weights=pop, minlength=world)})


# ----------------------------------------------------------------------------------- HPC
def stencil(dims: tuple[int, ...] = (8, 8), gpn: int = 8, halo_bytes: float = 4e6,
            steps: int = 4, periodic: bool = False) -> CommTrace:
    """Halo exchange on a Cartesian process grid, row-major rank order (MPI_Cart default)."""
    dims = tuple(dims)
    world = int(np.prod(dims))
    b = TraceBuilder(world)
    coords = np.array(np.unravel_index(np.arange(world), dims)).T
    src, dst = [], []
    for r in range(world):
        for ax in range(len(dims)):
            for d in (-1, 1):
                c = coords[r].copy(); c[ax] += d
                if periodic:
                    c[ax] %= dims[ax]
                elif not (0 <= c[ax] < dims[ax]):
                    continue
                src.append(r); dst.append(int(np.ravel_multi_index(c, dims)))
    for s in range(steps):
        b.add_arrays(src, dst, np.full(len(src), halo_bytes), 1, s, "halo")
    labels = {f"coord{ax}": coords[:, ax] for ax in range(len(dims))}
    labels["host"] = np.arange(world) // gpn
    return _finish(b, world, f"stencil{len(dims)}d", gpn,
                   f"Halo exchange on {'x'.join(map(str, dims))} grid, {halo_bytes:.2g} B per face",
                   ["uniform halo size on every face"], labels)


# ------------------------------------------------------------------------- ground truth
def planted_partition(n: int = 64, k: int = 8, p_in: float = 0.6, p_out: float = 0.05,
                      w_in: float = 1e6, w_out: float = 1e6, w_sigma: float = 0.5,
                      steps: int = 1, gpn: int | None = None, seed: int = 0) -> CommTrace:
    """Stochastic block model with log-normal weights. Ground truth: labels['block']."""
    rng = np.random.default_rng(seed)
    block = np.arange(n) // (n // k)
    b = TraceBuilder(n)
    for s in range(steps):
        same = block[:, None] == block[None, :]
        P = np.where(same, p_in, p_out)
        A = np.triu(rng.random((n, n)) < P, 1)
        W = np.where(same, w_in, w_out) * rng.lognormal(0, w_sigma, (n, n))
        ii, jj = np.nonzero(A)
        w = W[ii, jj]
        b.add_arrays(np.concatenate([ii, jj]), np.concatenate([jj, ii]),
                     np.concatenate([w, w]) / 2, 1, s, "p2p")
    return _finish(b, n, "planted_partition", gpn or n // k,
                   f"SBM n={n} k={k} p_in={p_in} p_out={p_out}",
                   ["synthetic ground truth, no real workload"], {"block": block})


def moe_phases(n_phases: int = 4, steps_per_phase: int = 25, affinity: float = 0.6,
               seed: int = 0, **kw) -> CommTrace:
    """MoE whose routing regime changes at phase boundaries (new popularity order and new
    rank-to-group mapping each phase). Models curriculum / data-mixture shifts. Used by I-GAPE.
    Within a phase, steps differ only by routing noise."""
    from .trace import concat
    world = kw.get("world", 64)
    rng = np.random.default_rng(seed)
    parts = []
    for ph in range(n_phases):
        t = moe(affinity=affinity, steps=steps_per_phase, seed=seed * 1000 + ph, **kw)
        perm = rng.permutation(world) if ph > 0 else np.arange(world)
        parts.append(t.relabel(perm))
    out = concat(parts)
    out.meta.update(name="moe_phases", description=f"MoE with {n_phases} routing regimes of "
                    f"{steps_per_phase} steps (affinity={affinity})",
                    phase_boundaries=[steps_per_phase * i for i in range(1, n_phases)])
    out.meta["assumptions"] = list(out.meta.get("assumptions", [])) + [
        "regime changes are abrupt and modelled as a relabelling of ranks"]
    return out


REGISTRY = {
    "moe_phases": moe_phases,
    "dp_ring": dp_ring,
    "dp_hier": dp_hier,
    "megatron_3d": megatron_3d,
    "moe": moe,
    "stencil": stencil,
    "planted_partition": planted_partition,
}

"""C1 — Does the access graph beat per-host frequency counting? (Paper 3, C-GAPE)

Sharded embedding-table lookups, 64 ranks on 8 hosts (block placement), Zipf popularity, a
per-group locality knob (affinity). Policies share the same per-host cache budget.
The bar to clear is rr_home+lfu (graph-oblivious). If graph_home+lfu does not beat it by a
margin worth the complexity, C-GAPE's premise needs revising — that is a publishable result too.
"""
import numpy as np
import matplotlib.pyplot as plt

from cage import expkit as X, partition as P
from gape import cache as CA

d = X.outdir("03-cgape", "c1_cache_policies")
affs = [0.0, 0.25, 0.5, 0.75, 0.9] if not X.quick() else [0.0, 0.5, 0.9]
cache_fracs = [0.01, 0.05, 0.2] if not X.quick() else [0.05]
rows = []
for a in affs:
    tr = CA.embedding_workload(affinity=a, n_objects=2048 if X.quick() else 4096, seed=1)
    S = tr.reads @ tr.reads.T          # rank-rank co-access similarity
    np.fill_diagonal(S, 0)
    placements = {"block": P.block(tr.n_ranks, 8), "co-access": P.partition(S, 8, "best")}
    for pname, host in placements.items():
        for cf in cache_fracs:
            cache = cf * tr.obj_bytes.sum()
            for pol in CA.POLICIES:
                rows.append({"affinity": a, "cache_frac": cf, "rank_placement": pname,
                             **CA.evaluate(tr, host, 8, pol, cache)})
X.save_json(d, rows)
cf_plot = 0.05 if 0.05 in cache_fracs else cache_fracs[0]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
series = [("block", "rr_home+lfu"), ("block", "graph_home+lfu"), ("co-access", "rr_home+lfu"),
          ("co-access", "graph_home+lfu")]
for i, (pl, pol) in enumerate(series):
    r = [x for x in rows if x["policy"] == pol and x["cache_frac"] == cf_plot and x["rank_placement"] == pl]
    ax.plot([x["affinity"] for x in r], [x["remote_frac"] for x in r], marker="o", color=X.SERIES[i],
            label=f"{pl} ranks · {pol}")
ax.set(xlabel="per-group access locality (affinity)", ylabel="share of bytes fetched remotely",
       ylim=(0, None), title=f"Remote fetches, cache = {cf_plot:.0%} of table per host")
ax.legend(fontsize=8)
X.save_fig(fig, d, "remote_frac_vs_affinity")
X.write_summary(d, "C1 — Cache / data placement policies", X.md_table(
    rows, ["affinity", "cache_frac", "rank_placement", "policy", "remote_frac", "max_host_inbound", "max_host_serve"],
    {"affinity": "{:.2f}", "cache_frac": "{:.2f}", "remote_frac": "{:.3f}",
     "max_host_inbound": "{:.3g}", "max_host_serve": "{:.3g}"}))
print(open(d / "summary.md").read())

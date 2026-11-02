# Does the MoE z survive a strength-preserving null? (4-cycle weight transfer keeps every s_i exact)
import numpy as np
from cage import workloads as W
from cage.metrics import louvain, modularity_significance, _null_sample
def to_M(t):
    return t.matrix() if hasattr(t,'matrix') else t
def strength_null(Wu, rng, iters):
    P = Wu.copy(); n = len(P)
    for _ in range(iters):
        a,b,c,d = rng.choice(n,4,replace=False)
        # +δ on (a,b),(c,d); -δ on (a,d),(c,b)
        lo = -min(P[a,b],P[c,d]); hi = min(P[a,d],P[c,b])
        if hi-lo <= 0: continue
        dl = rng.uniform(lo,hi)
        for (x,y,s) in ((a,b,1),(c,d,1),(a,d,-1),(c,b,-1)):
            P[x,y]+=s*dl; P[y,x]+=s*dl
    return P
for name,kw in [("MoE zipf 1.0",dict(zipf=1.0)),("MoE zipf 1.5",dict(zipf=1.5)),("MoE zipf1 aff0.6",dict(zipf=1.0,affinity=0.6))]:
    tr = W.moe(**kw)
    M = tr.matrix(per_step=True)
    Wu = M + M.T
    base = modularity_significance(M, n_null=20)
    rng = np.random.default_rng(1); qs=[]
    sh = _null_sample(Wu, rng)
    for i in range(20):
        P = strength_null(Wu, rng, 20000)
        qs.append(louvain(np.triu(P), seed=i+1)[1])
    qs=np.array(qs)
    s0=Wu.sum(1); s_sh=sh.sum(1)
    print(f"{name}: Q={base['Q']:.3f} | shuffle-null mean {base['null_mean']:.3f} z={base['z']:.1f} "
          f"(strength CV obs {s0.std()/s0.mean():.2f} vs shuffled {s_sh.std()/s_sh.mean():.2f}) "
          f"| strength-null mean {qs.mean():.3f} z={(base['Q']-qs.mean())/qs.std(ddof=1):.1f} p={(1+(qs>=base['Q']).sum())/21:.3f}")

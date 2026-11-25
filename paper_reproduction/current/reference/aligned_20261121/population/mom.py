"""Matched ablations of the zero-anchor finite-pool population procedure."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("LEC_SHARED", HERE.parents[1]))
sys.path.insert(0, str(SHARED / "logistic_population_betting_20261115"))
import experiment as e
from kernel import sample
import numpy as np

p = e.pop
BASE = SHARED / "logistic_population_betting_revision_20261115"
ARMS = ("whole", "uniform", "cap")

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def cumulative(cache, kernel, whole):
    c, w, left, right = kernel
    first = left >= 0
    second = right >= 0
    if whole:
        objects = np.unique(np.r_[left[first], right[second]] // cache.r)
        cache.objects(objects, "comparison")
    product = np.ones(len(c))
    product[first] = 2*cache.query(left[first], "comparison") - 1
    product[second] *= 2*cache.query(right[second], "comparison") - 1
    return c+w*product

def fit_result(row, cache, final, a, b, case, gap):
    if row["choice"] != "unresolved":
        e.finish(row, cache, final, a, b, case)
    else:
        row.update(labels=cache.count, stage_labels=cache.stages.copy())
    L, U = row["interval"]
    row.update(acceptable=bool(row["choice"]!="unresolved" and
               (gap<=case["g"] if row["choice"]=="A" else gap>=-case["g"])),
               covered=bool(L<=gap<=U))
    assert sum(row["stage_labels"].values()) == row["labels"]
    row.pop("final_fit", None)
    return row

def fallback(caches, inputs, seed, group, ci):
    a,b,y,pools,case,zeros,par,template = inputs
    # The exact refit blocks are the same for every ablation of this path.
    # Their score sequence is independent of previous moment acquisitions.
    name=f"case_{ci:02d}_{group:02d}.json"
    archived = SHARED/"logistic_adaptive_initialization_20261115/population_results"/name
    ref = None
    if archived.exists():
        ref = next(r for r in read(archived)["records"] if r["method"]=="zero_anchor")
        if not ref["fallback"]: ref=None
    if ref:
        assert ref["seed"]==seed and abs(ref["refit_range"]-par["R"])<1e-12
        hist=ref["refit_history"]
        choice=ref["choice"]
    else:
        # Same RNG, Newton tolerance, and EB looks as the published fallback.
        shadow=p.p.Cache(y)
        choice, _, _, hist=p.p.fallback(a,b,pools,case,zeros,par["R"],shadow,seed)
    rng=np.random.default_rng(seed+6000000)
    for pos in range(0,hist[-1]["blocks"],64):
        train=rng.choice(pools["training"],(64,case["n"]))
        test=rng.choice(pools["comparison"],64)
        for cache in caches.values():
            cache.objects(train,"refit")
            cache.objects(test,"refit")
    return choice,hist[-1]["interval"],hist[-1]["blocks"]

def run_path(ci,group,inputs):
    a,b,y,pools,case,zeros,par,template = inputs
    path=HERE/"moments"/f"{ci}_{group}.json"
    if path.exists(): return
    name=f"case_{ci:02d}_{group:02d}"
    arc=read(BASE/"results"/(name+".json"))
    frozen=np.load(BASE/"draws"/(name+".npz"))["z"]
    lec=next(r for r in arc["records"] if r["method"]=="lec")
    assert abs(par["M"]-arc["range"])<1e-10
    M=par["M"]+1e-8
    g=case["g"]
    eps0=arc["epsilon0"]
    separate=arc["old_public_epsilon"]
    gap=lec["audit_gap"]
    seed=lec["seed"]
    final=np.random.default_rng(p.s.SEED+9000000+ci*100000+40000+group).choice(pools["training"],case["n"])
    assert p.s.decision(-M-eps0,M+eps0,g)=="unresolved"
    same_bound = abs(separate-eps0) < 1e-15
    arms = ARMS if same_bound else (*ARMS, "separate")
    cache={k:p.p.Cache(y) for k in arms}
    state={k:template.state() for k in arms}
    epsilon=dict.fromkeys(arms,eps0)
    if not same_bound:
        epsilon["separate"]=separate
    support=dict.fromkeys(arms,M)
    r=y.shape[1]
    support["uniform"]=(4*r*r+2*r)*M
    stream={k:[] for k in arms}
    running={k:[-M,M] for k in arms}
    results={}
    histories={k:[] for k in arms}
    previous=0
    for phase,end in enumerate(p.p.CHECKS):
        live=[k for k in arms if k not in results]
        known={k:cache[k].values.copy() for k in live}
        rng=np.random.default_rng(seed+4000000+phase*100000)
        for start in range(previous,end,128):
            bs=min(128,end-start)
            ids=rng.choice(pools["training"],(bs,case["n"]))
            tests=rng.choice(pools["comparison"],bs)
            A=p.s.fast_coefficients(a,ids,tests,zeros[0],case["lam"],par["tc"][0],par["dc"][0])
            B=p.s.fast_coefficients(b,ids,tests,zeros[1],case["lam"],par["tc"][1],par["dc"][1])
            c,v,w=[A[i]-B[i] for i in range(3)]
            nodes=np.stack([tests,ids[:,0],ids[:,1]],axis=1)
            keys=(nodes[:,:,None]*r+np.arange(r)).reshape(bs,-1)
            u=rng.random(bs)
            assert np.max(abs(c)+abs(v).sum(1)+abs(w).sum(1))<=M
            for k in live:
                draw=sample(c,v,w,A[3],A[4],keys,known[k],u,k=="uniform")
                vals=cumulative(cache[k],draw,k=="whole")
                assert np.max(abs(vals))<=support[k]+1e-12
                if k=="cap" and start<len(frozen):
                    stop=min(end, start+bs, len(frozen))
                    assert np.max(abs(vals[:stop-start]-frozen[start:stop]))<1e-10,(ci,group,start)
                stream[k].append(vals)
        previous=end
        for k in live:
            epsilon[k]=min(epsilon[k],template.update(state[k],cache[k].values,par,p.s,p.from_moments)[0])
            lo,hi=e.intervals.interval(np.concatenate(stream[k]),support[k],.02)
            guard=1e-9*(1+support[k])
            running[k]=[max(running[k][0],lo-guard),min(running[k][1],hi+guard)]
            L,U=running[k][0]-epsilon[k],running[k][1]+epsilon[k]
            choice=p.s.decision(L,U,g) if running[k][0]<=running[k][1] else "unresolved"
            histories[k].append(dict(n=end,labels=cache[k].count,epsilon=epsilon[k],interval=[L,U],choice=choice))
            if choice!="unresolved" and (k!="cap" or end==p.p.CHECKS[-1]):
                row=dict(method=k,interval=[L,U],choice=choice,fallback=False,n=end)
                results[k]=fit_result(row,cache[k],final,a,b,case,gap)
    if same_bound:
        results["separate"]=dict(method="separate",labels=lec["labels"],stage_labels=lec["stage_labels"],
            choice=lec["choice"],interval=lec["interval"],fallback=lec["fallback"],n=lec["stopping_blocks"],
            acceptable=lec["acceptable_audit_mean"],covered=lec["covered_audit_mean"],identical_reference=True)
    missing={k:v for k,v in cache.items() if k not in results}
    if missing:
        choice,iv,n=fallback(missing,inputs,seed,group,ci)
        for k,cc in missing.items():
            row=dict(method=k,choice=choice,interval=iv,fallback=True,n=previous,refit_n=n)
            results[k]=fit_result(row,cc,final,a,b,case,gap)
    for k in arms:results[k]["history"]=histories[k]
    path.write_text(json.dumps(dict(case_index=ci,group=group,seed=seed,epsilon0=eps0,separate0=separate,
        supports=support,matched_reference_blocks=len(frozen),records=list(results.values()))),encoding="utf8")
    print(ci,group,[(k,v["labels"],v["choice"]) for k,v in results.items()],flush=True)

def run_case(ci, groups):
    case=p.s.CASES[ci]
    a,o,y,pools=p.s.load_domain(case["domain"])
    b=(1-case["alpha"])*a+case["alpha"]*o
    zeros=[np.zeros(z.shape[-1]) for z in (a,b)]
    par=p.p.parameters(a,b,pools,zeros,case["lam"])
    template=p.BoundTemplate(a,b,pools,zeros,case)
    inputs=(a,b,y,pools,case,zeros,par,template)
    for group in groups:run_path(ci,group,inputs)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--cases",type=int,nargs="+",default=list(range(8)))
    ap.add_argument("--groups",type=int,nargs="+",default=list(range(6,18)))
    ap.add_argument("--groups-per-job",type=int,default=12)
    args=ap.parse_args()
    (HERE/"moments").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs=[pool.submit(run_case,ci,args.groups[start:start+args.groups_per_job])
              for start in range(0,len(args.groups),args.groups_per_job) for ci in args.cases]
        for job in as_completed(jobs):job.result()

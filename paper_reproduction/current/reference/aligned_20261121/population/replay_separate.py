"""Re-run query-first population separate-bound paths on the frozen study inputs.

Set LEC_SHARED to the experiments/shared_resource directory containing the aligned
study, its original reference implementations, and public features/label pools.
Optional LOGISTIC_DATA_ROOT relocates the frozen pools. Prints path-level checks.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("LEC_SHARED", HERE.parents[1])).resolve()
ARCHIVED_MOMENTS = SHARED / "aligned_20261121/population/moments"
import mom as original
import numpy as np

p = original.p
e = original.e
CODES = {3: "T4", 6: "V3"}


def run_path(ci, group, inputs, out):
    result_path = out / f"path_{ci}_{group}.json"
    if result_path.exists():
        return
    a, b, y, pools, case, zeros, par, template = inputs
    name = f"case_{ci:02d}_{group:02d}"
    arc = original.read(original.BASE / "results" / (name + ".json"))
    ref = next(r for r in arc["records"] if r["method"] == "lec")
    cap = next(r for r in original.read(ARCHIVED_MOMENTS / f"{ci}_{group}.json")["records"] if r["method"] == "cap")
    frozen = np.load(original.BASE / "draws" / (name + ".npz"))["z"]
    seed = arc["seed"]
    final = np.random.default_rng(p.s.SEED + 9000000 + ci*100000 + 40000 + group).choice(pools["training"], case["n"])
    g = case["g"]
    cache = p.p.Cache(y)
    state = template.state()
    epsilon = template.update(state, cache.values, par, p.s, p.from_moments)[0]
    assert abs(epsilon - arc["old_public_epsilon"]) < 1e-11 and epsilon >= g
    epsilon0 = epsilon
    M = par["M"] + 1e-8
    assert abs(par["M"] - arc["range"]) < 1e-11
    running = [-M, M]
    guard = 1e-9*(1+M)
    assert p.s.decision(-M-epsilon, M+epsilon, g) == "unresolved"
    vals = []
    history = []
    previous = 0
    r = cache.r
    for phase, end in enumerate(p.p.CHECKS):
        known = cache.values.copy()
        rng = np.random.default_rng(seed + 4000000 + phase*100000)
        for start in range(previous, end, 128):
            bs = min(128, end-start)
            ids = rng.choice(pools["training"], (bs, case["n"]))
            tests = rng.choice(pools["comparison"], bs)
            A = p.s.fast_coefficients(a, ids, tests, zeros[0], case["lam"], par["tc"][0], par["dc"][0])
            B = p.s.fast_coefficients(b, ids, tests, zeros[1], case["lam"], par["tc"][1], par["dc"][1])
            c, v, w = (A[j]-B[j] for j in range(3))
            keys = (np.stack([tests, ids[:,0], ids[:,1]], axis=1)[:,:,None]*r + np.arange(r)).reshape(bs, -1)
            draw = original.sample(c, v, w, A[3], A[4], keys, known, rng.random(bs), False)
            query = original.cumulative(cache, draw, False)
            assert np.max(abs(query)) <= M
            if start < len(frozen):
                stop = min(start+bs, len(frozen))
                assert np.max(abs(query[:stop-start]-frozen[start:stop])) < 1e-10
            vals.append(query)
        previous = end
        epsilon = min(epsilon, template.update(state, cache.values, par, p.s, p.from_moments)[0])
        lo, hi = e.intervals.interval(np.concatenate(vals), M, .02)
        running = [max(running[0], lo-guard), min(running[1], hi+guard)]
        baseline = cap["history"][phase]
        assert baseline["labels"] == cache.count and baseline["n"] == end
        alower, aupper = baseline["interval"]
        assert max(abs((alower+baseline["epsilon"])-running[0]), abs((aupper-baseline["epsilon"])-running[1])) < 1e-9
        L, U = running[0]-epsilon, running[1]+epsilon
        choice = p.s.decision(L, U, g) if running[0] <= running[1] else "unresolved"
        history.append(dict(blocks=end, comparison_labels=cache.count, epsilon=epsilon,
                            J=list(running), interval=[L,U], choice=choice))
        if choice != "unresolved":
            break
    fallback = choice == "unresolved"
    refit_n = 0
    if fallback:
        assert previous == p.p.CHECKS[-1]
        choice, interval, refit_n = original.fallback({"separate":cache}, inputs, seed, group, ci)
    else:
        interval = [L,U]
    rdata = dict(method="separate", choice=choice, interval=interval, fallback=fallback)
    row = original.fit_result(rdata, cache, final, a, b, case, ref["audit_gap"])
    record = dict(code=CODES[ci],group=group,seed=seed,method="separate",labels=row["labels"],
                  choice=choice,target=ref["audit_gap"],target_se=ref["audit_se"],g=g,
                  lower=interval[0],upper=interval[1],covered=row["covered"],acceptable=row["acceptable"],
                  resolved=choice!="unresolved",fallback=fallback,stages=json.dumps(row["stage_labels"],sort_keys=True),
                  provenance="query-first separate bounds on matched weighted paths; cache shared with independent refits",
                  source_file=f"queryfirst_20261125/experiment/path_{ci}_{group}.json")
    assert row["labels"] == sum(row["stage_labels"].values())
    result_path.write_text(json.dumps(dict(code=CODES[ci],group=group,seed=seed,epsilon0=epsilon0,
            refit_blocks=refit_n,matched_frozen_draws=min(len(frozen),previous),history=history,row=record),indent=2),encoding="utf8")
    print(CODES[ci], group, previous, refit_n, row["labels"], epsilon, flush=True)


def run_case(ci,groups,workers,out):
    case=p.s.CASES[ci]
    a,o,y,pools=p.s.load_domain(case["domain"])
    b=(1-case["alpha"])*a+case["alpha"]*o
    zeros=[np.zeros(z.shape[-1]) for z in (a,b)]
    par=p.p.parameters(a,b,pools,zeros,case["lam"])
    template=p.BoundTemplate(a,b,pools,zeros,case)
    inputs=(a,b,y,pools,case,zeros,par,template)
    print("prepared",CODES[ci],flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(run_path,ci,g,inputs,out) for g in groups]
        for job in jobs:
            job.result()


def collect(out):
    paths=[json.loads((out/f"path_{ci}_{g}.json").read_text()) for ci in CODES for g in range(6,18)]
    with (out/"paths.csv").open("w", encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(paths[0]["row"]));writer.writeheader()
        writer.writerows(q["row"] for q in paths)
    (out/"histories.json").write_text(json.dumps(paths,indent=2),encoding="utf8")
    summary={}
    for ci,code in CODES.items():
        part=[q for q in paths if q["code"]==code]
        old_counts=[]
        differences=[]
        for path in part:
            old=original.read(ARCHIVED_MOMENTS/f"{ci}_{path['group']}.json")
            old=next(record for record in old["records"] if record["method"]=="separate")
            old_counts.append(old["labels"])
            row=path["row"]
            differences.append(row["labels"]-old["labels"])
            if row["fallback"]:
                assert old["fallback"] and path["history"][-1]["blocks"]==p.p.CHECKS[-1]
                assert old["choice"]==row["choice"] and np.allclose(old["interval"],[row["lower"],row["upper"]],rtol=0,atol=1e-12)
                assert row["labels"]>=old["labels"]
        summary[code]=dict(mean=statistics.mean(q["row"]["labels"] for q in part),
            sd=statistics.stdev(q["row"]["labels"] for q in part),fallbacks=sum(q["row"]["fallback"] for q in part),
            covered=sum(q["row"]["covered"] for q in part),acceptable=sum(q["row"]["acceptable"] for q in part),
            epsilon0=part[0]["epsilon0"],epsilon_end_range=[f(q["history"][-1]["epsilon"] for q in part) for f in (min,max)],
            comparison_labels_mean=statistics.mean(q["history"][-1]["comparison_labels"] for q in part),
            empty_intersections=sum(any(h["J"][0]>h["J"][1] for h in q["history"]) for q in part),
            former_direct_refit_mean=statistics.mean(old_counts),
            paired_mean_change=statistics.mean(differences))
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=int,nargs="+",default=list(CODES))
    parser.add_argument("--groups",type=int,nargs="+",default=list(range(6,18)))
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--out",type=Path,default=HERE / "replayed_paths")
    parser.add_argument("--collect",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.collect:
        collect(args.out)
    else:
        for ci in args.cases:
            run_case(ci,args.groups,args.workers,args.out)
        if all((args.out/f"path_{ci}_{g}.json").exists() for ci in CODES for g in range(6,18)):
            collect(args.out)

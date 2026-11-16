"""Conditional paid initialization for the finite-pool population experiment.

Pilot indices are independent of comparison and final-training draws. No paid
calibration is added. Every score bound covers all compatible label completions.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,argparse,hashlib,contextlib,traceback,gc
ROOT=Path(__file__).resolve().parent; SHARED=ROOT.parent
sys.path.insert(0,str(SHARED/'logistic_all_scenarios_reanchor_20261114'))
sys.path.insert(0,str(SHARED/'logistic_population_sequential_epsilon_20261114'))
sys.path.insert(0,str(SHARED/'logistic_population_cache_only_20261114'))
import population_jobs as p
from cache_moments import from_moments
from incremental import BoundTemplate
import numpy as np
s=p.s
PILOT_OBJECTS=32; TRIGGER_RATIO=.5
GROUPS=tuple(range(6,18))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    tmp.replace(path)


def key(epsilon,par,g):
    if par['M']+epsilon+1e-9<=g:return (0,0.,epsilon)
    if epsilon<g:return (1,(par['M']/(g-epsilon))**2,epsilon)
    return (2,epsilon,par['M'])


def run_path(case,ci,group,a,b,y,pools,anchors,par,template,epsilon,pilot,yp,method):
    seed=202611141701+ci*10000000+group*10000
    cache=p.Cache(y);cache.objects(pilot,'pilot')
    states=template.state()
    epsilon=min(epsilon,template.update(states,cache.values,par,s,from_moments)[0])
    initial=epsilon
    rawL,rawU=-par['M']-1e-9,par['M']+1e-9
    L,U=rawL-epsilon,rawU+epsilon
    choice=s.decision(L,U,case['g']);history=[];blocks=0;fallback=False
    # The original public-wide gate is retained after pilot initialization.
    gate=epsilon>=case['g']
    if choice=='unresolved' and not gate:
        previous=0
        for phase,end in enumerate(p.CHECKS):
            lo,hi,detail=p.moment_phase(a,b,pools,case,anchors,par,cache,phase,end-previous,seed)
            rawL=max(rawL,lo);rawU=min(rawU,hi)
            before=cache.count
            candidate,details=template.update(states,cache.values,par,s,from_moments)
            assert cache.count==before
            epsilon=min(epsilon,candidate)
            L,U=rawL-epsilon,rawU+epsilon
            choice='unresolved' if rawL>rawU else s.decision(L,U,case['g'])
            history.append(dict(blocks=end,labels=cache.count,epsilon=epsilon,
                raw_interval=[rawL,rawU],interval=[L,U],choice=choice,**detail))
            blocks=end;previous=end
            if choice!='unresolved':break
    if choice=='unresolved':
        fallback=True
        choice,L,U,fh=p.fallback(a,b,pools,case,anchors,par['R'],cache,seed)
    else:fh=[]
    before=cache.count;fitinfo=None
    final=np.random.default_rng(s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    if choice!='unresolved':
        yy=cache.objects(final,'final_training')
        _,fitinfo=s.fit((a if choice=='A' else b)[final],yy,case['lam'])
    # The audit has no influence on preparation, queries, or stopping.
    audit=p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    gap,se=audit['mean'],audit['se']
    row=dict(method=method,case=case['id'],case_index=ci,group=group,seed=seed,
        choice=choice,interval=[L,U],epsilon=epsilon,initial_epsilon=initial,
        blocks=blocks,history=history,fallback=fallback,refit_history=fh,
        public_gate=gate,pilot_objects=PILOT_OBJECTS,pilot_labels=cache.stages['pilot'],
        calibration_labels=0,labels=cache.count,stage_labels=cache.stages,
        final_new_labels=cache.count-before,final_fit=fitinfo,
        pilot_overlap_final=int(len(np.intersect1d(pilot,final))),
        decision_issued=choice!='unresolved',audit_gap=gap,audit_se=se,
        covered_audit_mean=bool(L<=gap<=U),
        excludes_audit_band=bool(U<gap-3*se or L>gap+3*se),
        acceptable_audit_mean=bool(choice!='unresolved' and (gap<=case['g'] if choice=='A' else gap>=-case['g'])),
        range=par['M'],refit_range=par['R'])
    assert row['labels']==sum(cache.stages.values())
    assert not any(k=='calibration' for k in cache.stages)
    return row


def worker(ci):
    with (ROOT/'logs'/f'population_{ci:02d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            case=s.CASES[ci];tick=time.perf_counter()
            a,o,y,pools=s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
            zero=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
            zp=p.parameters(a,b,pools,zero,case['lam'])
            zt=BoundTemplate(a,b,pools,zero,case)
            eps0=zt.update(zt.state(),np.full(y.size,np.nan),zp,s,from_moments)[0]
            trigger=eps0>=TRIGGER_RATIO*case['g'] and key(eps0,zp,case['g'])[0]!=0
            for group in GROUPS:
                dest=ROOT/'population_results'/f'case_{ci:02d}_{group:02d}.json'
                if dest.exists():continue
                pathstart=time.perf_counter()
                old=p.read(SHARED/'logistic_population_cache_only_20261114/results'/f'case_{ci:02d}_{group:02d}.json')
                oldmethod='refit_if_public_wide' if eps0>=case['g'] else 'free_cache_update'
                base=next(r for r in old['records'] if r['method']==oldmethod)
                base=dict(base,method='zero_anchor',source_method=oldmethod,source='matched published record')
                assert abs(base['public_epsilon']-eps0)<1e-10
                pilot=np.random.default_rng(202611150901+ci*100000+group*1000).choice(pools['training'],PILOT_OBJECTS)
                cache=p.Cache(y);yp=cache.objects(pilot,'pilot')
                anchors=[s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
                pp=p.parameters(a,b,pools,anchors,case['lam'])
                pt=BoundTemplate(a,b,pools,anchors,case)
                pe=pt.update(pt.state(),cache.values,pp,s,from_moments)[0]
                ze=zt.update(zt.state(),cache.values,zp,s,from_moments)[0]
                usefit=key(pe,pp,case['g'])<key(ze,zp,case['g'])
                rows=[base]
                zr=run_path(case,ci,group,a,b,y,pools,zero,zp,zt,ze,pilot,yp,'early_labels_zero_anchor')
                zr['selected_anchor']='zero';rows.append(zr)
                if usefit:
                    nr=run_path(case,ci,group,a,b,y,pools,anchors,pp,pt,pe,pilot,yp,'pilot_anchor')
                    nr['selected_anchor']='nonzero'
                else:nr=dict(zr,method='pilot_anchor')
                rows.append(nr)
                rows.append(dict(nr if trigger else base,method='adaptive_pilot',triggered=trigger,
                                 source='pilot_anchor' if trigger else 'zero_anchor'))
                save(dest,dict(case=case,group=group,trigger=trigger,epsilon0=eps0,
                    initialization=dict(pilot_ids=pilot.tolist(),zero_epsilon=ze,nonzero_epsilon=pe,
                        zero_range=zp['M'],nonzero_range=pp['M'],zero_proxy=key(ze,zp,case['g']),
                        nonzero_proxy=key(pe,pp,case['g']),selected_nonzero=usefit,
                        anchor_norms=[float(np.linalg.norm(t)) for t in anchors]),
                    records=rows,seconds=time.perf_counter()-pathstart))
                print(ci,group,[(r['method'],r['labels']) for r in rows],flush=True)
                del pt,pp,cache;gc.collect()
            return dict(case=ci,seconds=time.perf_counter()-tick)
        except Exception:traceback.print_exc();raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--cases',nargs='*',type=int,default=list(range(8)));args=ap.parse_args()
    for sub in ('logs','population_results'):(ROOT/sub).mkdir(parents=True,exist_ok=True)
    if not (ROOT/'population_protocol.json').exists():
        save(ROOT/'population_protocol.json',dict(cases=s.CASES,groups=GROUPS,pilot_objects=PILOT_OBJECTS,
            trigger_ratio=TRIGGER_RATIO,paid_calibration=False,
            pilot='Independent training-pool draws, separate RNG from comparison and final training',
            target='Unchanged empirical-pool expected risk over random training sets',
            gate='After optional pilot, epsilon >= g goes directly to exact-refit branch',
            anchor_selection='Same predeclared feasibility/range proxy as fixed CV, using bounded statistic range',
            query_interval='Fresh-phase bounded-mean interval, not CP; query .02 and refit .02 allocations retained',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(worker,ci) for ci in args.cases]):
            row=fu.result();print(row,flush=True)
            save(ROOT/'population_progress.json',dict(last=row,done=len(list((ROOT/'population_results').glob('case_*.json'))),total=96))


if __name__=='__main__':main()

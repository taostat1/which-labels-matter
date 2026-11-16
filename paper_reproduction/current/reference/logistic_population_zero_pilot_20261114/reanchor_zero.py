"""Change only the initial pilot in the preceding reanchoring experiment."""
from run import *

def zero_variant(case,ci,group,a,b,y,pools):
    seed=202611141701+ci*10000000+group*10000;cache=p.Cache(y)
    ts=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
    history=[];calibrations=[];L=-np.inf;U=np.inf;n=0;choice='unresolved';fallback_used=False
    for phase,end in enumerate(p.CHECKS):
        if phase>0:ts=p.refit_anchor(a,b,pools,cache,case['lam'])
        par=p.parameters(a,b,pools,ts,case['lam'])
        eps,cal=p.calibrate(a,b,pools,ts,case,cache,phase,seed,par);calibrations.append(cal)
        print(ci,group,'zero_reanchor','phase',phase,'epsilon',eps,'labels',cache.count,flush=True)
        if eps>=case['g']:break
        lo,hi,detail=p.moment_phase(a,b,pools,case,ts,par,cache,phase,end-n,seed)
        lo-=eps;hi+=eps;L=max(L,lo);U=min(U,hi);n=end
        choice='conflict' if L>U else s.decision(L,U,case['g'])
        history.append(dict(phase=phase,blocks=n,labels=cache.count,epsilon=eps,range=par['M'],anchor_norms=[float(np.linalg.norm(t)) for t in ts],
                            alpha=.02*p.WEIGHTS[phase],interval=[L,U],phase_interval=[lo,hi],choice=choice,**detail))
        if choice!='unresolved':break
    conflict=choice=='conflict'
    if choice in ['unresolved','conflict']:
        fallback_used=True;choice,L,U,refit_history=p.fallback(a,b,pools,case,ts,par['R'],cache,seed)
    else:refit_history=[]
    before=cache.count
    if choice!='unresolved':
        fi=np.random.default_rng(s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
        yf=cache.objects(fi,'final_training');theta,info=s.fit((a if choice=='A' else b)[fi],yf,case['lam'])
    else:info=None
    audit=p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    gap=audit['mean'];se=audit['se'];valid=choice!='unresolved'
    row=dict(method='zero_reanchor_calibrated',labels=cache.count,stage_labels=cache.stages,final_new_labels=cache.count-before,choice=choice,interval=[L,U],
             epsilon=eps,blocks=n,fallback=fallback_used,empty_intersection=conflict,history=history,calibrations=calibrations,refit_history=refit_history,
             audit_gap=gap,audit_se=se,covered_audit_mean=bool(L<=gap<=U),excludes_audit_band=bool(U<gap-3*se or L>gap+3*se),
             acceptable_audit_mean=bool(valid and (gap<=case['g'] if choice=='A' else gap>=-case['g'])),decision_issued=valid,final_fit=info,seed=seed)
    assert row['labels']==sum(cache.stages.values());assert cache.stages.get('pilot',0)==0
    return row

def worker_zero(job):
    ci,group=job;dest=ROOT/'reanchor_results'/f'case_{ci:02d}_{group:02d}.json'
    if dest.exists():return job
    with (ROOT/'logs'/f'reanchor_{ci:02d}_{group:02d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            start=time.perf_counter();case=s.CASES[ci];a,o,y,pools=s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
            row=zero_variant(case,ci,group,a,b,y,pools)
            old=p.read(PREVIOUS/'results'/f'population_{ci:02d}_{group:02d}.json')
            ref=next(r for r in old['records'] if r['method']=='reanchor');ref=dict(ref,method='paid_reanchor_calibrated')
            s.save(dest,dict(case=case,group=group,records=[row,ref],seconds=time.perf_counter()-start));return job
        except Exception:traceback.print_exc();raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    (ROOT/'reanchor_results').mkdir(exist_ok=True)
    cfg=dict(cases=s.CASES,groups=[0,1,2],change='Remove the initial 512 paid pilot draws; start from theta=0.',
             unchanged='Same mandatory calibration, reanchor-from-cache after each phase, range reduction, EB intervals, checkpoints, alpha spending, fallback and final-training draws.',
             reference='Frozen reanchor records in preceding experiment, source hashes checked.',
             rationale='Check whether the value of initial pilot purchases changes when subsequent anchors are refitted.')
    pp=ROOT/'reanchor_protocol.json'
    if pp.exists():assert p.read(pp)==cfg
    else:s.save(pp,cfg)
    hp=ROOT/'reanchor_source_hash.json';hh=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if hp.exists():assert p.read(hp)['sha256']==hh
    else:s.save(hp,dict(sha256=hh))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker_zero,j):j for j in [(ci,g) for ci in range(8) for g in range(3)]}
        for f in as_completed(fs):
            try:j=f.result();status='done'
            except Exception as exc:j=fs[f];status=repr(exc)
            done=len(list((ROOT/'reanchor_results').glob('case_*.json')))
            s.save(ROOT/'reanchor_progress.json',dict(done=done,total=24,last_job=j,status=status));print(done,'/24',j,status,flush=True)

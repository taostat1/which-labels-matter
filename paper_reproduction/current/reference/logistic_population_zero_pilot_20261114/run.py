"""Matched zero-pilot and paid-pilot population-risk comparisons."""
from pathlib import Path
import os,sys,json,time,argparse,contextlib,traceback,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parent;PREVIOUS=SHARED/'logistic_all_scenarios_reanchor_20261114'
sys.path.insert(0,str(PREVIOUS))
import population_jobs as p
import numpy as np
from public_bound import allowance
s=p.s

METHODS=['zero_public_first','paid_public_first','zero_calibrated']

def prepare(a,b,y,pools,case,ci,group,paid):
    cache=p.Cache(y)
    if paid:
        pilot=np.random.default_rng(s.SEED+ci*100000+group*1000).choice(pools['training'],512)
        yp=cache.objects(pilot,'pilot');anchors=[s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
    else:anchors=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
    par=p.parameters(a,b,pools,anchors,case['lam'])
    eps,details=allowance(a,b,pools,anchors,case,par,s)
    return cache,anchors,par,eps,details

def run_one(a,b,y,pools,case,ci,group,method,prepared=None):
    seed=202611141701+ci*10000000+group*10000
    paid=method=='paid_public_first'
    if prepared is None:cache,anchors,par,public_eps,public_details=prepare(a,b,y,pools,case,ci,group,paid)
    else:
        _,anchors,par,public_eps,public_details=prepared;cache=p.Cache(y)
        assert not paid
    public_first=method.endswith('public_first')
    eps=public_eps;calibrations=[];used_calibration=not public_first or eps>=case['g']
    if used_calibration:
        eps,cal=p.calibrate(a,b,pools,anchors,case,cache,0,seed,par);calibrations.append(cal)
        if public_first:eps=min(public_eps,eps)
    print(ci,group,method,'public_eps',public_eps,'used_eps',eps,'initial_labels',cache.count,flush=True)
    L=-np.inf;U=np.inf;history=[];n=0;choice='unresolved';fallback=False
    # Same public entry rule for both primary methods. The strict calibration
    # ablation keeps the previous flow, changing only the pilot/anchor.
    if public_first:
        L=-par['M']-eps-1e-9;U=par['M']+eps+1e-9
        choice=s.decision(L,U,case['g'])
    if choice=='unresolved' and eps<case['g']:
        for phase,end in enumerate(p.CHECKS):
            lo,hi,detail=p.moment_phase(a,b,pools,case,anchors,par,cache,phase,end-n,seed)
            lo-=eps;hi+=eps;L=max(L,lo);U=min(U,hi);n=end
            choice='conflict' if L>U else s.decision(L,U,case['g'])
            history.append(dict(phase=phase,blocks=n,labels=cache.count,epsilon=eps,range=par['M'],alpha=.02*p.WEIGHTS[phase],
                                interval=[L,U],phase_interval=[lo,hi],choice=choice,**detail))
            if choice!='unresolved':break
    conflict=choice=='conflict'
    if choice in ['unresolved','conflict']:
        fallback=True;choice,L,U,refit_history=p.fallback(a,b,pools,case,anchors,par['R'],cache,seed)
    else:refit_history=[]
    before=cache.count;info=None
    if choice!='unresolved':
        fi=np.random.default_rng(s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
        yy=cache.objects(fi,'final_training');theta,info=s.fit((a if choice=='A' else b)[fi],yy,case['lam'])
    audit=p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    gap,se=audit['mean'],audit['se'];issued=choice!='unresolved'
    row=dict(method=method,labels=cache.count,stage_labels=cache.stages,pilot_draws=512 if paid else 0,
             public_epsilon=public_eps,public_details=public_details,epsilon=eps,used_calibration=used_calibration,
             anchors=[t.tolist() for t in anchors],range=par['M'],refit_range=par['R'],calibrations=calibrations,
             choice=choice,interval=[L,U],blocks=n,history=history,fallback=fallback,refit_history=refit_history,
             empty_intersection=conflict,final_new_labels=cache.count-before,final_fit=info,seed=seed,
             audit_gap=gap,audit_se=se,decision_issued=issued,covered_audit_mean=bool(L<=gap<=U),
             excludes_audit_band=bool(U<gap-3*se or L>gap+3*se),
             acceptable_audit_mean=bool(issued and (gap<=case['g'] if choice=='A' else gap>=-case['g'])))
    assert row['labels']==sum(row['stage_labels'].values())
    assert paid or row['stage_labels'].get('pilot',0)==0
    return row

def worker(job):
    ci,group=job;dest=ROOT/'results'/f'case_{ci:02d}_{group:02d}.json'
    if dest.exists():return job
    with (ROOT/'logs'/f'case_{ci:02d}_{group:02d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            start=time.perf_counter();case=s.CASES[ci];a,o,y,pools=s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
            assert max(np.linalg.norm(a,axis=-1).max(),np.linalg.norm(b,axis=-1).max())<=1+2e-6
            zero=prepare(a,b,y,pools,case,ci,group,False)
            rows=[run_one(a,b,y,pools,case,ci,group,'zero_public_first',zero),
                  run_one(a,b,y,pools,case,ci,group,'paid_public_first'),
                  run_one(a,b,y,pools,case,ci,group,'zero_calibrated',zero)]
            old=p.read(PREVIOUS/'results'/f'population_{ci:02d}_{group:02d}.json')
            ref=next(r for r in old['records'] if r['method']=='fixed_anchor');ref=dict(ref,method='paid_calibrated')
            s.save(dest,dict(case=case,group=group,records=rows,reference=ref,seconds=time.perf_counter()-start));return job
        except Exception:traceback.print_exc();raise

def register():
    cfg=dict(cases=s.CASES,groups=[0,1,2],paths_per_group=1,methods=METHODS,reference='Frozen paid512 + mandatory calibration + fixed anchor + remaining contributions, from the preceding matched experiment.',
      target='Same declared population program risk: expected test BCE after exact training on size-n iid draws.',
      primary='Zero pilot vs 512 paid pilot draws; both first evaluate the same response-uniform public moment/error formulas and skip calibration when epsilon<g.',
      diagnostic='Zero pilot with mandatory calibration differs from the frozen paid-calibrated control only in the pilot/anchor.',
      paid_pilot='Same original 512 training draw indices; all acquired attributes count once and are cached.',
      anchors='Fixed after initialization. No sequential refitting; this comparison isolates whether initial pilot purchases pay off.',
      public_certificate='Enumerate 2^r hypothetical label vectors per public feature object to bound score norm and second moment; bound score mean using public Gram/coordinate envelopes. Insert in the existing global/local population theorems. No oracle labels are used.',
      calibration='Original phase-0 fresh draws and .01*WEIGHTS[0] allocation, 2048/8192/32768 training, 2048 test. Public-first takes min(public, calibrated).',
      inference='Unchanged fresh-phase empirical Bernstein, remaining-term reduction, phase alpha=.02*WEIGHTS[phase]; same capped exact-refit fallback with .02.',
      inspections=p.CHECKS,delta=.05,error_budgets=dict(calibration=.01,queries=.02,refits=.02),
      seeds='Same as preceding experiment, including independent calibration, comparison, refit, and final-fit streams.',
      outcome='Total distinct pilot+calibration+comparison+refit+final-training labels. Pilot is worthwhile iff subsequent label savings exceed its newly acquired pilot labels.',
      audit='Frozen independent 2048-fit mean and SE, not exact population truth.',scope='Eight unchanged configurations, three matched preparation/query paths per configuration; all reported.')
    path=ROOT/'protocol.json'
    if path.exists():assert p.read(path)==cfg
    else:s.save(path,cfg)
    files=[Path(__file__),ROOT/'public_bound.py',PREVIOUS/'population_jobs.py',SHARED/'shared_logistic_full_20261029/study.py',SHARED/'shared_logistic_full_20261029/core.py',SHARED/'shared_logistic_full_20261029/local_bounds.py']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};hp=ROOT/'source_hashes.json'
    if hp.exists():assert p.read(hp)==hashes
    else:s.save(hp,hashes)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--group',type=int);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();register()
    jobs=[(ci,g) for ci in range(8) for g in range(3) if (args.case is None or args.case==ci) and (args.group is None or args.group==g)]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in jobs}
        for f in as_completed(fs):
            try:j=f.result();status='done'
            except Exception as exc:j=fs[f];status=repr(exc)
            done=len(list((ROOT/'results').glob('case_*.json')));s.save(ROOT/'progress.json',dict(done=done,total=24,last_job=j,status=status))
            print(done,'/24',j,status,flush=True)

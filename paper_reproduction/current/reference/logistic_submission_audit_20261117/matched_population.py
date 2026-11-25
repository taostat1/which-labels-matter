"""Matched preparation/refinement study; the comparison interval is held fixed.

The legacy preparation uses 512 pilot objects and independent paid calibration.
Zero-anchor paths are the frozen 96 current-paper paths. The two paid arms share
all draws until their own stopping times, and final-fit purchases never leak
into a still-running arm. Oracle risk audits are read only after all decisions.
"""
from pathlib import Path
import os, sys, json, time, argparse, subprocess, inspect
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent
SHARED = Path(os.environ.get('LEC_SHARED', REFERENCE)).resolve()
sys.path.insert(0,str(SHARED/'logistic_population_betting_20261115'))
sys.path.insert(0, str(REFERENCE))
from population_stopping import assess_interval
import experiment as e
pop,np,intervals=e.pop,e.np,e.intervals
OUT = Path(os.environ.get('LEC_PAID_OUTPUT', ROOT / 'query_first_paid_results')).resolve()
for subdir in ('results','draws','logs'): (OUT/subdir).mkdir(parents=True,exist_ok=True)
# Freeze the existing sampler verbatim, changing only its explicit anchor input.
src=inspect.getsource(e.draw_phase).replace('phase,size,seed):','phase,size,seed,anchors):').replace('zero=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]','zero=anchors')
import hashlib
exec(compile(src,str(OUT/'sampler.py'),'exec'))

def run(ci,group):
    path=OUT/'results'/f'case_{ci:02d}_{group:02d}.json'
    if path.exists():return
    tick=time.time();case=pop.s.CASES[ci];seed=202611141701+ci*10000000+group*10000
    a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
    cache=pop.p.Cache(y)
    pilot=np.random.default_rng(pop.s.SEED+ci*100000+group*1000).choice(pools['training'],512)
    yp=cache.objects(pilot,'pilot');anchors=[pop.s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
    par=pop.p.parameters(a,b,pools,anchors,case['lam'])
    # The historical calibration has 3 looks x 2 candidates x 2 moment families.
    # Freeze phase-0 beta=.005 so every arm fits the same nominal .05 budget.
    saved=pop.p.WEIGHTS[0];pop.p.WEIGHTS[0]=.5
    epsilon0,cal=pop.p.calibrate(a,b,pools,anchors,case,cache,0,seed,par)
    pop.p.WEIGHTS[0]=saved
    template=pop.BoundTemplate(a,b,pools,anchors,case);states=template.state()
    # Both arms start with exactly the same paid certificate. Refinement occurs
    # only after entering the approximate comparison, as in Algorithm 2.
    eps=epsilon0;M=par['M']+1e-8;guard=1e-9*(1+M);running=[-M,M]
    methods=('pilot_fixed','pilot_cache');rows={};hist={m:[] for m in methods}
    final=np.random.default_rng(pop.s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    vals=[];previous=0;phases=[]
    entry, public_interval = assess_interval((-M, M), eps, case['g'])
    if entry!='unresolved':
        for m in methods:
            r=dict(method=m,choice=entry,interval=public_interval,stopping_blocks=0,fallback=False,epsilon=eps,history=[])
            e.finish(r,cache,final,a,b,case);rows[m]=r
    else:
        for phase,end in enumerate(pop.p.CHECKS):
            if len(rows)==2:break
            vv,detail=draw_phase(a,b,pools,case,par,cache,phase,end-previous,seed,anchors)
            vals.extend(vv.tolist());previous=end
            eps=min(eps,template.update(states,cache.values,par,pop.s,pop.from_moments)[0])
            lo,hi=intervals.interval(np.array(vals),M,.02)
            running=[max(running[0],lo-guard),min(running[1],hi+guard)]
            phases.append(dict(blocks=end,labels=cache.count,epsilon=eps,**detail))
            for m in methods:
                if m in rows:continue
                ee = eps if m == 'pilot_cache' else epsilon0
                choice, risk_interval = assess_interval(running, ee, case['g'])
                hist[m].append(dict(blocks=end,labels=cache.count,epsilon=ee,interval=risk_interval,choice=choice))
                if choice!='unresolved':
                    r=dict(method=m,choice=choice,interval=risk_interval,stopping_blocks=end,fallback=False,epsilon=ee,history=hist[m])
                    e.finish(r,cache,final,a,b,case);rows[m]=r
    if len(rows)<2:
        archive=SHARED/'logistic_adaptive_initialization_20261115/population_results'/f'case_{ci:02d}_{group:02d}.json'
        ref=None
        if archive.exists():
            doc=json.loads(archive.read_text('utf-8'));ref=next(r for r in doc['records'] if r['method']=='zero_anchor')
            if not ref['fallback']:ref=None
        if ref:
            assert ref['seed']==seed and abs(ref['refit_range']-par['R'])<1e-12
            rng=np.random.default_rng(seed+6000000);num=0;fh=[]
            eta=2e-9/case['lam'];guard_ref=1e-8
            for h in ref['refit_history']:
                while num<h['blocks']:
                    bs=min(64,h['blocks']-num);ids=rng.choice(pools['training'],(bs,case['n']));tests=rng.choice(pools['comparison'],bs)
                    cache.objects(ids,'refit');cache.objects(tests,'refit');num+=bs
                L,U=h['interval'];extra=14*(eta+guard_ref)*np.log(4*len(pop.p.CHECKS)/.02)/(3*(num-1))
                L-=extra;U+=extra;choice=pop.s.decision(L,U,case['g'])
                fh.append(dict(blocks=num,labels=cache.count,interval=[L,U],choice=choice))
                if choice!='unresolved':break
            assert choice!='unresolved','Archived stream must be extended for the wider interval'
            refit_mode='frozen identical exact-fit statistics; label union and numerical range guard recomputed'
        else:
            # Keep fits independent of preparation; match the zero-anchor refit start.
            zeros=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
            choice,L,U,fh=pop.p.fallback(a,b,pools,case,zeros,par['R']+2e-9/case['lam']+1e-8,cache,seed)
            refit_mode='exact paired fits recomputed'
        for m in methods:
            if m in rows:continue
            r=dict(method=m,choice=choice,interval=[L,U],stopping_blocks=previous,fallback=True,
                   epsilon=eps if m=='pilot_cache' else epsilon0,history=hist[m],refit_history=fh,refit_mode=refit_mode)
            if choice!='unresolved':e.finish(r,cache,final,a,b,case)
            else:r.update(labels=cache.count,stage_labels=cache.stages.copy())
            rows[m]=r
    # Audit access follows acquisition, stopping, and final fitting.
    audit=pop.p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    for r in rows.values():
        gap=audit['mean'];L,U=r['interval']
        r.update(case_index=ci,group=group,seed=seed,epsilon0=epsilon0,range=par['M'],refit_range=par['R'],
                 decision_issued=r['choice']!='unresolved',audit_gap=gap,audit_se=audit['se'],
                 acceptable_audit_mean=bool(r['choice']!='unresolved' and (gap<=case['g'] if r['choice']=='A' else gap>=-case['g'])),
                 covered_audit_mean=bool(L<=gap<=U),error_probability=.045,
                 calibration=cal,anchor_norms=[float(np.linalg.norm(t)) for t in anchors])
        assert r['labels']==sum(r['stage_labels'].values())
    assert all(not r['fallback'] or r['stopping_blocks']==pop.p.CHECKS[-1] for r in rows.values())
    pop.save(path,dict(case=case,case_index=ci,group=group,seed=seed,records=list(rows.values()),phase_records=phases,seconds=time.time()-tick))
    np.savez_compressed(OUT/'draws'/f'case_{ci:02d}_{group:02d}.npz',z=np.array(vals))
    print(ci,group,[(m,rows[m]['labels'],rows[m]['fallback']) for m in methods],flush=True)

def batch(workers):
    for d in ('results','logs','draws'):(OUT/d).mkdir(exist_ok=True)
    (OUT/'sampler.py').write_text(src,encoding='utf-8')
    protocol=dict(cases=pop.s.CASES,groups=list(range(6,18)),methods=['zero_fixed','zero_cache','pilot_fixed','pilot_cache'],
      pilot_draws=512,training_calibration_looks=[2048,8192,32768],test_calibration_draws=2048,
      nominal_error=.05,calibration_error=.005,query_error=.02,refit_error=.02,checkpoints=pop.p.CHECKS,
      label_budget_grid=[10000,20000,40000,80000],interval='same cumulative bounded-mean betting mixture',
      initial_bound='paid: independently calibrated nonzero-anchor bound; zero: public bound',
      baseline='matched seeds, final samples, audits and query/refit caps across the initialization arms',
      change_from_historical='legacy paid preparation, with the current cumulative interval to isolate preparation',
      audit='independent 2048-fit mean and SE; read only after decisions',
      zero_source='logistic_population_betting_revision_20261115/results')
    pop.save(OUT/'protocol.json',protocol)
    jobs=[(i,g) for i in range(8) for g in range(6,18)];live=[];errors=[];done=0
    while jobs or live:
        while jobs and len(live)<workers:
            ci,g=jobs.pop(0)
            if (OUT/'results'/f'case_{ci:02d}_{g:02d}.json').exists():done+=1;continue
            log=(OUT/'logs'/f'case_{ci:02d}_{g:02d}.log').open('w',encoding='utf-8')
            p=subprocess.Popen([sys.executable,str(__file__),'--case',str(ci),'--group',str(g)],stdout=log,stderr=subprocess.STDOUT)
            live.append((ci,g,p,log))
        remaining=[]
        for ci,g,p,log in live:
            if p.poll() is None:remaining.append((ci,g,p,log));continue
            log.close();done+=1
            if p.returncode:errors.append((ci,g,p.returncode))
            print('Completed',done,'/96',ci,g,p.returncode,flush=True)
        live=remaining;pop.save(OUT/'progress.json',dict(done=done,total=96,errors=errors))
        if live:time.sleep(1)
    if errors:raise RuntimeError(errors)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--group',type=int);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.case is None:batch(args.workers)
    else:run(args.case,args.group)

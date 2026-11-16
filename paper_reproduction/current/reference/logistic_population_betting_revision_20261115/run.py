"""Current Algorithm 2 and a fixed-bound ablation on the original 96 paths."""
from pathlib import Path
import os,sys,json,time,argparse,subprocess,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parent
sys.path.insert(0,str(SHARED/'logistic_population_betting_20261115'))
import experiment as e
pop,np,intervals=e.pop,e.np,e.intervals
METHODS=('lec','fixed_bound')

def run(ci,group):
    path=ROOT/'results'/f'case_{ci:02d}_{group:02d}.json'
    if path.exists():return
    tick=time.perf_counter();case=pop.s.CASES[ci]
    a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
    zero=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
    par=pop.p.parameters(a,b,pools,zero,case['lam'])
    template=pop.BoundTemplate(a,b,pools,zero,case);states=template.state();cache=pop.p.Cache(y)
    moment_eps=template.update(states,cache.values,par,pop.s,pop.from_moments)[0]
    epsilon0=moment_eps;cert=None
    if epsilon0>=case['g'] and ci in (3,6):
        integration=json.loads((SHARED/'logistic_population_paired_local_20261115'/f'integration_{ci}.json').read_text())
        cert=min(integration['certificates'],key=lambda v:v['epsilon']);epsilon0=min(epsilon0,cert['epsilon'])
    epsilon=epsilon0;seed=202611141701+ci*10000000+group*10000
    final=np.random.default_rng(pop.s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    M=par['M']+1e-8;guard=1e-9*(1+M);running=[-M,M]
    histories={m:[] for m in METHODS};rows={};allvals=[];phases=[];previous=0
    entry=pop.s.decision(-M-epsilon,M+epsilon,case['g'])
    if entry!='unresolved':
        for method in METHODS:
            row=dict(method=method,choice=entry,interval=[-M-epsilon,M+epsilon],stopping_blocks=0,fallback=False,epsilon=epsilon,history=[])
            e.finish(row,cache,final,a,b,case);rows[method]=row
    if epsilon0<case['g']:
        for phase,end in enumerate(pop.p.CHECKS):
            if len(rows)==len(METHODS):break
            vals,detail=e.draw_phase(a,b,pools,case,par,cache,phase,end-previous,seed)
            allvals.extend(vals.tolist());previous=end
            epsilon=min(epsilon,template.update(states,cache.values,par,pop.s,pop.from_moments)[0])
            lo,hi=intervals.interval(np.array(allvals),M,.02)
            running=[max(running[0],lo-guard),min(running[1],hi+guard)]
            phases.append(dict(phase=phase,blocks=end,labels=cache.count,epsilon=epsilon,raw_interval=list(running),**detail))
            for method in METHODS:
                if method in rows:continue
                eps=epsilon if method=='lec' else epsilon0
                L,U=running[0]-eps,running[1]+eps
                choice=pop.s.decision(L,U,case['g']) if L<=U else 'unresolved'
                histories[method].append(dict(blocks=end,labels=cache.count,raw_interval=list(running),interval=[L,U],epsilon=eps,choice=choice))
                if choice!='unresolved':
                    row=dict(method=method,choice=choice,interval=[L,U],stopping_blocks=end,fallback=False,epsilon=eps,history=histories[method])
                    e.finish(row,cache,final,a,b,case);rows[method]=row
                    print(ci,group,method,'STOP',end,row['labels'],flush=True)
    if len(rows)<len(METHODS):
        choice,L,U,fh=pop.p.fallback(a,b,pools,case,zero,par['R'],cache,seed)
        for method in METHODS:
            if method in rows:continue
            row=dict(method=method,choice=choice,interval=[L,U],stopping_blocks=previous,fallback=True,
                     epsilon=epsilon if method=='lec' else epsilon0,history=histories[method],refit_history=fh)
            if choice!='unresolved':e.finish(row,cache,final,a,b,case)
            else:row.update(labels=cache.count,stage_labels=cache.stages.copy())
            rows[method]=row
    audit=pop.p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    for row in rows.values():
        gap=audit['mean'];L,U=row['interval']
        row.update(case_index=ci,group=group,seed=seed,epsilon0=epsilon0,pilot_labels=0,calibration_labels=0,
                   acceptable_audit_mean=bool(row['choice']!='unresolved' and (gap<=case['g'] if row['choice']=='A' else gap>=-case['g'])),
                   covered_audit_mean=bool(L<=gap<=U),decision_issued=row['choice']!='unresolved',audit_gap=gap,audit_se=audit['se'],
                   error_probability=.045 if cert else .04,anchors='fixed_zero')
    assert rows['lec']['labels']<=rows['fixed_bound']['labels']
    earlier=SHARED/'logistic_population_betting_20261115/results'/f'case_{ci:02d}_{group:02d}.json'
    parity=None
    if earlier.exists():
        ref=next(r for r in json.loads(earlier.read_text())['records'] if r['method']=='cumulative_betting')
        r=rows['lec'];assert (r['labels'],r['choice'],r['stopping_blocks'])==(ref['labels'],ref['choice'],ref['stopping_blocks'])
        assert max(abs(np.array(r['interval'])-ref['interval']))<1e-10;parity=True
    np.savez_compressed(ROOT/'draws'/f'case_{ci:02d}_{group:02d}.npz',z=np.array(allvals))
    pop.save(path,dict(case=case,case_index=ci,group=group,seed=seed,old_public_epsilon=moment_eps,epsilon0=epsilon0,
                      public_certificate=cert,range=par['M'],records=[rows[m] for m in METHODS],phase_records=phases,
                      prior_interval_experiment_parity=parity,seconds=time.perf_counter()-tick))

def batch(workers):
    jobs=[(ci,g) for ci in range(8) for g in range(6,18)];running=[];errors=[];done=0;tick=time.time()
    for d in ('logs','results','draws'):(ROOT/d).mkdir(exist_ok=True)
    (ROOT/'protocol.json').write_text(json.dumps(dict(jobs=jobs,methods=METHODS,checks=pop.p.CHECKS,
        query_error=.02,refit_error=.02,public_integration_error=.005,total_error_budget=.05,
        anchors='fixed_zero',pilot_labels=0,paid_calibration_labels=0,
        fractions=intervals.FRACTIONS.tolist(),external_baselines='Frozen 288 exact-fit comparator records from 2026-11-14'),indent=2))
    while jobs or running:
        while jobs and len(running)<workers:
            ci,g=jobs.pop(0)
            if (ROOT/'results'/f'case_{ci:02d}_{g:02d}.json').exists():done+=1;continue
            log=(ROOT/'logs'/f'case_{ci:02d}_{g:02d}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(__file__),'--case',str(ci),'--group',str(g)],stdout=log,stderr=subprocess.STDOUT)
            running.append((ci,g,proc,log))
        keep=[]
        for ci,g,proc,log in running:
            if proc.poll() is None:keep.append((ci,g,proc,log));continue
            log.close();done+=1
            if proc.returncode:errors.append((ci,g,proc.returncode))
            print('FINISH',ci,g,proc.returncode,'done',done,flush=True)
        running=keep
        (ROOT/'progress.json').write_text(json.dumps(dict(done=done,total=96,errors=errors,seconds=time.time()-tick)))
        if running:time.sleep(1)
    if errors:raise RuntimeError(errors)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--case',type=int);ap.add_argument('--group',type=int);a=ap.parse_args()
    if a.case is None:batch(a.workers)
    else:run(a.case,a.group)

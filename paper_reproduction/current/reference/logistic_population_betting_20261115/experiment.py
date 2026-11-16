"""Interval-only comparison on identical label acquisition paths."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parent;WORK=ROOT.parents[2]
sys.path.insert(0,str(SHARED/'logistic_adaptive_initialization_20261115'))
import fixed_experiment as f
import population_experiment as pop
import numpy as np
import json,time,argparse,hashlib
import intervals

METHODS=('phase_EB','phase_betting','cumulative_betting','cumulative_plugin_betting')

def draw_phase(a,b,pools,case,par,cache,phase,size,seed):
    rng=np.random.default_rng(seed+4000000+phase*100000)
    known=cache.values.copy();vals=[];maxmass=0.;maxenvelope=0.;identity=0.;keysqueried=[]
    zero=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
    for start in range(0,size,128):
        bs=min(128,size-start);ids=rng.choice(pools['training'],(bs,case['n']));tests=rng.choice(pools['comparison'],bs)
        aa=pop.s.fast_coefficients(a,ids,tests,zero[0],case['lam'],par['tc'][0],par['dc'][0])
        bb=pop.s.fast_coefficients(b,ids,tests,zero[1],case['lam'],par['tc'][1],par['dc'][1])
        c,v,w=[aa[k]-bb[k] for k in range(3)];l,h=aa[3:]
        assert np.max(abs(c)+abs(v).sum(1)+abs(w).sum(1))<=par['M']+1e-8
        nodes=np.stack([tests,ids[:,0],ids[:,1]],1)
        keys=(nodes[:,:,None]*cache.r+np.arange(cache.r)).reshape(bs,-1)
        rows=pop.p.reduced_coefficients(c,v,w,l,h,keys,known);us=rng.random(bs)
        for i,(constant,kk,cc) in enumerate(rows):
            mass=float(abs(cc).sum());maxmass=max(maxmass,mass);maxenvelope=max(maxenvelope,abs(constant)+mass)
            assert abs(constant)+mass<=par['M']+1e-8
            if mass:
                prob=abs(cc)/mass;cdf=np.cumsum(prob);cdf[-1]=1.
                ix=min(np.searchsorted(cdf,us[i],side='right'),len(cc)-1);kl,kr=kk[ix]
                x=2*cache.query([kl],'comparison')[0]-1;keysqueried.append(int(kl))
                if kr>=0:x*=2*cache.query([kr],'comparison')[0]-1;keysqueried.append(int(kr))
                value=constant+np.sign(cc[ix])*mass*x
            else:value=constant
            vals.append(value)
        if start==0:
            fake=np.where(np.isfinite(known),2*known-1,np.where(np.arange(len(known))%3==0,1.,-1.))
            direct=pop.s.evaluate((c,v,w,l,h),fake[keys])
            for i,(constant,kk,cc) in enumerate(rows):
                prod=fake[kk[:,0]].copy();pair=kk[:,1]>=0;prod[pair]*=fake[kk[pair,1]]
                identity=max(identity,abs(direct[i]-constant-cc@prod))
            assert identity<1e-9
    vals=np.array(vals)
    return vals,dict(mean=float(vals.mean()),variance=float(vals.var(ddof=1)),max_remaining_mass=maxmass,
        max_observed_envelope=maxenvelope,identity_error=identity,
        query_hash=hashlib.sha256(np.array(keysqueried,dtype=np.int64).tobytes()).hexdigest())

def finish(row,physical,final,a,b,case):
    cache=pop.p.Cache(physical._oracle.reshape(-1,physical.r))
    cache.values=physical.values.copy();cache.stages=physical.stages.copy();before=cache.count
    yf=cache.objects(final,'final_training')
    _,info=pop.s.fit((a if row['choice']=='A' else b)[final],yf,case['lam'])
    row.update(labels=cache.count,stage_labels=cache.stages,final_new_labels=cache.count-before,final_fit=info)
    assert row['labels']==sum(row['stage_labels'].values())

def run(ci,group):
    path=ROOT/'results'/f'case_{ci:02d}_{group:02d}.json'
    if path.exists():return
    tick=time.perf_counter();case=pop.s.CASES[ci]
    a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
    ts=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])];par=pop.p.parameters(a,b,pools,ts,case['lam'])
    template=pop.BoundTemplate(a,b,pools,ts,case);states=template.state();cache=pop.p.Cache(y)
    epsilon0=template.update(states,cache.values,par,pop.s,pop.from_moments)[0]
    if ci in (3,6):
        integrated=json.loads((SHARED/'logistic_population_paired_local_20261115'/f'integration_{ci}.json').read_text())
        epsilon0=min(epsilon0,min(v['epsilon'] for v in integrated['certificates']))
    epsilon=epsilon0;seed=202611141701+ci*10000000+group*10000
    final=np.random.default_rng(pop.s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    M=par['M']+1e-8;guard=1e-9*(1+M)
    running={m:[-M,M] for m in METHODS};histories={m:[] for m in METHODS};rows={};allvals=[];phase_records=[];previous=0
    assert epsilon<case['g'],(case,epsilon)
    entry=pop.s.decision(-M-epsilon,M+epsilon,case['g'])
    if entry!='unresolved':
        for method in METHODS:
            row=dict(method=method,choice=entry,interval=[-M-epsilon,M+epsilon],stopping_blocks=0,
                     fallback=False,epsilon=epsilon,history=[])
            finish(row,cache,final,a,b,case);rows[method]=row
    for phase,end in enumerate(pop.p.CHECKS):
        if len(rows)==len(METHODS):break
        vals,detail=draw_phase(a,b,pools,case,par,cache,phase,end-previous,seed)
        allvals.extend(vals.tolist());previous=end
        eps=template.update(states,cache.values,par,pop.s,pop.from_moments)[0];epsilon=min(epsilon,eps)
        phase_records.append(dict(phase=phase,blocks=end,comparison_labels=cache.count,epsilon=epsilon,**detail))
        for method in METHODS:
            if method in rows:continue
            if method=='phase_EB':lo,hi=intervals.eb_phase(vals,par['M'],.02*pop.p.WEIGHTS[phase])
            elif method=='phase_betting':lo,hi=intervals.interval(vals,M,.02*pop.p.WEIGHTS[phase])
            else:lo,hi=intervals.interval(np.array(allvals),M,.02,method=='cumulative_plugin_betting')
            lo-=guard;hi+=guard;running[method][0]=max(running[method][0],lo);running[method][1]=min(running[method][1],hi)
            L,U=running[method][0]-epsilon,running[method][1]+epsilon
            choice=pop.s.decision(L,U,case['g']) if L<=U else 'unresolved'
            hist=dict(blocks=end,labels=cache.count,raw_interval=list(running[method]),interval=[L,U],epsilon=epsilon,choice=choice)
            histories[method].append(hist)
            if choice!='unresolved':
                row=dict(method=method,choice=choice,interval=[L,U],stopping_blocks=end,fallback=False,epsilon=epsilon,history=histories[method])
                finish(row,cache,final,a,b,case);rows[method]=row
                print(ci,group,method,'STOP',end,row['labels'],flush=True)
        print(ci,group,'phase',end,'labels',cache.count,flush=True)
        if len(rows)==len(METHODS):break
    if len(rows)<len(METHODS):
        archived=SHARED/'logistic_adaptive_initialization_20261115/population_results'/f'case_{ci:02d}_{group:02d}.json'
        ref=None
        if archived.exists():
            old=json.loads(archived.read_text('utf-8'));ref=next(v for v in old['records'] if v['method']=='zero_anchor')
            if not ref['fallback']:ref=None
        if ref:
            assert ref['seed']==seed and abs(par['R']-ref['refit_range'])<1e-12
            rng=np.random.default_rng(seed+6000000);num=0;fh=[]
            for h in ref['refit_history']:
                while num<h['blocks']:
                    bs=min(64,h['blocks']-num);ids=rng.choice(pools['training'],(bs,case['n']));test=rng.choice(pools['comparison'],bs)
                    cache.objects(ids,'refit');cache.objects(test,'refit');num+=bs
                fh.append(dict(h,labels=cache.count))
            choice=ref['choice'];L,U=ref['refit_history'][-1]['interval'];refit_mode='matched archived statistics; recomputed union'
        else:
            choice,L,U,fh=pop.p.fallback(a,b,pools,case,ts,par['R'],cache,seed);refit_mode='rerun exact paired fits'
        for method in METHODS:
            if method in rows:continue
            row=dict(method=method,choice=choice,interval=[L,U],stopping_blocks=previous,fallback=True,epsilon=epsilon,
                     history=histories[method],refit_history=fh,refit_mode=refit_mode)
            if choice!='unresolved':finish(row,cache,final,a,b,case)
            else:row.update(labels=cache.count,stage_labels=cache.stages.copy())
            rows[method]=row
    # Oracle audit is loaded only after every acquisition decision has finished.
    audit=pop.p.read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    for row in rows.values():
        gap=audit['mean'];L,U=row['interval']
        row.update(acceptable_audit_mean=bool(row['choice']!='unresolved' and (gap<=case['g'] if row['choice']=='A' else gap>=-case['g'])),
                   covered_audit_mean=bool(L<=gap<=U),audit_gap=gap,audit_se=audit['se'])
    # Retain all observed contributions for exact reproduction of interval comparisons.
    (ROOT/'draws').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(ROOT/'draws'/f'case_{ci:02d}_{group:02d}.npz',z=np.array(allvals))
    pop.save(path,dict(case=case,case_index=ci,group=group,seed=seed,epsilon0=epsilon0,range=par['M'],
                      methods=METHODS,records=[rows[m] for m in METHODS],phase_records=phase_records,
                      seconds=time.perf_counter()-tick))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True);ap.add_argument('--group',type=int,required=True)
    args=ap.parse_args();run(args.case,args.group)

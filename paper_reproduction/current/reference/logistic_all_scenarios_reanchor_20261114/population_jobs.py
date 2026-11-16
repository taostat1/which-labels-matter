"""Population extension: history-dependent anchors, fresh calibration and EB phases.

The fixed-CV CP construction is deliberately not used for random-training risk.
Both variants use the same pilot, first calibration and phase streams. All paid
labels are cached. Exact risks are accessible only in the retrospective audit.
"""
from pathlib import Path
import os,sys,json,gzip,time,contextlib,traceback,argparse,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parent
sys.path.insert(0,str(SHARED/'shared_logistic_full_20261029'))
import study as s
import numpy as np
from scipy.special import expit

CHECKS=[256,512,1024,2048,4096,8192,16384,32768]
WEIGHTS=np.array([2.**(-i-1) for i in range(8)]);WEIGHTS/=WEIGHTS.sum()

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def dump(p,obj):s.save(p,obj)

class Cache:
    def __init__(self,y):
        self._oracle=y.reshape(-1);self.r=y.shape[1];self.values=np.full(y.size,np.nan);self.stages={}
    @property
    def count(self):return int(np.isfinite(self.values).sum())
    def query(self,keys,stage):
        keys=np.asarray(keys,dtype=int);unique=np.unique(keys);new=unique[~np.isfinite(self.values[unique])]
        self.values[new]=self._oracle[new];self.stages[stage]=self.stages.get(stage,0)+len(new)
        return self.values[keys]
    def objects(self,ids,stage):
        ids=np.asarray(ids,dtype=int);keys=ids[...,None]*self.r+np.arange(self.r)
        return self.query(keys,stage)

def parameters(a,b,pools,ts,lam):
    geo=s.paired_geometry(a,b,*ts,lam,pools['training'],pools['comparison'])
    pars=[s.matrix_parameters(z[pools['training']],t) for z,t in zip((a,b),ts)]
    jq=[s.matrix_parameters(z[pools['comparison']],t)['weighted']['mean'] for z,t in zip((a,b),ts)]
    M=min(geo['range'],sum(s.centered_bound(t,lam) for t in ts))
    TB=geo['mean_row_train'][1]/(2*lam)
    R=min(1/lam,geo['mean_row_test'][0]*geo['mean_row_difference_train']/lam*(1+TB/4)+geo['mean_row_difference_test']*TB)
    tc=[s.test_constants(z[pools['comparison']],t) for z,t in zip((a,b),ts)]
    dc=[s.cache_design(z,t,lam) for z,t in zip((a,b),ts)]
    return dict(geo=geo,pars=pars,jq=jq,M=M,R=R,tc=tc,dc=dc)

def calibrate(a,b,pools,ts,case,cache,phase,seed,par):
    # These iid indices are generated only after the anchor has been fixed.
    rng=np.random.default_rng(seed+2000000+phase*100000)
    train=rng.choice(pools['training'],32768);test=rng.choice(pools['comparison'],2048)
    yy=cache.objects(test,'calibration');looks=[];before=cache.count
    beta=.01*WEIGHTS[phase]
    for m in (2048,8192,32768):
        yc=cache.objects(train[:m],'calibration');vals=[]
        for j,z in enumerate((a,b)):
            cal=s.calibration(z[train[:m]],yc,ts[j],case['lam'],z[pools['training']],beta/12)
            gr=s.calibration(z[test],yy,ts[j],0.,z[pools['comparison']],beta/12)
            local=min((s.localized_allowance(cal,gr,case['n'],case['lam'],par['pars'][j],par['geo']['Kq'][j],par['jq'][j],a.shape[-1],eta)
                       for eta in (1e-4,1e-5,1e-6,1e-7,1e-8)),key=lambda x:x['allowance'])
            vals.append(dict(global_bound=s.approx_bound(cal,case['n'],case['lam']),local=local))
        eps=sum(min(v['global_bound'],v['local']['allowance']) for v in vals)
        looks.append(dict(m=m,epsilon=eps,candidates=vals))
        if eps<case['g']:break
    return eps,dict(phase=phase,beta=beta,looks=looks,epsilon=eps,labels=cache.count)

def refit_anchor(a,b,pools,cache,lam):
    ids=pools['training'];known=cache.values.reshape(-1,cache.r)[ids];mask=np.isfinite(known)
    return [s.fit(z[ids][mask],known[mask],lam)[0] for z in (a,b)]

def reduced_coefficients(c,v,w,l,h,keys,known_values):
    """Substitute known labels, x_i^2=1 and merge repeated unknown keys."""
    left=np.r_[np.arange(v.shape[1]),l];right=np.r_[np.full(v.shape[1],-1),h]
    output=[]
    for i in range(len(c)):
        constant=float(c[i]);terms={};coef=np.r_[v[i],w[i]]
        for coeff,ll,rr in zip(coef,left,right):
            if coeff==0:continue
            kl=int(keys[i,ll]);kr=int(keys[i,rr]) if rr>=0 else -1
            if kr==kl:constant+=coeff;continue
            vl=known_values[kl];vr=known_values[kr] if kr>=0 else np.nan
            if np.isfinite(vl):coeff*=2*vl-1;kl=-1
            if kr>=0 and np.isfinite(vr):coeff*=2*vr-1;kr=-1
            if kl<0:kl,kr=kr,kl
            if kl<0:constant+=coeff;continue
            if kr>=0 and kl>kr:kl,kr=kr,kl
            key=(kl,kr);terms[key]=terms.get(key,0.)+coeff
        kk=np.array(list(terms),dtype=int).reshape(-1,2);cc=np.array(list(terms.values()),dtype=float)
        keep=cc!=0;output.append((constant,kk[keep],cc[keep]))
    return output

def moment_phase(a,b,pools,case,ts,par,cache,phase,size,seed):
    rng=np.random.default_rng(seed+4000000+phase*100000);known=cache.values.copy();vals=[]
    maxmass=0.;known_only=0;identity_error=0.
    for start in range(0,size,128):
        bs=min(128,size-start);ids=rng.choice(pools['training'],(bs,case['n']));tests=rng.choice(pools['comparison'],bs)
        aa=s.fast_coefficients(a,ids,tests,ts[0],case['lam'],par['tc'][0],par['dc'][0])
        bb=s.fast_coefficients(b,ids,tests,ts[1],case['lam'],par['tc'][1],par['dc'][1])
        c,v,w=[aa[k]-bb[k] for k in range(3)];l,h=aa[3:]
        assert np.max(abs(c)+abs(v).sum(1)+abs(w).sum(1))<=par['M']+1e-8
        nodes=np.stack([tests,ids[:,0],ids[:,1]],1)
        keys=(nodes[:,:,None]*cache.r+np.arange(cache.r)).reshape(bs,-1)
        rows=reduced_coefficients(c,v,w,l,h,keys,known);us=rng.random(bs)
        for i,(constant,kk,cc) in enumerate(rows):
            mass=float(abs(cc).sum());maxmass=max(maxmass,mass)
            assert abs(constant)+mass<=par['M']+1e-8
            if mass:
                prob=abs(cc)/mass;cdf=np.cumsum(prob);cdf[-1]=1.
                ix=min(np.searchsorted(cdf,us[i],side='right'),len(cc)-1);kl,kr=kk[ix]
                x=2*cache.query([kl],'comparison')[0]-1
                if kr>=0:x*=2*cache.query([kr],'comparison')[0]-1
                value=constant+np.sign(cc[ix])*mass*x
            else:value=constant;known_only+=1
            vals.append(value)
        # Identity test uses a fabricated completion, not oracle labels.
        if start==0:
            fake=np.where(np.isfinite(known),2*known-1,np.where(np.arange(len(known))%3==0,1.,-1.))
            direct=s.evaluate((c,v,w,l,h),fake[keys])
            for i,(constant,kk,cc) in enumerate(rows):
                prod=fake[kk[:,0]].copy();pair=kk[:,1]>=0;prod[pair]*=fake[kk[pair,1]]
                identity_error=max(identity_error,abs(direct[i]-constant-cc@prod))
            assert identity_error<1e-9
    vals=np.array(vals);lo,hi,rad=s.interval(vals.sum(),vals@vals,len(vals),par['M'],0,.02*WEIGHTS[phase],1)
    # Coefficient/cdf floating-point guards, independent of unknown labels.
    guard=1e-9*(1+par['M'])
    return lo-guard,hi+guard,dict(mean=float(vals.mean()),variance=float(vals.var(ddof=1)),max_remaining_mass=maxmass,known_only_blocks=known_only,identity_error=identity_error)

def fallback(a,b,pools,case,ts,R,cache,seed):
    rng=np.random.default_rng(seed+6000000);su=sq=0.;num=0;history=[];lo=-np.inf;hi=np.inf;choice='unresolved'
    for end in CHECKS:
        while num<end:
            bs=min(64,end-num);ids=rng.choice(pools['training'],(bs,case['n']));test=rng.choice(pools['comparison'],bs)
            yy=cache.objects(ids,'refit');yt=cache.objects(test,'refit')
            ta,_=s.batch_fit(a[ids],yy,case['lam'],ts[0]);tb,_=s.batch_fit(b[ids],yy,case['lam'],ts[1])
            va=np.einsum('brd,bd->br',a[test],ta);vb=np.einsum('brd,bd->br',b[test],tb)
            z=(np.logaddexp(0,va)-np.logaddexp(0,vb)-yt*(va-vb)).mean(1)
            assert np.max(abs(z))<=R+1e-8
            su+=z.sum();sq+=z@z;num+=bs
        lo,hi,rad=s.interval(su,sq,num,R,2e-9/case['lam'],.02,len(CHECKS));choice=s.decision(lo,hi,case['g'])
        history.append(dict(blocks=num,labels=cache.count,interval=[lo,hi],choice=choice))
        if choice!='unresolved':break
    return choice,lo,hi,history

def run_variant(case,ci,group,a,b,y,pools,mode):
    seed=202611141701+ci*10000000+group*10000;cache=Cache(y)
    pilot=np.random.default_rng(s.SEED+ci*100000+group*1000).choice(pools['training'],512)
    yp=cache.objects(pilot,'pilot');ts=[s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
    history=[];calibrations=[];L=-np.inf;U=np.inf;n=0;choice='unresolved';fallback_used=False
    for phase,end in enumerate(CHECKS):
        if phase>0 and mode=='reanchor':ts=refit_anchor(a,b,pools,cache,case['lam'])
        if phase==0 or mode=='reanchor':
            par=parameters(a,b,pools,ts,case['lam'])
            eps,cal=calibrate(a,b,pools,ts,case,cache,phase,seed,par);calibrations.append(cal)
        print(ci,group,mode,'phase',phase,'epsilon',eps,'labels',cache.count,flush=True)
        if eps>=case['g']:break
        lo,hi,detail=moment_phase(a,b,pools,case,ts,par,cache,phase,end-n,seed)
        lo-=eps;hi+=eps;L=max(L,lo);U=min(U,hi);n=end
        choice='conflict' if L>U else s.decision(L,U,case['g'])
        history.append(dict(phase=phase,blocks=n,labels=cache.count,epsilon=eps,range=par['M'],anchor_norms=[float(np.linalg.norm(t)) for t in ts],
                            alpha=.02*WEIGHTS[phase],interval=[L,U],phase_interval=[lo,hi],choice=choice,**detail))
        if choice!='unresolved':break
    conflict=choice=='conflict'
    if choice in ['unresolved','conflict']:
        fallback_used=True;choice,L,U,refit_history=fallback(a,b,pools,case,ts,par['R'],cache,seed)
    else:refit_history=[]
    before=cache.count
    if choice!='unresolved':
        fi=np.random.default_rng(s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
        yf=cache.objects(fi,'final_training');theta,info=s.fit((a if choice=='A' else b)[fi],yf,case['lam'])
    else:info=None
    audit=read(SHARED/'shared_logistic_full_20261029/audit_results.json')['cases'][case['id']]['difference']
    gap=audit['mean'];se=audit['se'];valid=choice!='unresolved'
    result=dict(method=mode,labels=cache.count,stage_labels=cache.stages,final_new_labels=cache.count-before,choice=choice,interval=[L,U],
                epsilon=eps,blocks=n,fallback=fallback_used,empty_intersection=conflict,history=history,calibrations=calibrations,refit_history=refit_history,
                audit_gap=gap,audit_se=se,covered_audit_mean=bool(L<=gap<=U),excludes_audit_band=bool(U<gap-3*se or L>gap+3*se),
                acceptable_audit_mean=bool(valid and (gap<=case['g'] if choice=='A' else gap>=-case['g'])),
                decision_issued=valid,final_fit=info,seed=seed)
    assert result['labels']==sum(cache.stages.values())
    return result

def worker(job):
    ci,group=job;dest=ROOT/'results'/f'population_{ci:02d}_{group:02d}.json'
    if dest.exists():return job
    with (ROOT/'logs'/f'population_{ci:02d}_{group:02d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            start=time.perf_counter();case=s.CASES[ci];a,o,y,pools=s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
            rows=[run_variant(case,ci,group,a,b,y,pools,mode) for mode in ['fixed_anchor','reanchor']]
            refs=[v for v in read(SHARED/'logistic_cp_revision_20261106/population/records.json') if v['case']==case['id'] and v['path']==group]
            dump(dest,dict(case=case,group=group,records=rows,original=refs,seconds=time.perf_counter()-start));return job
        except Exception:traceback.print_exc();raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--group',type=int);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    dump(ROOT/'population_protocol.json',dict(cases=s.CASES,groups=[0,1,2],paths_per_group=1,checkpoints=CHECKS,
        error_budgets=dict(calibration=.01,moment_sampling=.02,refits=.02),phase_weights=WEIGHTS.tolist(),
        methods=['fixed_anchor_remaining','refitted_anchor_remaining'],
        pilot='Original 512 training draws for groups 0-2; both methods start from the same pilot.',
        anchor='After each phase, refit to all known training-pool attribute rows; unknown labels are not imputed. Fixed-anchor control keeps the initial pilot fit.',
        calibration='Fresh iid training and test calibration draws after each new anchor; 2048/8192/32768 training and 2048 test draws, all distinct purchases charged.',
        inference='Within each phase, freeze anchor and cache substitution; use fresh independent general-valued blocks with EB. Intersect target-risk intervals across phases. No cross-phase pooling.',
        accounting='Distinct pilot, calibration, comparison, refit and final-training labels. No multi-route planning purchases in these two variants.',
        comparison='Archived manuscript method includes its original planning/predictor preparation. Compare the two new methods to isolate reanchoring; original is a descriptive reference, not identical preparation.',
        audit='Independent frozen 2048-fit audit mean and standard error, not exact population truth.',scope='All eight configurations; three independent preparation/query paths each; exploratory.'))
    jobs=[(ci,g) for ci in range(8) for g in range(3) if (args.case is None or ci==args.case) and (args.group is None or g==args.group)]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in jobs}
        for f in as_completed(fs):
            try:j=f.result();status='done'
            except Exception as err:j=fs[f];status=repr(err)
            done=len(list((ROOT/'results').glob('population_*.json')))
            dump(ROOT/'population_progress.json',dict(done=done,total=24,last_job=j,status=status));print(done,'/24',j,status,flush=True)

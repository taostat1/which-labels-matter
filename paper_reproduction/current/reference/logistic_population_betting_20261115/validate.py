"""Exhaustive conditional-unbiasedness and sequential-coverage checks."""
from pathlib import Path
import itertools,json,time
import numpy as np
import experiment as ex
import intervals
ROOT=Path(__file__).resolve().parent

def cache_identity():
    rng=np.random.default_rng(202611151207)
    a=rng.normal(size=(3,2,3));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    b=.7*a+.3*rng.normal(size=a.shape);b/=np.maximum(1,np.linalg.norm(b,axis=-1,keepdims=True))
    ids=np.array(list(itertools.product((0,1),repeat=3)));test=np.full(len(ids),2)
    keys=(np.stack([test,ids[:,0],ids[:,1]],1)[:,:,None]*2+np.arange(2)).reshape(len(ids),-1)
    worst_identity=worst_target=0.;checks=0;targets=0
    for nonzero in (False,True):
        ts=[rng.normal(size=3)*.2 for _ in range(2)] if nonzero else [np.zeros(3),np.zeros(3)]
        tcs=[ex.pop.s.test_constants(z[[2]],t) for z,t in zip((a,b),ts)]
        coeff=[ex.pop.s.fast_coefficients(z,ids,test,t,.4,tc,ex.pop.s.cache_design(z,t,.4))
               for z,t,tc in zip((a,b),ts,tcs)]
        c,v,w=[coeff[0][k]-coeff[1][k] for k in range(3)];l,h=coeff[0][3:]
        for bits in itertools.product((0.,1.),repeat=6):
            y=np.array(bits).reshape(3,2);x=2*y.ravel()-1
            direct=ex.pop.s.evaluate((c,v,w,l,h),x[keys]);full=[]
            for jj in ids:
                rr=[ex.pop.s.quadratic_risk(z[[2]],y[[2]],t,ex.pop.s.step(z[jj],y[jj],t,.4),tc)
                    for z,t,tc in zip((a,b),ts,tcs)]
                full.append(rr[0]-rr[1])
            worst_target=max(worst_target,abs(direct.mean()-np.mean(full)));targets+=1
            for mask in itertools.product((False,True),repeat=6):
                known=np.where(mask,y.ravel(),np.nan)
                rows=ex.pop.p.reduced_coefficients(c,v,w,l,h,keys,known)
                means=[]
                for i,(constant,kk,cc) in enumerate(rows):
                    mass=abs(cc).sum()
                    if mass:
                        prod=x[kk[:,0]].copy();pair=kk[:,1]>=0;prod[pair]*=x[kk[pair,1]]
                        # Independently enumerate the query distribution.
                        zz=constant+np.sign(cc)*mass*prod
                        mean=(abs(cc)/mass)@zz
                    else:mean=constant
                    means.append(mean);checks+=1
                    worst_identity=max(worst_identity,abs(mean-direct[i]))
                assert abs(np.mean(means)-np.mean(full))<2e-12
    assert worst_identity<2e-12 and worst_target<2e-12
    return dict(conditional_query_checks=checks,target_checks=targets,max_query_error=worst_identity,max_target_error=worst_target)

def coverage():
    # Exact enumeration, including history-dependent non-Bernoulli observation
    # laws. Every branch has conditional mean mu; the law changes with history.
    out=[];horizon=11;alpha=.1
    for adaptive in (False,True):
        for mu in (-.7,-.2,0.,.4,.8):
            for adaptive_law in (False,True):
                crossed=0.;mass=0.;expected_cap=[0.,0.];monotone_error=0.
                stack=[([],1.,False)]
                while stack:
                    vals,weight,bad=stack.pop();t=len(vals)
                    if t:
                        logs=[intervals.log_capital(vals,mu,1.,side,adaptive,alpha) for side in (1,-1)]
                        bad=bad or max(logs)>np.log(2/alpha)
                    if t==horizon:
                        mass+=weight;crossed+=weight*bad
                        for k in range(2):expected_cap[k]+=weight*np.exp(logs[k])
                        continue
                    if adaptive_law and t and vals[-1]>mu:
                        low=(mu-1)/2;high=1.
                    else:low=-1.;high=(mu+1)/2 if adaptive_law else 1.
                    prob=(mu-low)/(high-low)
                    stack.extend([(vals+[high],weight*prob,bad),(vals+[low],weight*(1-prob),bad)])
                assert abs(mass-1)<1e-10 and crossed<=alpha+1e-12
                assert max(abs(np.array(expected_cap)-1))<1e-10
                out.append(dict(adaptive_bets=adaptive,adaptive_law=adaptive_law,mean=mu,
                                exact_crossing_probability=crossed,terminal_expected_capitals=expected_cap))
    rng=np.random.default_rng(202611151208);inversions=0
    for adaptive in (False,True):
        for n in (1,2,19,100):
            z=rng.uniform(-1,1,n)
            lo,hi=intervals.interval(z,1.,alpha,adaptive)
            for side in (1,-1):
                logs=np.array([intervals.log_capital(z,m,1.,side,adaptive,alpha) for m in np.linspace(-.999,.999,101)])
                assert np.all(side*np.diff(logs)<=1e-10)
            for m in np.linspace(-.99,.99,101):
                accepted=max(intervals.log_capital(z,m,1.,s,adaptive,alpha) for s in (1,-1))<=np.log(2/alpha)
                assert accepted==(lo<=m<=hi);inversions+=1
    return dict(horizon=horizon,alpha=alpha,law_tests=out,inversion_checks=inversions)

def baseline_parity():
    out=[]
    for ci in (3,6):
        for group in (6,7,8):
            p=ROOT/'results'/f'case_{ci:02d}_{group:02d}.json'
            old=ROOT.parent/'logistic_population_paired_local_20261115/query_traces'/f'case_{ci}_{group}.json'
            new=json.loads(p.read_text());prior=json.loads(old.read_text())
            row=next(v for v in new['records'] if v['method']=='phase_EB')
            assert row['labels']==prior['fallback_labels']
            assert len(row['history'])==len(prior['history'])
            worst=0.
            for now,oldh,detail in zip(row['history'],prior['history'],new['phase_records']):
                assert now['blocks']==oldh['blocks'] and now['labels']==oldh['labels']
                worst=max(worst,float(np.max(abs(np.array(now['raw_interval'])-oldh['raw_interval']))))
                assert abs(detail['mean']-oldh['mean'])<1e-14
                assert abs(detail['variance']-oldh['variance'])<1e-14
            assert worst<1e-12
            out.append(dict(case=ci,group=group,new_labels=row['labels'],old_labels=prior['fallback_labels'],
                            max_interval_difference=worst))
    return out

def main():
    tick=time.perf_counter();out=dict(cache=cache_identity(),coverage=coverage(),parity=baseline_parity())
    out.update(passed=True,seconds=time.perf_counter()-tick)
    (ROOT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

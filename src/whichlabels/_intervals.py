"""Clopper--Pearson intervals for the fixed signed-product sampler."""
import time
import numpy as np
from scipy.stats import beta

def endpoints(k,n,alpha):
    if not (0<=k<=n and n>0 and 0<alpha<1): raise ValueError((k,n,alpha))
    lo=0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi=1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return max(0.,lo-2e-12),min(1.,hi+2e-12)

def mapping_guard(coef,cdf,mass):
    """Worst-case drift from ideal abs-weight probabilities, including RNG grid."""
    actual=np.diff(np.r_[0.,np.clip(cdf,0.,1.)]);ideal=np.abs(coef)/mass
    assert actual.min()>=0
    return float(mass*np.abs(actual-ideal).sum()+mass*len(coef)/2**53)

def query_trial(poly,plan,oracle,rng,checks,g,delta):
    start=time.perf_counter();n=0;k=0;history=[];choice='unresolved'
    coef=poly['coef'];mass=float(np.abs(coef).sum());constant=float(poly['constant'])
    assert np.all(plan['pred']==0), 'CP requires the zero-predictor two-point sampler'
    assert np.allclose(plan['q'],np.abs(coef)/mass,rtol=1e-13,atol=0)
    guard=2e-9
    drift=mapping_guard(coef,plan['cdf'],mass)
    assert drift<guard,(drift,guard)
    for end in checks:
        idx=np.searchsorted(plan['cdf'],rng.random(end-n),side='right')
        xl=2*oracle.query_keys(poly['left'][idx],'comparison')-1
        xr=2*oracle.query_keys(poly['right'][idx],'comparison')-1
        signs=np.sign(coef[idx])*xl*xr
        assert np.all(np.abs(signs)==1)
        k+=int(np.count_nonzero(signs>0));n=end
        pl,pu=endpoints(k,n,delta/len(checks))
        center=constant+mass*(2*k/n-1)
        sl=constant+mass*(2*pl-1);su=constant+mass*(2*pu-1)
        lo=sl-poly['epsilon']-guard;hi=su+poly['epsilon']+guard
        choice='A' if hi<=g else 'B' if lo>=-g else 'unresolved'
        history.append(dict(draws=n,successes=k,estimate=center,
            statistical_radius=max(center-sl,su-center),statistical_lower=sl,statistical_upper=su,
            approximation_radius=poly['epsilon'],lower=lo,upper=hi,labels=oracle.count,
            choice=choice,interval='CP',alpha=delta/len(checks)))
        if choice!='unresolved':break
    return dict(choice=choice,lower=lo,upper=hi,estimate=center,draws=n,
        seconds=time.perf_counter()-start,history=history,mapping_error_bound=drift)

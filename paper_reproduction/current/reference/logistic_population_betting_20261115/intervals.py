"""Bounded-mean confidence sequences following Waudby-Smith & Ramdas (2024).

The finite mixture below uses fixed, prespecified betting fractions. It is a
specialization of the paper's betting framework, not a new concentration law.
"""
import numpy as np
from scipy.special import logsumexp

FRACTIONS=np.array([1/64,1/32,1/16,1/8,1/4,1/2,.75,.9,.99])

def log_capital(values,mean,M,side,adaptive=False,alpha=.02):
    z=np.asarray(values,dtype=float)
    assert np.all(abs(z)<=M+1e-12)
    denom=M+mean if side==1 else M-mean
    if denom<=0:
        if np.all(z==mean):return 0.
        return float('inf')
    if not adaptive:
        terms=side*(z[:,None]-mean)*(FRACTIONS[None]/denom)
        assert np.min(terms)>-1
        return float(logsumexp(np.log1p(terms).sum(0))-np.log(len(FRACTIONS)))
    # Predictable plug-in bets; every variance estimate uses previous values.
    x=(z+M)/(2*M);n=len(x);t=np.arange(1,n+1,dtype=float)
    pred=(.5+np.r_[0.,np.cumsum(x)[:-1]])/t
    resid=(x-pred)**2
    var=(.25+np.r_[0.,np.cumsum(resid)[:-1]])/t
    raw=np.sqrt(2*np.log(2/alpha)/(var*t*np.log1p(t)))/(2*M)
    lam=np.minimum(raw,.5/denom)
    return float(np.log1p(side*lam*(z-mean)).sum())

def interval(values,M,alpha=.02,adaptive=False):
    if not len(values):return -M,M
    critical=np.log(2/alpha)
    # The one-sided capitals are monotone in the tested mean. Return outward
    # roots so numerical root finding only widens the confidence interval.
    lo,hi=-M,M
    if log_capital(values,lo,M,1,adaptive,alpha)>critical:
        a,b=lo,hi
        for _ in range(46):
            m=(a+b)/2
            if log_capital(values,m,M,1,adaptive,alpha)>critical:a=m
            else:b=m
        lo=a
    if log_capital(values,hi,M,-1,adaptive,alpha)>critical:
        a,b=-M,M
        for _ in range(46):
            m=(a+b)/2
            if log_capital(values,m,M,-1,adaptive,alpha)>critical:b=m
            else:a=m
        hi=b
    return float(lo),float(hi)

def eb_phase(values,M,alpha):
    n=len(values);var=np.var(values,ddof=1);t=np.log(4/alpha)
    h=np.sqrt(2*var*t/n)+14*M*t/(3*(n-1))
    m=np.mean(values)
    return float(m-h),float(m+h)

"""Vectorized scalar remainder extrema; same valid local-bound formula."""
import numpy as np
from scipy.special import expit
def remainder(v,lo,hi):
    p=expit(v);e=expit(-abs(v));w=e*(1-e)
    def der(u):return expit(v+u)-p-w*u
    def rem(u):return np.logaddexp(0,v+u)-np.logaddexp(0,v)-p*u-.5*w*u*u
    knots=[lo,np.clip(np.minimum(0.,-2*v),lo,hi),np.clip(np.maximum(0.,-2*v),lo,hi),hi]
    out=np.maximum.reduce([abs(rem(t)) for t in knots])
    for a,b in zip(knots[:-1],knots[1:]):
        a=a.copy();b=b.copy();fa=der(a);fb=der(b);mask=fa*fb<0
        if not mask.any():continue
        for _ in range(60):
            mid=(a+b)/2;fm=der(mid);left=fa*fm<=0
            b=np.where(mask&left,mid,b);a=np.where(mask&~left,mid,a);fa=np.where(mask&~left,fm,fa)
        out=np.maximum(out,np.where(mask,abs(rem((a+b)/2)),0.))
    return out+3e-13
def install():
    import local_core
    local_core.loss_remainder=remainder

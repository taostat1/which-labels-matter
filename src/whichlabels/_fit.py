"""Newton solver and BCE risk from the frozen experiment implementation."""
import numpy as np
from scipy.special import expit

def fit(z,y,lam,tol=1e-10,maxiter=40,start=None):
    """Damped Newton, all coefficients penalized; a residual certifies accuracy."""
    d=z.shape[-1]; z=z.reshape(-1,d); y=y.reshape(-1)
    t=np.zeros(d) if start is None else start.copy(); calls=0
    for it in range(maxiter):
        v=z@t; p=expit(v)
        g=z.T@(p-y)/len(y)+lam*t
        if np.linalg.norm(g)<=tol: break
        h=(z.T*(p*(1-p)))@z/len(y)+lam*np.eye(d)
        s=np.linalg.solve(h,g); old=np.logaddexp(0,v).mean()-y@v/len(y)+lam*t@t/2
        rate=1.
        for _ in range(30):
            u=t-rate*s; vv=z@u
            obj=np.logaddexp(0,vv).mean()-y@vv/len(y)+lam*u@u/2
            calls+=1
            if obj<=old-1e-4*rate*(g@s)+1e-15: break
            rate*=.5
        t=u
    residual=float(np.linalg.norm(z.T@(expit(z@t)-y)/len(y)+lam*t))
    if residual>tol*10: raise RuntimeError(f'Newton residual {residual}')
    return t,dict(iterations=it,objective_calls=calls,residual=residual)


def risk(z,y,t):
    v=np.einsum('ird,d->ir',z,t)
    return float(np.mean(np.logaddexp(0,v)-y*v))

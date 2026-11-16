"""Conditional certificates; never reads an unrevealed response."""
from pathlib import Path
import sys,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_manuscript_paid_anchor_20261114'))
from anchor_core import candidate_geometry,paired_certificate,mapped_bound,cross_bound,sigmoid_weight
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq

def residual_range(v,lo,hi):
    p=expit(v);w=sigmoid_weight(v)
    fun=lambda s:expit(v+s)-p-w*s
    values=[fun(lo),fun(hi)]
    for s in [np.zeros_like(v),-2*v]:
        values.append(np.where((lo<=s)&(s<=hi),fun(s),np.nan))
    a=np.nanmin(values,axis=0)-3e-14;b=np.nanmax(values,axis=0)+3e-14
    return a,b

def loss_remainder(v,lo,hi):
    """Extrema of Taylor remainder: endpoints and roots of its derivative."""
    out=[]
    for vv,l,h in zip(v,lo,hi):
        pp=expit(vv);ww=sigmoid_weight(vv)
        der=lambda u:expit(vv+u)-pp-ww*u
        rem=lambda u:np.logaddexp(0,vv+u)-np.logaddexp(0,vv)-pp*u-.5*ww*u*u
        knots=sorted(set([float(l),float(h)]+[float(t) for t in [0.,-2*vv] if l<t<h]))
        points=list(knots)
        for a,b in zip(knots[:-1],knots[1:]):
            if der(a)*der(b)<0:points.append(brentq(der,a,b,xtol=1e-13))
        out.append(max(abs(rem(u)) for u in points)+3e-13)
    return np.array(out)

def norm_box(z,lo,hi):
    center=(lo+hi)/2;rad=(hi-lo)/2
    return min(mapped_bound(z,np.maximum(abs(lo),abs(hi))),
               np.linalg.norm(z.T@center/len(z))+mapped_bound(z,rad))*(1+2e-12)+2e-14

def refine(old,lam,known_eval=None,iterations=12):
    g=dict(old);z,q=g['z'],g['q'];n=len(z);d=z.shape[1]
    rad=abs(g['V'][:,g['unknown']]).sum(1)
    radq=abs(g['Vq'][:,g['unknown']]).sum(1)
    center=z@g['ck'];centerq=q@g['ck']
    slo,shi=center-rad,center+rad
    rlo,rhi=residual_range(g['v'],slo,shi)
    rho=np.maximum(abs(rlo),abs(rhi))
    # Initial global strong-convexity radius is valid for every completion.
    e=min(g['e'],norm_box(z,rlo,rhi)/lam)
    train_error=g['rows']*e;test_error=g['qrows']*e
    tmax=abs(g['v']+center)+rad
    if known_eval is None:known_eval=np.full(len(q),np.nan)
    known=np.isfinite(known_eval);ycenter=np.where(known,(known_eval+1)/2,.5)
    best_fit=float('inf');best_metric=None
    trace=[]
    for it in range(iterations):
        wmin=sigmoid_weight(tmax+train_error)
        M=lam*np.eye(d)+(z.T*wmin)@z/n
        ev,V=np.linalg.eigh(M);invroot=(V/np.sqrt(ev))@V.T
        zr=z@invroot;qr=q@invroot
        R=norm_box(zr,rlo,rhi)
        newe=R/np.sqrt(ev[0])
        et=R*np.linalg.norm(zr,axis=1);eq=R*np.linalg.norm(qr,axis=1)
        e=min(e,newe);train_error=np.minimum(train_error,et);test_error=np.minimum(test_error,eq)
        lower=g['vq']+centerq-radq-test_error;upper=g['vq']+centerq+radq+test_error
        glo=expit(lower)-ycenter-.5*(~known);ghi=expit(upper)-ycenter+.5*(~known)
        G=norm_box(qr,glo,ghi)
        risk_bound=R*G
        row_bound=np.mean(np.maximum(abs(glo),abs(ghi))*test_error)
        best_fit=min(best_fit,risk_bound,row_bound)
        trace.append(dict(iteration=it,parameter_radius=e,residual_metric=R,gradient_metric=G,fit_bound=best_fit,min_curvature=float(ev[0])))
        best_metric=(M,R)
    # Tightened Euclidean radii can also enter the existing paired argument.
    lower=g['v']+center-rad-train_error;upper=g['v']+center+rad+train_error
    lowweight=sigmoid_weight(np.maximum(abs(lower),abs(upper)))
    highweight=sigmoid_weight(np.where((lower<=0)&(upper>=0),0,np.minimum(abs(lower),abs(upper))))
    w0=sigmoid_weight(g['v']);wdev=np.maximum(abs(lowweight-w0),abs(highweight-w0))
    gl=g['vq']+centerq-radq-test_error;gu=g['vq']+centerq+radq+test_error
    glo=expit(gl)-ycenter-.5*(~known);ghi=expit(gu)-ycenter+.5*(~known)
    grad=norm_box(q,glo,ghi)
    rem=float(loss_remainder(g['vq'],centerq-radq,centerq+radq).mean())
    g.update(e=e,rho=rho,r0=norm_box(g['U'],rlo,rhi),wdev=wdev,
             eta=cross_bound(g['U'],z,wdev),gradient=grad,remainder=min(g['remainder'],rem),
             separate=min(g['separate'],best_fit+rem,grad*e+rem),local_trace=trace,
             train_error=train_error,test_error=test_error,residual_lo=rlo,residual_hi=rhi)
    return g

def certificate(a,b,lam,known_eval=None):
    old=paired_certificate(a,b)
    aa=refine(a,lam,known_eval);bb=refine(b,lam,known_eval)
    new=paired_certificate(aa,bb)
    new['epsilon']=min(new['epsilon'],old['epsilon'])
    return old,new,aa,bb

"""Paired logistic remainder certificates, conditional on acquired labels.

Only public designs and NaN-masked purchased responses enter these routines.
The certificate is uniform over all completions of the unobserved responses.
"""
from pathlib import Path
import os, sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_adaptive_initialization_20261115'))
import fixed_experiment as f
import numpy as np
from scipy.special import expit
from anchor_core import paired_certificate, affine_difference
from local_core import refine, candidate_geometry, mapped_bound, cross_bound, sigmoid_weight

LIP=1/(6*np.sqrt(3))
CRITICAL=2*np.arctanh(1/np.sqrt(3))

def curvature_lipschitz(radius):
    t=np.minimum(np.maximum(radius,0),CRITICAL)
    return sigmoid_weight(t)*np.tanh(t/2)+2e-15

def paired(a,b):
    old=paired_certificate(a,b)
    dz=a['z']-b['z']; dq=a['q']-b['q']; du=a['U']-b['U']
    rd=np.linalg.norm(dz,axis=1); rdq=np.linalg.norm(dq,axis=1)
    dv=abs(a['v']-b['v']); dvq=abs(a['vq']-b['vq'])
    ds=affine_difference(a,b); dsq=affine_difference(a,b,True)
    ell=np.maximum(a['ell'],b['ell']); av=np.maximum(abs(a['v']),abs(b['v']))
    # Derivatives of phi(v,s), uniform along the segment between candidates.
    lu=np.minimum.reduce([np.full(len(ell),.25),curvature_lipschitz(av+ell)*ell,
                          av*ell/8+ell**2/16])
    if not np.any(av):lu=np.minimum(lu,.25-sigmoid_weight(ell)+2e-15)
    lv=np.minimum(ell**2/16,2*curvature_lipschitz(av+ell)*ell)
    dr=min(mapped_bound(du,a['rho'])+mapped_bound(b['U'],lu*ds+lv*dv),
           mapped_bound(du,b['rho'])+mapped_bound(a['U'],lu*ds+lv*dv))
    radius=np.maximum(abs(a['v'])+a['ell']+a['rows']*a['e'],
                      abs(b['v'])+b['ell']+b['rows']*b['e'])
    lip=curvature_lipschitz(radius)
    anchor_weight=abs(sigmoid_weight(a['v'])-sigmoid_weight(b['v']))
    de=a['e']+b['e']; den=[]
    for x,y in ((a,b),(b,a)):
        geom=cross_bound(du,x['z'],x['wdev'])+cross_bound(y['U'],dz,x['wdev'])
        weights=cross_bound(y['U'],y['z'],anchor_weight+lip*(dv+ds+.5*rd*y['e']))
        feedback=.5*cross_bound(y['U'],y['z'],lip*x['rows'])
        denominator=1-x['eta']-y['e']*feedback;den.append(denominator)
        if denominator>0:de=min(de,(dr+y['e']*(geom+weights))/denominator)
    if np.array_equal(a['z'],b['z']) and np.array_equal(a['t'],b['t']):de=0.
    inc=[]
    for x,y in ((a,b),(b,a)):
        # Integration over the error segment supplies the factor one half.
        dl=dvq+dsq+.5*(x['qrows']*de+rdq*y['e'])
        dg=mapped_bound(dq,expit(abs(x['vq'])+x['ellq']+x['qrows']*x['e']))
        dg+=mapped_bound(y['q'],.25*dl)
        inc.append(dg*x['e']+y['gradient']*de)
    eq=np.maximum(a['ellq'],b['ellq']); vq=np.maximum(abs(a['vq']),abs(b['vq']))
    rp=np.minimum(eq**2/(12*np.sqrt(3)),np.minimum(1+.25*eq,vq*eq**2/16+eq**3/48))
    remainder=float(np.mean(rp*dsq+eq**3*dvq/48))
    result=min(inc)+remainder
    return dict(epsilon=float(min(old['epsilon'],result)+1e-12),
                new_paired=float(result),old=float(old['epsilon']),de=float(de),
                residual_difference=float(dr),denominators=den,
                increment=float(min(inc)),remainder=remainder)

def certificate(za,zb,qa,qb,anchors,lam,known_train,known_test):
    aa=candidate_geometry(za,qa,anchors[0],lam,known_train)
    bb=candidate_geometry(zb,qb,anchors[1],lam,known_train)
    a=refine(aa,lam,known_test);b=refine(bb,lam,known_test)
    out=paired(a,b)
    out['old']=float(min(paired_certificate(aa,bb)['epsilon'],paired_certificate(a,b)['epsilon']))
    out['epsilon']=min(out['epsilon'],out['old'])
    return out

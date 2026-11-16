"""Response-uniform population score moments from public finite-feature laws.

For r <= 8, enumeration is over hypothetical binary response vectors, not
acquired responses. No label array or oracle is accepted by these functions.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.special import expit

def score_envelope(z,theta,penalty):
    n,r,d=z.shape
    assert r<=12
    signs=2*((np.arange(2**r)[:,None]>>np.arange(r))&1)-1
    centers=[];maxima=[];gram=np.zeros((d,d));absmean=np.zeros(d)
    for start in range(0,n,256):
        zz=z[start:start+256];pp=expit(np.einsum('ird,d->ir',zz,theta))
        cc=np.einsum('ird,ir->id',zz,pp-.5)/r+penalty*theta
        unknown=np.einsum('kr,ird->ikd',signs,zz,optimize=True)/(2*r)
        sq=np.sum((cc[:,None,:]-unknown)**2,axis=-1)
        maxima.append(sq.max(1));centers.append(cc)
        gram+=np.einsum('ird,ire->de',zz,zz,optimize=True)/(n*r)
        absmean+=abs(zz).sum((0,1))/(n*r)
    maxima=np.concatenate(maxima);center=np.concatenate(centers).mean(0)
    spectral=max(0.,np.linalg.eigvalsh(gram)[-1])
    # Cauchy--Schwarz bounds the mean signed-label contribution by sqrt(K)/2.
    second=float(maxima.mean())
    mu=min(np.sqrt(second),np.linalg.norm(center)+.5*np.sqrt(spectral),np.linalg.norm(abs(center)+.5*absmean))
    guard=1e-12*(1+float(maxima.max()))
    return dict(mu2=float(mu*mu+guard),second=second+guard,B=float(np.sqrt(maxima.max()+guard)),
                feature_objects=n,attributes=r,hypothetical_responses=2**r,mean_gram_norm=float(spectral),
                center_norm=float(np.linalg.norm(center)),certificate='uniform over all binary label completions')

def allowance(a,b,pools,anchors,case,par,study):
    rows=[]
    for j,(z,t) in enumerate(zip((a,b),anchors)):
        train=score_envelope(z[pools['training']],t,case['lam'])
        test=score_envelope(z[pools['comparison']],t,0.)
        local=min((study.localized_allowance(train,test,case['n'],case['lam'],par['pars'][j],par['geo']['Kq'][j],par['jq'][j],z.shape[-1],eta)
                   for eta in [1e-4,1e-5,1e-6,1e-7,1e-8]),key=lambda d:d['allowance'])
        global_bound=study.approx_bound(train,case['n'],case['lam'])
        rows.append(dict(training_moments=train,test_gradient=test,local=local,global_bound=global_bound,used=min(global_bound,local['allowance'])))
    return float(sum(r['used'] for r in rows)+1e-10),rows

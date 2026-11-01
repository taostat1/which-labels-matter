"""Polynomial and inspection interval from the frozen CV implementation."""
import time
import numpy as np
from scipy import sparse
from ._bounds import bound

def decision(lo,hi,g):return 'A' if hi<=g else 'B' if lo>=-g else 'unresolved'


def polynomial(a,b,splits,lam):
    """Exact sparse polynomial of the zero-anchor one-step quadratic CV score."""
    tick=time.perf_counter();p=a.shape[0]*a.shape[1];r=a.shape[1];d=a.shape[-1]
    row_parts=[];col_parts=[];value_parts=[];constant=0.;bounds=[]
    surrogate_parameters=[]
    for tr,te in splits:
        trainkeys=(tr[:,None]*r+np.arange(r)).ravel()
        testkeys=(te[:,None]*r+np.arange(r)).ravel()
        u,v=np.triu_indices(len(trainkeys),1)
        quad=np.zeros((len(trainkeys),len(trainkeys)));cross=np.zeros((len(testkeys),len(trainkeys)))
        allowance=0.;ls=[]
        for sign,z in ((1.,a),(-1.,b)):
            zt=z[tr].reshape(-1,d);zq=z[te].reshape(-1,d)
            H=lam*np.eye(d)+zt.T@zt/(4*len(zt))
            L=np.linalg.solve(H,zt.T)/(2*len(zt))
            C=zq.T@zq/(4*len(zq))
            quad+=sign*.5*(L.T@C@L)
            cross-=sign*(zq@L)/(2*len(zq))
            allowance+=bound(z[tr],z[te],lam)['epsilon'];ls.append(L)
        constant+=float(np.trace(quad))/len(splits)
        qvalues=(quad[u,v]+quad[v,u])/len(splits)
        row_parts.append(np.minimum(trainkeys[u],trainkeys[v]).astype(np.int32))
        col_parts.append(np.maximum(trainkeys[u],trainkeys[v]).astype(np.int32));value_parts.append(qvalues)
        left=np.repeat(testkeys,len(trainkeys));right=np.tile(trainkeys,len(testkeys))
        row_parts.append(np.minimum(left,right).astype(np.int32));col_parts.append(np.maximum(left,right).astype(np.int32))
        value_parts.append(cross.ravel()/len(splits));bounds.append(allowance);surrogate_parameters.append(ls)
    mat=sparse.coo_matrix((np.concatenate(value_parts),(np.concatenate(row_parts),np.concatenate(col_parts))),shape=(p,p)).tocsr()
    mat.sum_duplicates();mat.eliminate_zeros();co=mat.tocoo()
    result=dict(constant=constant,left=co.row.copy(),right=co.col.copy(),coef=co.data.copy(),
                epsilon=float(np.mean(bounds)),epsilon_max=float(np.max(bounds)),
                split_allowances=np.asarray(bounds),terms=int(co.nnz),seconds=time.perf_counter()-tick,
                parameters=surrogate_parameters)
    return result


def query_plan(poly,hp,r):
    tick=time.perf_counter();left=poly['left'];right=poly['right'];coef=poly['coef']
    pred=hp['singles'].ravel()[left]*hp['singles'].ravel()[right]
    same=left//r==right//r
    pred[same]=hp['pairs'][left[same]//r,hp['mapping'][left[same]%r,right[same]%r]]
    ab=abs(coef);mass=ab.sum();scores=ab*np.sqrt(np.maximum(1-pred*pred,.01))
    probability=.9*scores/scores.sum()+.1*ab/mass
    cdf=np.cumsum(probability);cdf[-1]=1.
    ratio=coef/probability
    low=float(np.min(np.minimum(ratio*(-1-pred),ratio*(1-pred))))
    high=float(np.max(np.maximum(ratio*(-1-pred),ratio*(1-pred))))
    return dict(pred=pred,q=probability,cdf=cdf,base=float(poly['constant']+coef@pred),
                low=low,high=high,width=high-low,seconds=time.perf_counter()-tick)


def query_trial(poly,plan,oracle,rng,checks,g,delta):
    tick=time.perf_counter();su=0.;sq=0.;n=0;history=[];choice='unresolved'
    for end in checks:
        idx=np.searchsorted(plan['cdf'],rng.random(end-n),side='right')
        xl=2*oracle.query_keys(poly['left'][idx],'comparison')-1
        xr=2*oracle.query_keys(poly['right'][idx],'comparison')-1
        residual=poly['coef'][idx]/plan['q'][idx]*(xl*xr-plan['pred'][idx])
        assert residual.min()>=plan['low']-1e-10 and residual.max()<=plan['high']+1e-10
        su+=float(residual.sum());sq+=float(residual@residual);n=end
        variance=max(0.,(sq-su*su/n)/(n-1));t=np.log(4*len(checks)/delta)
        stat=np.sqrt(2*variance*t/n)+7*plan['width']*t/(3*(n-1))
        center=plan['base']+su/n;half=stat+poly['epsilon']+2e-9
        lo,hi=center-half,center+half;choice=decision(lo,hi,g)
        history.append(dict(draws=n,estimate=center,statistical_radius=stat,approximation_radius=poly['epsilon'],
                            lower=lo,upper=hi,labels=oracle.count,choice=choice))
        if choice!='unresolved':break
    return dict(choice=choice,lower=lo,upper=hi,estimate=center,draws=n,
                seconds=time.perf_counter()-tick,history=history)


def audit_polynomial(poly,y,splits,a,b,lam):
    x=(2*y-1).ravel();sparse_value=poly['constant']+float(np.sum(poly['coef']*x[poly['left']]*x[poly['right']]))
    values=[];parameter_errors=[];risk_errors=[]
    for split_idx,(tr,te) in enumerate(splits):
        row=[]
        for j,z in enumerate((a,b)):
            t=poly['parameters'][split_idx][j]@(2*y[tr]-1).ravel()
            logits=np.einsum('ird,d->ir',z[te],t)
            row.append(float(np.mean(np.log(2)+(.5-y[te])*logits+logits*logits/8)))
        values.append(row[0]-row[1])
    direct=float(np.mean(values))
    assert abs(direct-sparse_value)<2e-10,(direct,sparse_value)
    return dict(surrogate_delta=direct,polynomial_delta=sparse_value,identity_error=abs(direct-sparse_value))

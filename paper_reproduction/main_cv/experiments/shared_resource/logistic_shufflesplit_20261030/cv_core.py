"""Finite-schedule ShuffleSplit comparison with a strict attribute-label ledger."""
from pathlib import Path
import os,sys,time,json,hashlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'_deps'))
sys.path.insert(0,str(ROOT.parent/'shared_logistic_full_20261029'))
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.model_selection import ShuffleSplit
import core as old
from diagnose import bound

def save(path,obj):old.save(path,obj)

def progress(stage,done,total,note,state='running'):
    item=dict(stage=stage,done=done,total=total,percent=100*done/max(total,1),note=note,
              state=state,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'progress.json',item)
    (ROOT/'进程.txt').write_text(
        f"GoEmotions ShuffleSplit 迁移实验\n更新时间：{item['updated']}\nPID：{os.getpid()}\n"
        f"阶段：{stage}\n进度：{done}/{total}（{item['percent']:.1f}%）\n状态：{state}\n{note}\n"
        '进度表示计算完成比例，不表示决策成功率。\n',encoding='utf-8-sig')

class Oracle:
    def __init__(self,y):
        self._y=y;self.r=y.shape[1];self.seen=np.zeros(y.size,dtype=bool)
        self.stage_counts={};self.requests={}
    def query_keys(self,keys,stage):
        keys=np.asarray(keys,dtype=np.int64)
        unique=np.unique(keys);new=int((~self.seen[unique]).sum());self.seen[unique]=True
        self.stage_counts[stage]=self.stage_counts.get(stage,0)+new
        self.requests[stage]=self.requests.get(stage,0)+keys.size
        return self._y.reshape(-1)[keys]
    def query_objects(self,ids,stage):
        ids=np.asarray(ids)
        return self.query_keys(ids[...,None]*self.r+np.arange(self.r),stage)
    @property
    def count(self):return int(self.seen.sum())

def key_union(splits,r):
    return np.unique(np.concatenate([np.r_[a,b] for a,b in splits])[:,None]*r+np.arange(r))

def fit(z,y,lam,start=None):return old.fit(z,y,lam,start=start)

def mean_loss(z,y,theta):return old.risk(z,y,theta)

def exact_cv(a,b,oracle,splits,lam):
    tick=time.perf_counter();loss=[];fits=[];iterations=0
    for tr,te in splits:
        yt=oracle.query_objects(tr,'comparison');yq=oracle.query_objects(te,'comparison')
        row=[];tt=[]
        for z in (a,b):
            theta,info=fit(z[tr],yt,lam);iterations+=info['iterations'];tt.append(theta)
            row.append(mean_loss(z[te],yq,theta))
        loss.append(row);fits.append(tt)
    return dict(delta=float(np.mean(np.array(loss)[:,0]-np.array(loss)[:,1])),
                split_losses=np.asarray(loss),fits=np.asarray(fits),iterations=iterations,
                seconds=time.perf_counter()-tick)

def approximate_cv(a,b,oracle,splits,lam,g):
    tick=time.perf_counter();all_y=oracle.query_objects(np.arange(len(a)),'preparation')
    anchors=[fit(z,all_y,lam)[0] for z in (a,b)];values=[];bounds=[];iterations=0
    for tr,te in splits:
        yt=oracle.query_objects(tr,'comparison');yq=oracle.query_objects(te,'comparison')
        row=[];err=0.
        for z,t in zip((a,b),anchors):
            d=z.shape[-1];zt=z[tr].reshape(-1,d);zq=z[te].reshape(-1,d)
            p=expit(zt@t);h=(zt.T*(p*(1-p)))@zt/len(zt)+lam*np.eye(d)
            step=-np.linalg.solve(h,zt.T@(p-yt.reshape(-1))/len(zt)+lam*t)
            one=t+step
            residual=np.linalg.norm(zt.T@(expit(zt@one)-yt.reshape(-1))/len(zt)+lam*one)
            distance=residual/lam
            tq=expit(zq@one)
            grad=np.linalg.norm(zq.T@(tq-yq.reshape(-1))/len(zq))
            kq=np.linalg.eigvalsh(zq.T@zq/len(zq))[-1]
            err+=grad*distance+kq*distance**2/8
            row.append(mean_loss(z[te],yq,one))
        values.append(row[0]-row[1]);bounds.append(err)
    estimate=float(np.mean(values));epsilon=float(np.mean(bounds))
    choice=decision(estimate-epsilon,estimate+epsilon,g)
    did_fallback=choice=='unresolved'
    if did_fallback:
        x=exact_cv(a,b,oracle,splits,lam);estimate=x['delta'];epsilon=2e-9
        choice=decision(estimate-epsilon,estimate+epsilon,g);iterations=x['iterations']
    return dict(estimate=estimate,epsilon=epsilon,choice=choice,fallback=did_fallback,
                seconds=time.perf_counter()-tick,iterations=iterations,anchors=np.asarray(anchors))

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

def helper(a,b,pilot,oracle,penalty=.01):
    tick=time.perf_counter();y=oracle.query_objects(pilot,'preparation');r=y.shape[1]
    singles=np.empty((len(a),r));pairs=np.empty((len(a),r*(r-1)//2));mapping=np.full((r,r),-1,int)
    for k in range(r):
        z=np.c_[a[:,k],b[:,k],np.ones(len(a))]
        theta,_=fit(z[pilot,None,:],y[:,k,None],penalty)
        singles[:,k]=np.clip(2*expit(z@theta)-1,-.98,.98)
    for k,(i,j) in enumerate(zip(*np.triu_indices(r,1))):
        z=np.c_[(a[:,i]+a[:,j])/2,(b[:,i]+b[:,j])/2,np.ones(len(a))]
        theta,_=fit(z[pilot,None,:],(y[:,i]==y[:,j]).astype(float)[:,None],penalty)
        pairs[:,k]=np.clip(2*expit(z@theta)-1,-.98,.98);mapping[i,j]=mapping[j,i]=k
    return dict(singles=singles,pairs=pairs,mapping=mapping,seconds=time.perf_counter()-tick)

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

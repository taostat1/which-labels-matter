"""Shared model bounds and a finite-population exact-CV stopping baseline."""
from pathlib import Path
import os, sys, json, time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'logistic_shufflesplit_20261030'
sys.path.insert(0,str(PREVIOUS))
import cv_core as base
import study
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.model_selection import KFold

def save(path,obj):base.save(path,obj)

def progress(stage,done,total,note,state='running'):
    item=dict(stage=stage,done=done,total=total,percent=100*done/max(total,1),note=note,
              state=state,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'progress.json',item)
    (ROOT/'进程.txt').write_text(f"序贯交叉验证与普通 K 折实验\n更新时间：{item['updated']}\n"
        f"阶段：{stage}\n进度：{done}/{total}（{item['percent']:.1f}%）\n状态：{state}\n"
        f"PID：{item['pid']}\n{note}\n进度是计算完成比例，不是决策成功率。\n",encoding='utf-8-sig')

def op_gram(z):
    z=z.reshape(-1,z.shape[-1]);return max(0.,float(np.linalg.eigvalsh(z.T@z/len(z))[-1]))

def public_bounds(a,b,splits,lam):
    """All binary labelings satisfy the returned per-split risk-difference bounds."""
    tick=time.perf_counter();out=[]
    for tr,te in splits:
        ks=[op_gram(z[tr]) for z in (a,b)];kqs=[op_gram(z[te]) for z in (a,b)]
        details=[base.bound(z[tr],z[te],lam) for z in (a,b)]
        norms=[]
        for k,detail in zip(ks,details):
            s=detail['radius'];dist=min(k*s**3/48,k*s*s/(12*np.sqrt(3)))/lam
            norms.append(min(.5*np.sqrt(k)/lam,np.sqrt(2*np.log(2)/lam),s+dist))
        da=op_gram(a[tr]-b[tr]);dq=op_gram(a[te]-b[te]);ta,tb=norms;ka,kb=ks;qa,qb=kqs
        pd=min(ta+tb,np.sqrt(da)*(expit(tb)+.25*np.sqrt(kb)*tb)/lam,
               np.sqrt(da)*(expit(ta)+.25*np.sqrt(ka)*ta)/lam)
        paired=expit(max(ta,tb))*min(np.sqrt(qa)*pd+np.sqrt(dq)*tb,np.sqrt(qb)*pd+np.sqrt(dq)*ta)
        lo=-.5*np.sqrt(qa)*ta-.5*np.sqrt(qb)*tb-qb*tb*tb/8
        hi=.5*np.sqrt(qa)*ta+.5*np.sqrt(qb)*tb+qa*ta*ta/8
        out.append(dict(lower=max(lo,-paired)-2e-9,upper=min(hi,paired)+2e-9,
                        epsilon=sum(x['epsilon'] for x in details),norms=norms,paired_bound=paired))
    return dict(lower=np.array([x['lower'] for x in out]),upper=np.array([x['upper'] for x in out]),
                epsilon=float(np.mean([x['epsilon'] for x in out])),details=out,seconds=time.perf_counter()-tick)

def _log_wealth(values,total,mu,fractions,upper=False):
    """Mixture of fixed fractions of admissible finite-population bets."""
    values=np.asarray(values,float)
    if upper:values=1-values;mu=1-mu
    previous=np.r_[0.,np.cumsum(values)[:-1]]
    conditional=(total*mu-previous)/(total-np.arange(len(values)))
    # Inversions stay in the feasible finite-population mean interval.
    if np.any(conditional < -1e-12):return np.inf
    denominator=np.maximum(conditional,1e-15)
    ratios=values/denominator
    factors=1+np.asarray(fractions)[:,None]*(ratios[None,:]-1)
    return float(logsumexp(np.log(np.maximum(factors,1e-300)).sum(axis=1))-np.log(len(fractions)))

def betting_interval(values,total,lower,upper,delta,fractions):
    """Two-sided 1-delta confidence sequence for a uniformly permuted finite list."""
    if not len(values):return float(lower),float(upper)
    values=np.asarray(values,float)
    if len(values)==total:
        val=float(values.mean());return val-2e-9,val+2e-9
    width=upper-lower
    if width<=1e-14:return float(lower),float(upper)
    x=(values-lower)/width
    assert x.min()>=-1e-7 and x.max()<=1+1e-7,(x.min(),x.max(),lower,upper)
    x=np.clip(x,0,1);left=float(x.sum()/total);right=float((x.sum()+total-len(x))/total)
    threshold=np.log(2/delta)
    lo=left;hi=right
    if _log_wealth(x,total,left,fractions)>threshold:
        l,r=left,right
        for _ in range(38):
            mid=(l+r)/2
            if _log_wealth(x,total,mid,fractions)>threshold:l=mid
            else:r=mid
        lo=l  # outward rounding of the lower endpoint
    if _log_wealth(x,total,right,fractions,True)>threshold:
        l,r=left,right
        for _ in range(38):
            mid=(l+r)/2
            if _log_wealth(x,total,mid,fractions,True)>threshold:r=mid
            else:l=mid
        hi=r  # outward rounding of the upper endpoint
    return lower+width*lo-2e-9,lower+width*hi+2e-9

def sequential_cv(a,b,oracle,splits,lam,g,bounds,rng,delta,fractions):
    """Never accesses a split loss until its required labels have been purchased."""
    tick=time.perf_counter();B=len(splits);cache={};iterations=0;history=[];prefix=[]
    permutation=rng.permutation(B)
    needed=[np.unique(np.r_[tr,te]) for tr,te in splits]
    def evaluate(i):
        nonlocal iterations
        if i in cache:return
        tr,te=splits[i];yt=oracle.query_objects(tr,'comparison');yq=oracle.query_objects(te,'comparison')
        losses=[]
        for z in (a,b):
            theta,meta=base.fit(z[tr],yt,lam);iterations+=meta['iterations']
            losses.append(base.mean_loss(z[te],yq,theta))
        cache[i]=float(losses[0]-losses[1])
        assert bounds['lower'][i]<=cache[i]<=bounds['upper'][i]
    lo=float(bounds['lower'].mean());hi=float(bounds['upper'].mean())
    choice=base.decision(lo,hi,g);empty_protection=False
    def update_interval():
        nonlocal lo,hi,choice,empty_protection
        unobserved=[j for j in range(B) if j not in cache]
        detlo=(sum(cache.values())+bounds['lower'][unobserved].sum())/B-2e-9
        dethi=(sum(cache.values())+bounds['upper'][unobserved].sum())/B+2e-9
        clo,chi=betting_interval(prefix,B,float(bounds['lower'].min()),float(bounds['upper'].max()),delta,fractions)
        lo=max(lo,detlo,clo);hi=min(hi,dethi,chi)
        if lo>hi:
            empty_protection=True
            for j in range(B):evaluate(j)
            val=float(np.mean(list(cache.values())));lo=val-2e-9;hi=val+2e-9
        choice=base.decision(lo,hi,g)
        history.append(dict(prefix_splits=len(prefix),computed_splits=len(cache),lower=lo,upper=hi,
                            labels=oracle.count,choice=choice))
    for i in permutation:
        if choice!='unresolved':break
        evaluate(int(i));prefix.append(cache[int(i)])
        update_interval()
        if choice!='unresolved':break
        # Extra already-paid labels can make other exact splits free in annotation.
        # Their values tighten a deterministic interval; betting uses only the
        # uniform permutation prefix, so this reuse does not alter its sampling law.
        complete=oracle.seen.reshape(-1,oracle.r).all(axis=1)
        for j in range(B):
            if j not in cache and complete[needed[j]].all():
                evaluate(j);update_interval()
                if choice!='unresolved':break
    assert choice!='unresolved' and lo<=hi+1e-8
    return dict(choice=choice,lower=lo,upper=hi,seconds=time.perf_counter()-tick,
                computed_splits=len(cache),prefix_splits=len(prefix),iterations=iterations,history=history,
                feature_only=len(prefix)==0,empty_protection=empty_protection)

def paired_moment_trial(poly,plan,attribute_oracle,object_oracle,rng,checks,g,delta):
    """Exactly matched statistics and stopping; only query granularity changes."""
    tick=time.perf_counter();su=0.;sq=0.;n=0;history=[];query_seconds=[0.,0.]
    for end in checks:
        idx=np.searchsorted(plan['cdf'],rng.random(end-n),side='right')
        l=poly['left'][idx];r=poly['right'][idx]
        qt=time.perf_counter()
        xl=2*attribute_oracle.query_keys(l,'comparison')-1;xr=2*attribute_oracle.query_keys(r,'comparison')-1
        query_seconds[0]+=time.perf_counter()-qt
        qt=time.perf_counter()
        object_oracle.query_objects(np.unique(np.r_[l,r]//object_oracle.r),'comparison')
        query_seconds[1]+=time.perf_counter()-qt
        residual=poly['coef'][idx]/plan['q'][idx]*(xl*xr-plan['pred'][idx])
        assert residual.min()>=plan['low']-1e-10 and residual.max()<=plan['high']+1e-10
        su+=float(residual.sum());sq+=float(residual@residual);n=end
        variance=max(0.,(sq-su*su/n)/(n-1));t=np.log(4*len(checks)/delta)
        half=np.sqrt(2*variance*t/n)+7*plan['width']*t/(3*(n-1))+poly['epsilon']+2e-9
        center=plan['base']+su/n;lo=center-half;hi=center+half;choice=base.decision(lo,hi,g)
        history.append(dict(draws=n,lower=lo,upper=hi,choice=choice,attribute_labels=attribute_oracle.count,object_labels=object_oracle.count))
        if choice!='unresolved':break
    elapsed=time.perf_counter()-tick
    assert attribute_oracle.count<=object_oracle.count
    return dict(choice=choice,lower=lo,upper=hi,draws=n,history=history,
                attribute_seconds=elapsed-query_seconds[1],object_seconds=elapsed-query_seconds[0])

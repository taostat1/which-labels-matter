"""Same-permission sequential active-CV comparator from the paper."""
import time
import numpy as np
from scipy.special import expit, logsumexp
from ._bounds import bound
from ._moments import decision
from .solver import fit

def op_gram(z):
    z=z.reshape(-1,z.shape[-1]);return max(0.,float(np.linalg.eigvalsh(z.T@z/len(z))[-1]))


def public_bounds(a,b,splits,lam):
    """All binary labelings satisfy the returned per-split risk-difference bounds."""
    tick=time.perf_counter();out=[]
    for tr,te in splits:
        ks=[op_gram(z[tr]) for z in (a,b)];kqs=[op_gram(z[te]) for z in (a,b)]
        details=[bound(z[tr],z[te],lam) for z in (a,b)]
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

class InnerCS:
    """Fixed-grid empirical-Bernstein test supermartingales, valid at all times."""
    def __init__(self,low,high,delta,lambdas):
        self.low=low;self.high=high;self.width=high-low;self.delta=delta
        self.lambdas=np.asarray(lambdas);self.psi=-np.log1p(-self.lambdas)-self.lambdas
        self.n=0;self.sum=0.;self.variation=0.;self.lo=low;self.hi=high
    def add(self,values):
        if self.width<=1e-15:return
        x=(np.asarray(values)-self.low)/self.width
        assert x.min()>=-1e-8 and x.max()<=1+1e-8
        x=np.clip(x,0,1)
        before=self.sum+np.r_[0.,np.cumsum(x)[:-1]]
        predicted=(.5+before)/(np.arange(len(x))+self.n+1)
        self.variation+=float(np.sum((x-predicted)**2));self.sum+=float(x.sum());self.n+=len(x)
        radius=np.min((np.log(2*len(self.lambdas)/self.delta)+self.psi*self.variation)/(self.lambdas*self.n))
        self.lo=max(self.lo,self.low+self.width*(self.sum/self.n-radius))
        self.hi=min(self.hi,self.low+self.width*(self.sum/self.n+radius))


class FoldEvaluation:
    def __init__(self,a,b,oracle,tr,te,lam,bounds,idx,delta,cfg,rng):
        self.oracle=oracle;self.idx=idx;self.rng=rng;self.cfg=cfg;self.n=0
        r=oracle.r;labels=oracle.query_objects(tr,'cv_training')
        self.iterations=0;logits=[]
        for z in (a,b):
            theta,meta=fit(z[tr],labels,lam);self.iterations+=meta['iterations']
            logits.append(np.einsum('ird,d->ir',z[te],theta).ravel())
        va,vb=logits;difference=va-vb
        self.constant=float(np.mean(np.logaddexp(0,va)-np.logaddexp(0,vb)-.5*difference))
        self.coefficient=-difference/(2*len(difference))
        self.keys=(te[:,None]*r+np.arange(r)).ravel()
        self.bound_lo=float(bounds['lower'][idx]);self.bound_hi=float(bounds['upper'][idx])
        self.lo=self.bound_lo;self.hi=self.bound_hi
        known=oracle.seen[self.keys];unknown=np.flatnonzero(~known & (self.coefficient!=0))
        # Every prediction uses only already-paid training labels. No evaluation
        # response is used to train either candidate or this frozen predictor.
        means=(labels.sum(axis=0)+.5)/(len(tr)+1)
        pred=np.tile(2*means-1,len(te))[unknown]
        coef=self.coefficient[unknown]
        self.unknown=unknown;self.pred=pred
        value=self.constant
        if known.any():value+=float(self.coefficient[known]@(2*oracle.query_keys(self.keys[known],'evaluation')-1))
        value+=float(coef@pred);self.base=value
        if len(unknown):
            mass=abs(coef);score=mass*np.sqrt(np.maximum(1-pred**2,.01))
            self.q=.9*score/score.sum()+.1*mass/mass.sum()
            self.cdf=np.cumsum(self.q);self.cdf[-1]=1.
            ratio=coef/self.q
            low=float(np.min(np.minimum(ratio*(-1-pred),ratio*(1-pred))))
            high=float(np.max(np.maximum(ratio*(-1-pred),ratio*(1-pred))))
            self.cs=InnerCS(low,high,delta,cfg['inner_lambdas'])
        else:
            self.q=np.empty(0);self.cdf=np.empty(0);self.cs=None
        self.refresh()
    def refresh(self):
        known=self.oracle.seen[self.keys];coef=self.coefficient
        value=self.constant
        if known.any():value+=float(coef[known]@(2*self.oracle.query_keys(self.keys[known],'evaluation')-1))
        remainder=float(abs(coef[~known]).sum())
        lo=value-remainder-2e-9;hi=value+remainder+2e-9
        if self.cs is not None and self.n:
            lo=max(lo,self.base+self.cs.lo-2e-9);hi=min(hi,self.base+self.cs.hi+2e-9)
        self.lo=max(self.lo,lo);self.hi=min(self.hi,hi)
        self.exact=remainder<=1e-14
        if self.lo>self.hi:
            self.complete()
    def expected_new(self):
        if self.cs is None or self.exact:return 0.
        batch=self.cfg['active_initial_draws'] if self.n==0 else self.n
        unseen=~self.oracle.seen[self.keys[self.unknown]]
        return float(np.sum(-np.expm1(batch*np.log1p(-np.minimum(self.q[unseen],1-1e-15)))))
    def sample(self):
        if self.exact:return
        if self.n>=self.cfg['active_max_draws_per_fold']:
            self.complete();return
        batch=self.cfg['active_initial_draws'] if self.n==0 else self.n
        batch=min(batch,self.cfg['active_max_draws_per_fold']-self.n)
        picks=np.searchsorted(self.cdf,self.rng.random(batch),side='right')
        keys=self.keys[self.unknown[picks]]
        xx=2*self.oracle.query_keys(keys,'evaluation')-1
        residual=self.coefficient[self.unknown[picks]]/self.q[picks]*(xx-self.pred[picks])
        self.cs.add(residual);self.n+=batch;self.refresh()
    def complete(self):
        x=2*self.oracle.query_keys(self.keys,'evaluation')-1
        value=float(self.constant+self.coefficient@x)
        self.lo=value-2e-9;self.hi=value+2e-9;self.exact=True


def interval_from_folds(folds,order,bounds,delta,fractions,midpoint=False):
    B=len(bounds['lower']);seen=order[:len(folds)]
    lower=np.array([folds[i].lo for i in seen]);upper=np.array([folds[i].hi for i in seen])
    if midpoint:lower=upper=(lower+upper)/2
    unknown=order[len(folds):]
    detlo=float((lower.sum()+bounds['lower'][unknown].sum())/B)
    dethi=float((upper.sum()+bounds['upper'][unknown].sum())/B)
    glo=float(bounds['lower'].min());ghi=float(bounds['upper'].max())
    betlo=betting_interval(lower,B,glo,ghi,delta,fractions)[0]
    bethi=betting_interval(upper,B,glo,ghi,delta,fractions)[1]
    return max(detlo,betlo),min(dethi,bethi)


def active_cv(a,b,oracle,splits,lam,g,bounds,cfg,permutation_seed,query_seed):
    tick=time.perf_counter();B=len(splits);order=list(np.random.default_rng(permutation_seed).permutation(B))
    folds={};history=[];events=[];delta_outer=cfg['delta']/2;delta_inner=cfg['delta']/(2*B)
    lo=float(bounds['lower'].mean());hi=float(bounds['upper'].mean());choice=decision(lo,hi,g)
    empty_protection=False
    while choice=='unresolved':
        for fold in folds.values():fold.refresh()
        can_refine=[i for i in order[:len(folds)] if not folds[i].exact]
        if not folds:
            action='open'
        elif len(folds)==B:
            action='refine' if can_refine else 'finish'
        elif not can_refine:
            action='open'
        else:
            midlo,midhi=interval_from_folds(folds,order,bounds,delta_outer,cfg['outer_betting_fractions'],True)
            between=max(0.,midhi-midlo);within=max(0.,(hi-lo)-between)
            action='open' if between>=within else 'refine'
        if action=='open':
            i=order[len(folds)];tr,te=splits[i]
            folds[i]=FoldEvaluation(a,b,oracle,tr,te,lam,bounds,i,delta_inner,cfg,
                                   np.random.default_rng(query_seed+1009*int(i)))
            if not folds[i].exact:folds[i].sample()
        elif action=='refine':
            i=max(can_refine,key=lambda j:(folds[j].hi-folds[j].lo)/(1+folds[j].expected_new()))
            folds[i].sample()
        else:
            i=-1
        for fold in folds.values():fold.refresh()
        clo,chi=interval_from_folds(folds,order,bounds,delta_outer,cfg['outer_betting_fractions'])
        lo=max(lo,clo);hi=min(hi,chi)
        if lo>hi or len(history)>1024:
            empty_protection=True
            for j in order:
                if j not in folds:
                    tr,te=splits[j];folds[j]=FoldEvaluation(a,b,oracle,tr,te,lam,bounds,j,delta_inner,cfg,np.random.default_rng(query_seed+1009*int(j)))
                folds[j].complete()
            val=float(np.mean([(f.lo+f.hi)/2 for f in folds.values()]));lo=val-2e-9;hi=val+2e-9
        choice=decision(lo,hi,g)
        history.append(dict(action=action,fold=int(i),opened_folds=len(folds),draws=sum(f.n for f in folds.values()),
                            lower=lo,upper=hi,choice=choice,labels=oracle.count,
                            training_labels=oracle.stage_counts.get('cv_training',0),evaluation_labels=oracle.stage_counts.get('evaluation',0)))
        if action=='finish' and choice=='unresolved':raise RuntimeError('Exact finite CV should support a decision')
    return dict(choice=choice,lower=lo,upper=hi,seconds=time.perf_counter()-tick,opened_folds=len(folds),
                evaluation_draws=sum(f.n for f in folds.values()),history=history,empty_protection=empty_protection,
                fold_intervals={str(i):[f.lo,f.hi] for i,f in folds.items()})


def zero_helper(objects,r):
    mapping=np.full((r,r),-1,int)
    for k,(i,j) in enumerate(zip(*np.triu_indices(r,1))):mapping[i,j]=mapping[j,i]=k
    return dict(singles=np.zeros((objects,r)),pairs=np.zeros((objects,r*(r-1)//2)),mapping=mapping,seconds=0.)

"""Exact logistic evaluation with selective attributes inside sequential CV."""
from pathlib import Path
import os,sys,time,json
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EARLIER=ROOT.parent/'logistic_cv_earlystop_kfold_20261030'
sys.path.insert(0,str(EARLIER))
import extensions as previous
from extensions import base,study,np,expit

def save(path,obj):base.save(path,obj)

def progress(stage,done,total,note,state='running'):
    item=dict(stage=stage,done=done,total=total,percent=100*done/max(total,1),note=note,state=state,
              updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid())
    save(ROOT/'progress.json',item)
    (ROOT/'进程.txt').write_text(f"同权限主动CV、先导预算与训练扩充实验\n更新时间：{item['updated']}\n"
        f"阶段：{stage}\n进度：{done}/{total}（{item['percent']:.1f}%）\n状态：{state}\nPID：{os.getpid()}\n{note}\n",encoding='utf-8-sig')

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
            theta,meta=base.fit(z[tr],labels,lam);self.iterations+=meta['iterations']
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
    betlo=previous.betting_interval(lower,B,glo,ghi,delta,fractions)[0]
    bethi=previous.betting_interval(upper,B,glo,ghi,delta,fractions)[1]
    return max(detlo,betlo),min(dethi,bethi)

def active_cv(a,b,oracle,splits,lam,g,bounds,cfg,permutation_seed,query_seed):
    tick=time.perf_counter();B=len(splits);order=list(np.random.default_rng(permutation_seed).permutation(B))
    folds={};history=[];events=[];delta_outer=cfg['delta']/2;delta_inner=cfg['delta']/(2*B)
    lo=float(bounds['lower'].mean());hi=float(bounds['upper'].mean());choice=base.decision(lo,hi,g)
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
        choice=base.decision(lo,hi,g)
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

def expand_complete_objects(oracle,priority,budget):
    """Use paid labels first; buy the cheapest missing labels to complete objects."""
    assert oracle.count<=budget
    observed=oracle.seen.reshape(-1,oracle.r).sum(axis=1)
    rank=np.empty(len(priority),int);rank[priority]=np.arange(len(priority))
    order=np.lexsort((rank,oracle.r-observed))
    for obj in order:
        cost=int(oracle.r-observed[obj])
        if not cost:continue
        if oracle.count+cost>budget:continue
        oracle.query_objects(np.array([obj]),'training_growth')
    complete=np.flatnonzero(oracle.seen.reshape(-1,oracle.r).all(axis=1))
    assert oracle.count<=budget and len(complete)*oracle.r<=oracle.count
    return complete

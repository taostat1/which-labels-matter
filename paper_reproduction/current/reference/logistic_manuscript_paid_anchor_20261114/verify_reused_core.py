"""Exhaustive small-label checks of the new anchor expansion and bounds."""
from itertools import product
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'logistic_paid_anchor_20261108'))
from anchor_core import *
ROOT=Path(__file__).resolve().parent
from whichlabels._moments import polynomial as zero_polynomial

rng=np.random.default_rng(202611082101)
max_identity=0.;max_ratio=0.;checks=0;positive_contraction=0
for seed in range(4):
    a=rng.normal(size=(4,2,3));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    b=.85*a+.15*rng.normal(size=a.shape);b/=np.maximum(1,np.linalg.norm(b,axis=-1,keepdims=True))
    splits=[(np.array([0,1]),np.array([2,3])),(np.array([2,3]),np.array([0,1]))]
    for lam in (.05,.5):
        for mode in ('zero','nonzero'):
            y0=rng.integers(0,2,size=(4,2)).astype(float)
            known=np.full(8,np.nan)
            if mode=='nonzero':
                known[:2]=y0[0]
                anchors=[fit(z[:1],y0[:1],lam)[0] for z in (a,b)]
            else:anchors=[np.zeros(3),np.zeros(3)]
            poly=prepare(a,b,splits,anchors,lam,known)
            positive_contraction+=sum(d>0 for v in poly['bounds'] for d in v['denominators'])
            unknown=np.flatnonzero(~np.isfinite(known))
            for signs in product([0.,1.],repeat=len(unknown)):
                yy=known.copy();yy[unknown]=signs;yy=yy.reshape(4,2)
                au=audit(poly,a,b,yy,splits,anchors,lam)
                max_identity=max(max_identity,au['polynomial_identity_error'])
                max_ratio=max(max_ratio,au['actual_error']/max(poly['epsilon'],1e-30))
                checks+=1
            if mode=='zero':
                old=zero_polynomial(a,b,splits,lam)
                assert abs(old['constant']-poly['constant'])<1e-12
                assert poly['single_terms']==0
                for _ in range(5):
                    xx=rng.choice([-1.,1.],size=8)
                    oldval=old['constant']+old['coef']@(xx[old['left']]*xx[old['right']])
                    newval=poly['constant']+poly['coef']@(xx[poly['left']]*xx[poly['right']])
                    assert abs(oldval-newval)<1e-12

toy=dict(constant=0.,left=np.array([0,1,0]),right=np.array([-1,-1,1]),
         coef=np.array([.03,-.02,.05]),mass=.1,cdf=np.array([.3,.5,1.]),epsilon=.0001)
cache=LabelCache(2,1,ArrayLabelProvider(np.array([[0.],[1.]])))
trial=sample(toy,cache,91283,.005)
assert trial['screen']=='sampled' and trial['choice']=='A'
assert trial['interval'][0]<=-.1<=trial['interval'][1] and cache.count==2

result=dict(exhaustive_label_assignments=checks,max_identity_error=max_identity,
            max_actual_error_over_certified_bound=max_ratio,positive_contraction_directions=positive_contraction,
            zero_anchor_polynomial_matches_original=True,
            single_and_pair_cp_sample_path_verified=True)
save(ROOT/'verification.json',result)
print((ROOT/'verification.json').read_text('utf-8'),flush=True)

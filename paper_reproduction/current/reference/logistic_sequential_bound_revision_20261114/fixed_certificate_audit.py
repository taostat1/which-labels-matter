"""Exhaustive completion checks for the certificate used by the main rule."""
from pathlib import Path
import os,sys,itertools,json
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_local_anchor_refinement_20261114'))
from local_core import candidate_geometry,certificate
from anchor_core import fit,risk
import numpy as np

rng=np.random.default_rng(202611141933)
a=rng.normal(size=(3,2,3));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
b=.6*a+.4*rng.normal(size=a.shape);b/=np.maximum(1,np.linalg.norm(b,axis=-1,keepdims=True))
responses=np.array(list(itertools.product([0.,1.],repeat=6)))
signs=2*responses-1
patterns=[[],[0,4],[1,2,4,5],list(range(6))]
rows=[]
for lam in [.125,.5,2.]:
    geometry=[candidate_geometry(z[:2],z[2:],np.zeros(3),lam,np.full(4,np.nan)) for z in (a,b)]
    exact=[];approx=[]
    for y,x in zip(responses,signs):
        yy=y.reshape(3,2);f=[];q=[]
        for z,g in zip((a,b),geometry):
            theta,_=fit(z[:2],yy[:2],lam)
            f.append(risk(z[2:],yy[2:],theta))
            t=g['L']@x[:4];logits=z[2:].reshape(2,3)@t
            q.append(np.mean(np.log(2)-x[4:]*logits/2+logits**2/8))
        exact.append(f[0]-f[1]);approx.append(q[0]-q[1])
    errors=abs(np.array(exact)-np.array(approx));excess=[];checks=0
    for pat in patterns:
        options=list(itertools.product([-1.,1.],repeat=len(pat)))
        for values in options:
            known=np.full(6,np.nan);known[pat]=values
            ga,gb=[candidate_geometry(z[:2],z[2:],np.zeros(3),lam,known[:4]) for z in (a,b)]
            old,new,aa,bb=certificate(ga,gb,lam,known[4:])
            mask=np.ones(64,dtype=bool) if not pat else np.all(signs[:,pat]==known[pat],axis=1)
            checks+=int(mask.sum());worst=float(errors[mask].max())
            excess.append(worst-new['epsilon'])
            assert worst<=new['epsilon']+1e-9,(lam,pat,values,worst,new['epsilon'])
            assert worst<=new['separate']+1e-9
    rows.append(dict(lambda_value=lam,cache_states=sum(2**len(p) for p in patterns),completion_checks=checks,
                     largest_exact_approximation_error=float(errors.max()),largest_bound_excess=max(excess)))
out=dict(passed=True,details=rows,checks=sum(v['completion_checks'] for v in rows),
         description='All 64 binary completions of a 3-object x 2-attribute design, 85 cache states, three penalties; paired and separate bounds audited.')
(ROOT/'fixed_certificate_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

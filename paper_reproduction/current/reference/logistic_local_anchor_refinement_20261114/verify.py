from local_core import *
from anchor_core import fit,risk
import itertools,json,time
rng=np.random.default_rng(202611141301);rows=[];start=time.perf_counter()
for design in range(4):
    a=rng.normal(size=(5,1,2));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    b=a.copy() if design==0 else .7*a+.3*rng.normal(size=a.shape)/3
    for lam in [.125,.01,.001]:
        for typ in ['zero','nonzero']:
            anchors=[np.zeros(2),np.zeros(2)] if typ=='zero' else [rng.normal(size=2),rng.normal(size=2)]
            if design==0:anchors[1]=anchors[0].copy()
            known=np.array([1.,np.nan,np.nan]);kq=np.full(2,np.nan)
            gg=[candidate_geometry(z[:3],z[3:],t,lam,known) for z,t in zip((a,b),anchors)]
            old,new,aa,bb=certificate(*gg,lam,kq)
            maxerr=0
            for bits in itertools.product([-1.,1.],repeat=4):
                x=np.r_[1.,bits];y=((x+1)/2)[:,None];loss=[];approx=[]
                for z,t,g in zip((a,b),anchors,(aa,bb)):
                    hat=fit(z[:3],y[:3],lam)[0];step=g['c']+g['L']@x[:3];theta=t+step
                    e=hat-theta
                    assert np.linalg.norm(e)<=g['e']+2e-8
                    assert np.all(abs(g['z']@e)<=g['train_error']+2e-8)
                    assert np.all(abs(g['q']@e)<=g['test_error']+2e-8)
                    q=g['q'];u=q@step;v=g['vq'];yy=y[3:].ravel()
                    ap=np.mean(np.logaddexp(0,v)-yy*v+(expit(v)-yy)*u+.5*sigmoid_weight(v)*u*u)
                    actual=risk(z[3:],y[3:],hat)
                    assert abs(actual-ap)<=g['separate']+2e-8
                    loss.append(actual);approx.append(ap)
                err=abs((loss[0]-loss[1])-(approx[0]-approx[1]));maxerr=max(maxerr,err)
                assert err<=new['epsilon']+3e-8,(design,lam,typ,err,new)
            rows.append(dict(design=design,lam=lam,anchor=typ,old=old['epsilon'],new=new['epsilon'],max_actual=maxerr,labelings=16))
out=dict(designs=4,cases=len(rows),labelings=sum(x['labelings'] for x in rows),checks='parameter norms, train/test logit errors, individual and paired risk errors',seconds=time.perf_counter()-start,rows=rows)
(ROOT/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

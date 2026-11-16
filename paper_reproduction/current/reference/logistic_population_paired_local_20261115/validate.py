"""Exhaustive response completions and independently fitted exact risks."""
from paired_local import *
from public_cap import cap
import population_experiment as pop
import itertools,time

def main():
    rng=np.random.default_rng(202611152707);rows=[];count=0;worst=0.;tick=time.perf_counter()
    for lam in (.05,.125,.5):
        for anchor in ('zero','nonzero'):
            for overlap in (0.,.1,1.):
                a=rng.normal(size=(4,2,3));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
                other=rng.normal(size=a.shape);other/=np.maximum(1,np.linalg.norm(other,axis=-1,keepdims=True))
                b=(1-overlap)*a+overlap*other
                ts=[np.zeros(3),np.zeros(3)] if anchor=='zero' else [rng.normal(size=3)*.5,rng.normal(size=3)*.5]
                # Two known labels, six missing labels, all completions.
                x=np.full(8,np.nan);x[[0,5]]=[1,-1]
                bd=certificate(a[:2],b[:2],a[2:],b[2:],ts,lam,x[:4],x[4:])
                cc=cap(a[:2],b[:2],a[2:],b[2:],lam)['cap'] if anchor=='zero' else float('inf')
                maxerr=0.
                for signs in itertools.product((-1,1),repeat=6):
                    xx=x.copy();xx[np.isnan(xx)]=signs;y=((xx+1)/2).reshape(4,2);errors=[];es=[]
                    for z,t in zip((a,b),ts):
                        theta=pop.s.fit(z[:2],y[:2],lam)[0]
                        step=pop.s.step(z[:2],y[:2],t,lam);bar=t+step
                        approx=pop.s.quadratic_risk(z[2:],y[2:],t,step,pop.s.test_constants(z[2:],t))
                        errors.append(pop.s.risk(z[2:],y[2:],theta)-approx);es.append(theta-bar)
                    err=abs(errors[0]-errors[1]);de=np.linalg.norm(es[0]-es[1])
                    assert err<=bd['epsilon']+2e-8,(lam,anchor,overlap,err,bd)
                    assert de<=bd['de']+2e-8,(lam,anchor,overlap,de,bd)
                    assert err<=cc+2e-8,(err,cc)
                    maxerr=max(maxerr,err);count+=1
                rows.append(dict(lam=lam,anchor=anchor,difference=overlap,epsilon=bd['epsilon'],max_error=maxerr,cap=None if not np.isfinite(cc) else cc))
    pop.save(ROOT/'validation.json',dict(completions=count,passed=True,records=rows,seconds=time.perf_counter()-tick))
    print('PASS',count,'completions',time.perf_counter()-tick,flush=True)
if __name__=='__main__':main()

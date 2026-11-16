"""Exhaustive small-law checks of the new public moment certificate."""
from run import *
from public_bound import score_envelope
from scipy.special import expit
from itertools import product

def check():
    rng=np.random.default_rng(202611141933)
    a=rng.normal(size=(4,2,3));a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    b=.65*a+.35*rng.normal(size=a.shape);b/=np.maximum(1,np.linalg.norm(b,axis=-1,keepdims=True))
    pools=dict(training=np.arange(2),comparison=np.arange(2,4))
    orders=list(product(range(2),repeat=2));rows=[];score_checks=0
    for lam in [.125,.5,2.]:
        for mode in ['zero','nonzero']:
            ts=[np.zeros(3),np.zeros(3)] if mode=='zero' else [np.array([.4,-.1,.25]),np.array([.3,.15,-.2])]
            case=dict(n=2,lam=lam);par=p.parameters(a,b,pools,ts,lam)
            eps,details=allowance(a,b,pools,ts,case,par,s)
            envelopes=[[score_envelope(z[ids],t,penalty) for ids,penalty in [(pools['training'],lam),(pools['comparison'],0.)]] for z,t in zip((a,b),ts)]
            maxerror=0.;max_moment_excess=-np.inf
            for bits in product([0.,1.],repeat=8):
                y=np.array(bits).reshape(4,2);exact=[];approx=[]
                for j,(z,t) in enumerate(zip((a,b),ts)):
                    for k,(ids,penalty) in enumerate([(pools['training'],lam),(pools['comparison'],0.)]):
                        zz=z[ids];yy=y[ids]
                        chi=np.einsum('ird,ir->id',zz,expit(np.einsum('ird,d->ir',zz,t))-yy)/2+penalty*t
                        env=envelopes[j][k]
                        diffs=[np.sum(chi.mean(0)**2)-env['mu2'],np.sum(chi**2,axis=1).mean()-env['second'],np.sum(chi**2,axis=1).max()-env['B']**2]
                        max_moment_excess=max(max_moment_excess,*diffs);score_checks+=3
                    ee=[];aa=[]
                    for ids in orders:
                        ids=np.array(ids);tt,_=s.fit(z[ids],y[ids],lam)
                        step=s.step(z[ids],y[ids],t,lam)
                        ee.append(s.risk(z[2:],y[2:],tt));aa.append(s.quadratic_risk(z[2:],y[2:],t,step,par['tc'][j]))
                    exact.append(np.mean(ee));approx.append(np.mean(aa))
                error=abs((exact[0]-exact[1])-(approx[0]-approx[1]));maxerror=max(maxerror,error)
            assert max_moment_excess<=1e-10
            assert maxerror<=eps+1e-9
            rows.append(dict(lam=lam,anchor=mode,label_completions=256,training_samples_enumerated=4,
                             public_epsilon=eps,max_exact_expected_risk_error=maxerror,max_moment_excess=max_moment_excess))
    s.save(ROOT/'validation.json',dict(passed=True,score_inequality_checks=score_checks,
          exact_population_comparisons=256*len(rows),details=rows,
          interpretation='Exhaustive numerical implementation checks; analytical justification is in public_bound_note.txt.'))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':check()

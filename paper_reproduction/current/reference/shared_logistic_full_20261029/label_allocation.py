"""Price-free optimal integer allocation for a finite-sample logistic risk bound."""
from pathlib import Path
import json, math, time
import numpy as np
from scipy.optimize import brentq
from scipy.special import comb

ROOT=Path(__file__).resolve().parent


def bernstein(coeff,p):
    p=np.asarray(p);n=len(coeff)-1
    return sum(float(c)*comb(n,k)*p**k*(1-p)**(n-k) for k,c in enumerate(coeff))


def logistic_curve(n,lam,weight):
    def h(t):
        if t==0:return 0.
        t=abs(t)
        th=brentq(lambda z:lam*z+.5*np.tanh(z/2)-t/2,0,t/(2*lam),xtol=1e-14,rtol=1e-14)
        assert abs(lam*th+.5*np.tanh(th/2)-t/2)<1e-13
        return float(np.logaddexp(th/2,-th/2)-np.log(2.))
    values=[h(k/n) for k in range(n+1)]
    bc=np.array([sum(comb(k,i)*values[abs(2*i-k)] for i in range(k+1))/2**k for k in range(n+1)])
    dc=weight*(bc-bc[::-1])
    d1=n*np.diff(dc)/2;d2=n*(n-1)*np.diff(dc,2)/4
    # Bound the propagated root residual and finite-sum rounding explicitly.
    h_error=1e-13/(2*lam)+1e-12
    e1=weight*2*n*h_error;e2=weight*2*n*(n-1)*h_error
    low=float(d1.min()-e1);upper=float(d1.max()+e1);second=float(max(abs(d2),default=0.)+e2)
    assert low>0
    return dict(n=n,lam=lam,weight=weight,coeff=dc.tolist(),a=low,b=upper,q=second,numerical_h_allowance=h_error)


def radius(counts,b,q,delta):
    counts=np.asarray(counts,float)
    return float(np.sqrt(2*np.log(2/delta)*np.sum(b*b/counts))+.5*np.sum(q/counts))


def allocate(b,q,g,delta):
    """Solve the continuous convex program, then round each count upward."""
    b=np.asarray(b,float);q=np.asarray(q,float);t=np.log(2/delta)
    def equation(A):
        z=np.sqrt(A*b*b+q/2);v0=np.sum(b*b/z);q0=np.sum(q/z)
        return t/A+t*q0/(4*A*A*v0)-g
    lo=t/g;hi=lo*2
    while equation(hi)>0:hi*=2
    A=lo if abs(equation(lo))<1e-14*g else brentq(equation,lo,hi,xtol=1e-10,rtol=1e-13)
    z=np.sqrt(A*b*b+q/2);v0=np.sum(b*b/z);scale=2*A*A*v0/t
    x=scale*z;N=np.maximum(1,np.ceil(x)).astype(int)
    rad=radius(x,b,q,delta);V=np.sum(b*b/x)
    gradient=-(t*b*b/np.sqrt(2*t*V)+q/2)/(x*x)
    residual=float(np.max(abs(1+scale*scale*gradient)))
    assert abs(rad-g)<max(1e-10,1e-7*g)
    assert residual<1e-9
    assert radius(N,b,q,delta)<=g+1e-12
    assert N.sum()<x.sum()+len(b)+1e-9
    return dict(continuous=x.tolist(),counts=N.tolist(),pairs=int(N.sum()),labels=int(2*N.sum()),
                radius=radius(N,b,q,delta),continuous_lower_bound=float(2*x.sum()),rounding_gap=float(2*(N.sum()-x.sum())),kkt_residual=residual)


def uniform_allocation(b,q,g,delta):
    t=np.log(2/delta);v=float(np.sum(b*b));h=float(np.sum(q))
    root=(np.sqrt(2*t*v)+np.sqrt(2*t*v+2*g*h))/(2*g)
    each=max(1,int(np.ceil(root*root)));counts=np.full(len(b),each)
    assert radius(counts,b,q,delta)<=g+1e-12
    return dict(counts=counts.tolist(),pairs=int(counts.sum()),labels=int(2*counts.sum()),radius=radius(counts,b,q,delta))


def make_case(n,lambdas,g,eta0=.9):
    K=len(lambdas);curves=[logistic_curve(n,l,1/K) for l in lambdas]
    b=np.array([c['b'] for c in curves]);q=np.array([c['q'] for c in curves])
    gap=lambda e:sum(bernstein(c['coeff'],(1+e)/2) for c in curves)
    boundary=float(gap(eta0));out=dict(n=n,lambdas=lambdas,K=K,g=g,eta0=eta0,curves=curves,boundary=boundary)
    if g>=boundary:return dict(out,no_query=True)
    opt=allocate(b,q,g,.05);uni=uniform_allocation(b,q,g,.05)
    eta=brentq(lambda e:gap(e)-2*g,0,eta0) if 2*g<boundary else eta0
    boundary_eta=brentq(lambda e:gap(e)-g,0,eta0)
    d=boundary_eta*np.log((1+boundary_eta)/(1-boundary_eta));kl=.95*np.log(.95/.05)+.05*np.log(.05/.95)
    lower=2*kl/d
    return dict(out,no_query=False,optimal=opt,uniform=uni,eta_hard=float(eta),hard_gap=float(gap(eta)),
                label_lower_bound=lower,boundary_eta=float(boundary_eta),weight_concentration=float(b.sum()**2/(K*np.sum(b*b))))


def main():
    cases=[('balanced',4,[.5]*4,.005),('heterogeneous',4,[.1,.4,2.,8.],.005),
           ('sparse',4,[.1]+[4.]*7,.002),('larger_tolerance',32,[.5]*4,.005)]
    out=[];rng=np.random.default_rng(202610291923);reps=4000
    for name,n,lams,g in cases:
        c=make_case(n,lams,g);c['name']=name;c['replicates']=reps
        if not c['no_query']:
            c['validation']=[]
            for sign in [-1,0,1]:
                eta=sign*c['eta_hard'];truth=sign*c['hard_gap']
                for method in ['optimal','uniform']:
                    counts=np.array(c[method]['counts']);samples=2*rng.binomial(counts,(1+eta)/2,size=(reps,len(counts)))/counts-1
                    estimates=sum(bernstein(v['coeff'],(1+samples[:,e])/2) for e,v in enumerate(c['curves']))
                    rad=c[method]['radius'];wrong=((estimates>=0)&(truth<-g))|((estimates<0)&(truth>g))
                    coverage=np.abs(estimates-truth)<=rad
                    # A fresh final training sample is actually generated and fitted.
                    # It has n objects with all 2K labels; none was used in comparison.
                    start=time.perf_counter();K=len(counts)
                    u=rng.choice([-1.,1.],size=(reps,n,K));v=u*(2*rng.binomial(1,(1+eta)/2,size=(reps,n,K))-1)
                    chosen_sign=np.where(estimates>=0,-1.,1.)
                    train_mean=((u+chosen_sign[:,None,None]*v)/2).mean(axis=1)
                    penalty=np.asarray(lams)[None,:];theta=train_mean/(2*(penalty+.25))
                    for _ in range(30):
                        grad=penalty*theta+.5*np.tanh(theta/2)-train_mean/2
                        if np.max(abs(grad))<1e-12:break
                        theta-=grad/(penalty+.25/np.cosh(theta/2)**2)
                    assert np.max(abs(grad))<1e-12
                    final_seconds=time.perf_counter()-start
                    c['validation'].append(dict(sign=sign,method=method,truth=truth,errors=int(wrong.sum()),covered=int(coverage.sum()),
                        mean_estimate=float(estimates.mean()),se=float(estimates.std(ddof=1)/np.sqrt(reps)),labels=c[method]['labels'],
                        complete_labels=c[method]['labels']+2*K*n,final_labels=2*K*n,
                        final_fits=reps,final_seconds=final_seconds,final_gradient_max=float(np.max(abs(grad)))))
        out.append(c)
        print(name,'boundary',c['boundary'],'optimal',c.get('optimal',{}).get('labels',0),'uniform',c.get('uniform',{}).get('labels',0),flush=True)
    rates=[]
    for tol in [.005,.0025,.00125,.000625]:
        v=make_case(4,[.1,.4,2.,8.],tol)
        rates.append(dict(g=tol,labels=v['optimal']['labels'],uniform=v['uniform']['labels'],
                          normalized=v['optimal']['labels']*tol**2,lower=v['label_lower_bound']))
    result=dict(cases=out,rates=rates,delta=.05,seed=202610291923,description='Exact independent binomial product averages; label counts only.')
    (ROOT/'label_allocation_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()

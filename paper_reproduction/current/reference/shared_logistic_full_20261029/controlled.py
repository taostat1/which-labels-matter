"""Exact logistic association families and independent fitted-risk validation."""
from core import *
from functools import lru_cache

def theta(t,lam):
    t=np.asarray(t,float);lo=np.full(t.shape,-1/(2*lam));hi=-lo
    for _ in range(55):
        mid=(lo+hi)/2;v=lam*mid+.5*np.tanh(mid/2)
        lo=np.where(v<t/2,mid,lo);hi=np.where(v>=t/2,mid,hi)
    return (lo+hi)/2

def phi(x):return np.logaddexp(x/2,-x/2)

@lru_cache(maxsize=20000)
def exact(n,lam,eta,mu=0):
    lawA=np.array([(1-2*mu+eta)/4,(1-eta)/2,(1+2*mu+eta)/4])
    lawB=np.array([(1-eta)/4,(1+eta)/2,(1-eta)/4])
    assert min(lawA.min(),lawB.min())>=-1e-14
    ps=[]
    for law in (lawA,lawB):
        p=np.ones(1)
        for _ in range(n):p=np.convolve(p,law)
        ps.append(p)
    grid=np.arange(-n,n+1)/n;t=theta(grid,lam)
    return float(ps[0]@(phi(t)-mu*t/2)-ps[1]@phi(t))

def main():
    rng=np.random.default_rng(202610292340);rows=[]
    for n in (4,8,16,64):
        for mu in (0.,.02):
            for eta in (-.8,-.4,0.,.4,.8):
                lam=.5;truth=exact(n,lam,eta,mu);reps=4096
                probs=np.array([(1-2*mu+eta)/4,(1-eta)/4,(1-eta)/4,(1+2*mu+eta)/4])
                labels=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])[rng.choice(4,(reps,n),p=probs)]
                xa=labels.mean(2).mean(1);xb=(labels[:,:,0]-labels[:,:,1]).mean(1)/2
                ta=theta(xa,lam);tb=theta(xb,lam);vals=phi(ta)-mu*ta/2-phi(tb)
                resid=max(np.max(abs(lam*ta+.5*np.tanh(ta/2)-xa/2)),np.max(abs(lam*tb+.5*np.tanh(tb/2)-xb/2)))
                rows.append(dict(n=n,lam=lam,mu=mu,eta=eta,exact=truth,mean=float(vals.mean()),se=float(vals.std(ddof=1)/np.sqrt(reps)),reps=reps,residual=float(resid),g=.005))
    # The derivative formula is checked against finite differences and strict convexity.
    checks=[]
    for lam in (.125,.5,2.):
        grid=np.linspace(-1,1,2001);tt=theta(grid,lam);p=expit(tt);d2=p*(1-p);d1=p-.5;d3=d2*(1-2*p)
        hh=(d2*(lam+d2)-d1*d3)/(4*(lam+d2)**3)
        assert hh.min()>0
        for n in (2,8,32):
            for eta in (-.6,0,.6):
                der=(exact(n,lam,eta+1e-5)-exact(n,lam,eta-1e-5))/2e-5
                assert hh.min()/(2*n)-1e-8<=der<=hh.max()/(2*n)+1e-8
                checks.append(dict(lam=lam,n=n,eta=eta,derivative=der))
    noquery=[dict(n=n,maximum_gap=exact(n,.5,.8),g=.005,no_query=exact(n,.5,.8)<=.005) for n in (4,8,16,64)]
    # Fixed-budget query validation: exact monotone risk intervals from Hoeffding bounds for eta.
    query=[];delta=.05;K=7
    for n in (4,8,16,64):
        for eta in (-.8,.8):
            total=2000;decision_counts=[];covers=[];errors=[];unresolved=0
            for _ in range(total):
                prod=2*(rng.random(1024)<(1+eta)/2)-1;chosen='unresolved';covered=True;stop=0
                if exact(n,.5,.8)<=.005:chosen='A'
                else:
                    for q in (16,32,64,128,256,512,1024):
                        ave=prod[:q].mean();rad=np.sqrt(2*np.log(2*K/delta)/q)
                        lo=exact(n,.5,float(np.clip(ave-rad,-.8,.8)));hi=exact(n,.5,float(np.clip(ave+rad,-.8,.8)))
                        covered=covered and lo-1e-10<=exact(n,.5,eta)<=hi+1e-10
                        if hi<=.005:chosen='A';stop=q;break
                        if lo>=-.005:chosen='B';stop=q;break
                truth=exact(n,.5,eta)
                unresolved+=chosen=='unresolved'
                decision_counts.append(stop);covers.append(covered);errors.append((chosen=='A' and truth>.005) or (chosen=='B' and truth<-.005))
            query.append(dict(n=n,eta=eta,reps=total,mean_pairs=float(np.mean(decision_counts)),sd_pairs=float(np.std(decision_counts,ddof=1)),median_pairs=float(np.median(decision_counts)),coverage=sum(covers),errors=sum(errors),unresolved=unresolved))
    save(ROOT/'controlled_results.json',dict(risk=rows,derivative_checks=checks,no_query=noquery,queries=query))
    print('Controlled validation:',len(rows),'risk cells;',len(checks),'curvature checks;',len(query),'query cells')

if __name__=='__main__':main()

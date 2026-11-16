"""Independent numerical checks of the risk derivatives and convex allocation."""
from label_allocation import *
from scipy.optimize import minimize
import mpmath as mp

mp.mp.dps=70

def precise_coefficients(n,lam,K):
    lm=mp.mpf(str(lam));bc=[]
    for k in range(n+1):
        hvals=[]
        for i in range(k+1):
            t=mp.mpf(abs(2*i-k))/n
            if t==0:h=mp.mpf(0)
            else:
                th=mp.findroot(lambda x:lm*x+mp.tanh(x/2)/2-t/2,t/(2*(lm+mp.mpf('.25'))))
                h=mp.log(mp.cosh(th/2))
            hvals.append(mp.binomial(k,i)*h/mp.power(2,k))
        bc.append(mp.fsum(hvals))
    return [(bc[k]-bc[n-k])/K for k in range(n+1)]


def main():
    data=json.loads((ROOT/'label_allocation_results.json').read_text('utf-8'))
    checked=0;maxrel=0.;optimizer_checks=[]
    for case in data['cases']:
        for curve in case['curves']:
            n=curve['n'];dc=precise_coefficients(n,curve['lam'],case['K'])
            d1=[mp.mpf(n)*(dc[k+1]-dc[k])/2 for k in range(n)]
            d2=[mp.mpf(n*(n-1))*(dc[k+2]-2*dc[k+1]+dc[k])/4 for k in range(n-1)]
            assert mp.mpf(curve['a'])<min(d1)
            assert mp.mpf(curve['b'])>max(d1)
            assert mp.mpf(curve['q'])>max([abs(x) for x in d2],default=0)
            maxrel=max(maxrel,max(float(abs(mp.mpf(x)-v)) for x,v in zip(curve['coeff'],dc)))
            checked+=1
        if case['no_query']:continue
        b=np.array([c['b'] for c in case['curves']]);q=np.array([c['q'] for c in case['curves']]);g=case['g']
        target=np.array(case['optimal']['continuous']);scale=target.sum()
        result=minimize(lambda u:u.sum(),np.full(len(b),2*max(target)/scale),
              method='SLSQP',bounds=[(1e-9,None)]*len(b),
              constraints=[dict(type='ineq',fun=lambda u:1-radius(u*scale,b,q,.05)/g)],
              options=dict(ftol=1e-12,maxiter=2000))
        assert result.success,result.message
        assert abs(result.fun-1)<1e-6
        optimizer_checks.append(dict(name=case['name'],independent_objective_ratio=float(result.fun),
                                     integer_gap=case['optimal']['rounding_gap']))
        # The numerical lower bound must not exceed a uniformly valid upper count.
        assert case['label_lower_bound']<=case['optimal']['labels']
    out=dict(high_precision_curves=checked,max_absolute_coefficient_discrepancy=maxrel,
             independent_convex_solver=optimizer_checks,passed=True)
    (ROOT/'label_theory_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()

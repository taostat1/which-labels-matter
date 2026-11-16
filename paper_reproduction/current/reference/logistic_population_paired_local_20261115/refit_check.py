from paired_local import *
import population_experiment as pop
import json

def main():
    rows=[]
    for ci in (3,6):
        case=pop.s.CASES[ci];a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        ts=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])];par=pop.p.parameters(a,b,pools,ts,case['lam'])
        seed=202611141701+ci*10000000+6*10000
        old=json.loads((pop.ROOT/'population_results'/f'case_{ci:02d}_06.json').read_text('utf-8'))
        ref=next(v for v in old['records'] if v['method']=='zero_anchor')
        # Large tolerance stops after one checkpoint, leaving all numerical
        # statistics and the eight-inspection error allocation unchanged.
        _,_,_,history=pop.p.fallback(a,b,pools,dict(case,g=10.),ts,par['R'],pop.p.Cache(y),seed)
        expected=ref['refit_history'][0]
        error=float(np.max(abs(np.array(history[0]['interval'])-expected['interval'])))
        assert error<1e-11 and history[0]['blocks']==expected['blocks']
        rows.append(dict(case=case['id'],blocks=history[0]['blocks'],interval_error=error,passed=True))
        print(rows[-1],flush=True)
    pop.save(ROOT/'refit_parity.json',dict(records=rows,role='Numerical audit only, excluded from algorithm purchases.'))
if __name__=='__main__':main()

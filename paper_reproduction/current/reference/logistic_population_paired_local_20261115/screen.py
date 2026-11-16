from pathlib import Path
import json,time,argparse
from paired_local import *
import population_experiment as pop

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plans',type=int,default=32)
    args=ap.parse_args();out=[]
    for ci in (3,6):
        case=pop.s.CASES[ci];tick=time.perf_counter()
        old=json.loads((pop.ROOT/'population_results'/f'case_{ci:02d}_06.json').read_text('utf-8'))
        a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        ids=np.asarray(old['initialization']['pilot_ids']);cache=pop.p.Cache(y)
        yp=cache.objects(ids,'pilot');r=y.shape[1]
        ts=[pop.s.fit(z[ids],yp,case['lam'])[0] for z in (a,b)]
        rng=np.random.default_rng(202611151717+ci)
        plans=[(rng.choice(pools['training'],case['n']),rng.choice(pools['comparison'],128)) for _ in range(args.plans)]
        x=2*cache.values-1
        for name,anchors in [('zero',[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]),('pilot32',ts)]:
            rec=[]
            for tr,te in plans:
                tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
                rec.append(certificate(a[tr],b[tr],a[te],b[te],anchors,case['lam'],x[tk],x[qk]))
            row=dict(case=case,anchor=name,plans=args.plans,old_mean=float(np.mean([v['old'] for v in rec])),
                     new_mean=float(np.mean([v['epsilon'] for v in rec])),records=rec,seconds=time.perf_counter()-tick)
            out.append(row);pop.save(ROOT/'screen.json',out)
            print(case['id'],name,{k:v for k,v in row.items() if k not in ('records','case')},flush=True)
if __name__=='__main__':main()

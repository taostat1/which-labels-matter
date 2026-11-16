"""Matched query traces. Final decisions use the certified public integral.

Refit numerical statistics are reused from the matched archived path: models,
seeds, samples, fitting tolerance and interval are unchanged. Label unions are
recomputed from the current physical cache, never copied from the archive.
"""
from paired_local import *
import population_experiment as pop
import json,time,argparse

def run(ci,group):
    case=pop.s.CASES[ci];tick=time.perf_counter()
    a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
    anchors=[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]
    par=pop.p.parameters(a,b,pools,anchors,case['lam'])
    template=pop.BoundTemplate(a,b,pools,anchors,case);states=template.state()
    cache=pop.p.Cache(y);seed=202611141701+ci*10000000+group*10000
    rawL,rawU=-par['M']-1e-9,par['M']+1e-9;history=[];previous=0
    for phase,end in enumerate(pop.p.CHECKS):
        lo,hi,detail=pop.p.moment_phase(a,b,pools,case,anchors,par,cache,phase,end-previous,seed)
        rawL=max(rawL,lo);rawU=min(rawU,hi);previous=end
        cached_eps=template.update(states,cache.values,par,pop.s,pop.from_moments)[0]
        history.append(dict(blocks=end,labels=cache.count,raw_interval=[rawL,rawU],cached_epsilon=cached_eps,**detail))
        print(ci,group,end,cache.count,[rawL,rawU],flush=True)
    old=json.loads((pop.ROOT/'population_results'/f'case_{ci:02d}_{group:02d}.json').read_text('utf-8'))
    ref=next(v for v in old['records'] if v['method']=='zero_anchor')
    assert ref['fallback'] and ref['seed']==seed
    assert abs(par['R']-ref['refit_range'])<1e-12
    rng=np.random.default_rng(seed+6000000);num=0;refit_history=[]
    for h in ref['refit_history']:
        while num<h['blocks']:
            bs=min(64,h['blocks']-num);ids=rng.choice(pools['training'],(bs,case['n']));test=rng.choice(pools['comparison'],bs)
            cache.objects(ids,'refit');cache.objects(test,'refit');num+=bs
        refit_history.append(dict(h,labels=cache.count))
    final=np.random.default_rng(pop.s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    yf=cache.objects(final,'final_training')
    _,fit_info=pop.s.fit((a if ref['choice']=='A' else b)[final],yf,case['lam'])
    row=dict(case=case,group=group,seed=seed,history=history,baseline_labels=ref['labels'],
             fallback_labels=cache.count,fallback_choice=ref['choice'],refit_history=refit_history,
             fallback_interval=ref['refit_history'][-1]['interval'],stage_labels=cache.stages,seconds=time.perf_counter()-tick,
             final_fit=fit_info,
             refit_statistics='Matched archived numerical values; physical label union recomputed.',
             status='trace; integrate public-error certificate before interpreting a stopping decision')
    pop.save(ROOT/'query_traces'/f'case_{ci}_{group}.json',row)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True);ap.add_argument('--group',type=int,required=True)
    args=ap.parse_args();run(args.case,args.group)

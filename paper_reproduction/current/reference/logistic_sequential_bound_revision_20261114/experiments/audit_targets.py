"""Independently refit one unchanged pool in every reported configuration."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import validate_full as run


def check(job):
    a,b,y,plan,_,_,references,_=run.data(job)
    _,value=run.e.exact_fits(a,b,y,plan,job['lam'])
    record=dict(target=references[0].get('target',references[0].get('cv_delta')))
    difference=abs(value-record['target'])
    assert difference<=2e-9,(job,difference)
    return dict(job=job,refitted_target=value,frozen_target=record['target'],
                difference=difference,passed=True)


if __name__=='__main__':
    jobs=[j for j in run.configs() if j['pool']==0]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(check,jobs))
    run.save(run.ROOT/'target_refit_audit.json',dict(configurations=len(rows),checks=rows,
        max_difference=max(r['difference'] for r in rows),passed=True))
    print('Refitted targets agree in',len(rows),'configurations.')

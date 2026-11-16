from local_core import *
from anchor_core import prepare,sample,LabelCache,ArrayLabelProvider,fit
from run import load_domain,SOURCE,summarize_record,exact_fits
from whichlabels import prepare_comparison,compare
from whichlabels._intervals import endpoints,mapping_guard
import json,gc,time,csv
checks=[128,256,512,1024,2048,4096,8192,16384,32768]
protocol=dict(domain='image',penalty=.125,tolerance=.005,delta=.05,pools=[0,1,2],paths_per_pool=1,
              plans='All 32 original full-validation splits; original final subset 0.',pilot_objects=32,
              methods=['zero_static','pilot_old_static','pilot_local_static','pilot_local_cache'],
              seed=202611141601,scope='Exploratory full-chain follow-up selected after bound diagnosis; not independent validation.',
              fixed='Anchor, polynomial, proposal and CP inspections fixed. Only deterministic error allowance updated.',
              accounting='Distinct pilot/query/fallback/final-training labels; all methods fit the selected candidate.')
(ROOT/'chain_protocol.json').write_text(json.dumps(protocol,indent=2))
rows=[];aa,bb,yy=load_domain('image');began=time.perf_counter()
for pool in protocol['pools']:
    sch=np.load(SOURCE/'coverage/results'/f'schedule_image_{pool:02d}.npz');ids=sch['pool_ids']
    a,b,y=aa[ids],bb[ids],yy[ids];r=y.shape[1];plan=list(zip(sch['train'],sch['test']));final=sch['final'][0]
    seed=protocol['seed']+pool;lam=.125;g=.005
    cache=LabelCache(len(a),r,ArrayLabelProvider(y));zero=prepare_comparison(a,b,plan,lam)
    row=compare(zero,cache,final,tolerance=g,delta=.05,inspections=checks,seed=seed)
    row.update(pool=pool,method='zero_static',initial_epsilon=zero.epsilon);rows.append(row);del zero;gc.collect()
    pilot=np.random.default_rng(202611141401+pool+100).permutation(len(a))[:32]
    paid=LabelCache(len(a),r,ArrayLabelProvider(y));yp=paid.query_objects(pilot,'pilot')
    anchors=[fit(z[pilot],yp,lam)[0] for z in (a,b)]
    poly=prepare(a,b,plan,anchors,lam,paid._values.copy());old_eps=poly['epsilon']
    def refined(values):
        xx=np.where(np.isfinite(values),2*values-1,np.nan);eps=[]
        for tr,te in plan:
            tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
            gg=[candidate_geometry(z[tr],z[te],t,lam,xx[tk]) for z,t in zip((a,b),anchors)]
            eps.append(certificate(*gg,lam,xx[qk])[1]['epsilon'])
        return float(np.mean(eps))
    local_eps=min(old_eps,refined(paid._values));print('pool',pool,'old/local epsilon',old_eps,local_eps,flush=True)
    for method,eps in [('pilot_old_static',old_eps),('pilot_local_static',local_eps),('pilot_local_cache',local_eps)]:
        cache=LabelCache(len(a),r,ArrayLabelProvider(y));cache.query_objects(pilot,'pilot')
        poly['epsilon']=eps;start=time.perf_counter()
        if method!='pilot_local_cache':trial=sample(poly,cache,seed,g)
        else:
            rng=np.random.default_rng(seed);n=k=0;history=[];c=poly['constant'];mass=poly['mass']
            assert mapping_guard(poly['coef'],poly['cdf'],mass)<2e-9
            choice='unresolved'
            for end in checks:
                ix=np.searchsorted(poly['cdf'],rng.random(end-n),side='right')
                xl=2*cache.query_keys(poly['left'][ix],'comparison')-1
                kr=poly['right'][ix];mask=kr>=0;xr=np.ones(len(ix));xr[mask]=2*cache.query_keys(kr[mask],'comparison')-1
                k+=int(np.sum(np.sign(poly['coef'][ix])*xl*xr>0));n=end
                cp= endpoints(k,n,.05/len(checks))
                eps=min(eps,refined(cache._values))
                lo=c+mass*(2*cp[0]-1)-eps-2e-9;hi=c+mass*(2*cp[1]-1)+eps+2e-9
                choice='A' if hi<=g else 'B' if lo>=-g else 'unresolved'
                history.append(dict(draws=n,labels=cache.count,epsilon=eps,interval=[lo,hi],choice=choice))
                print('pool',pool,'dynamic',n,'labels',cache.count,'eps',round(eps,6),choice,flush=True)
                if choice!='unresolved':break
            trial=dict(choice=choice,interval=[lo,hi],draws=n,history=history)
        row=summarize_record(trial,cache,a,b,plan,lam,final,g,
            dict(pool=pool,method=method,initial_epsilon=old_eps if method=='pilot_old_static' else local_eps,final_epsilon=eps),0.)
        row['comparison_seconds']=time.perf_counter()-start
        rows.append(row)
        print('pool',pool,method,'labels',row['total_labels'],'fallback',row['fallback'],flush=True)
        (ROOT/'chain_before_audit.json').write_text(json.dumps(rows,indent=2))
    # Full-label audits run only after all four acquisition paths are finished.
    parameters,target=exact_fits(a,b,y,plan,lam)
    for row in rows:
        if row['pool']!=pool:continue
        row.update(target=target,acceptable=bool(target<=g if row['choice']=='A' else target>=-g),
                   covered=bool(row['interval'][0]<=target<=row['interval'][1]))
    (ROOT/'chain_records.json').write_text(json.dumps(rows,indent=2))
    del poly,paid,cache;gc.collect()
print('DONE',len(rows),round(time.perf_counter()-began,1),flush=True)

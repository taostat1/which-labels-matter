"""Exploratory ablation: identical bound updates without a paid anchor."""
from local_core import *
from anchor_core import LabelCache,ArrayLabelProvider
from run import load_domain,SOURCE,summarize_record
from whichlabels import prepare_comparison
from whichlabels._intervals import endpoints,mapping_guard
import json,gc,time

checks=[128,256,512,1024,2048,4096,8192,16384,32768]
protocol=dict(domain='image',penalty=.125,tolerance=.005,delta=.05,pools=[0,1,2],paths_per_pool=1,
    pilot_objects=0,seed=202611141601,method='zero_local_cache',
    plans='All 32 original full-validation splits; original final subset 0.',
    scope='Post-screen exploratory ablation; same pools and query seeds as the paid-anchor chain.',
    fixed='Zero anchor, original polynomial, original proposal and inspections; only deterministic bound changes.')
(ROOT/'zero_dynamic_protocol.json').write_text(json.dumps(protocol,indent=2))
audits=json.loads((ROOT/'chain_records.json').read_text());rows=[]
aa,bb,yy=load_domain('image');began=time.perf_counter()
for pool in protocol['pools']:
    sch=np.load(SOURCE/'coverage/results'/f'schedule_image_{pool:02d}.npz');ids=sch['pool_ids']
    a,b,y=aa[ids],bb[ids],yy[ids];r=y.shape[1];plan=list(zip(sch['train'],sch['test']));final=sch['final'][0]
    lam=.125;g=.005;cache=LabelCache(len(a),r,ArrayLabelProvider(y))
    start=time.perf_counter();prepared=prepare_comparison(a,b,plan,lam)
    poly=prepared.polynomial;cdf=prepared.proposal['cdf'];eps=prepared.epsilon
    initial=eps;c=poly['constant'];mass=float(abs(poly['coef']).sum());anchor=np.zeros(a.shape[-1])
    assert mapping_guard(poly['coef'],cdf,mass)<2e-9
    def refined(values):
        xx=np.where(np.isfinite(values),2*values-1,np.nan);ee=[]
        for tr,te in plan:
            tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
            gg=[candidate_geometry(z[tr],z[te],anchor,lam,xx[tk]) for z in (a,b)]
            ee.append(certificate(*gg,lam,xx[qk])[1]['epsilon'])
        return float(np.mean(ee))
    eps=min(eps,refined(cache._values))
    # Match the public entry check, with the improved deterministic allowance.
    lo=c-mass-eps-2e-9;hi=c+mass+eps+2e-9
    choice='A' if hi<=g else 'B' if lo>=-g else 'unresolved'
    n=k=0;history=[];rng=np.random.default_rng(protocol['seed']+pool)
    if choice=='unresolved':
        for end in checks:
            ix=np.searchsorted(cdf,rng.random(end-n),side='right')
            xl=2*cache.query_keys(poly['left'][ix],'comparison')-1
            kr=poly['right'][ix];mask=kr>=0;xr=np.ones(len(ix))
            xr[mask]=2*cache.query_keys(kr[mask],'comparison')-1
            k+=int(np.sum(np.sign(poly['coef'][ix])*xl*xr>0));n=end
            cp=endpoints(k,n,.05/len(checks));eps=min(eps,refined(cache._values))
            lo=c+mass*(2*cp[0]-1)-eps-2e-9;hi=c+mass*(2*cp[1]-1)+eps+2e-9
            choice='A' if hi<=g else 'B' if lo>=-g else 'unresolved'
            history.append(dict(draws=n,labels=cache.count,epsilon=eps,interval=[lo,hi],choice=choice))
            print('pool',pool,'zero dynamic',n,'labels',cache.count,'eps',round(eps,6),choice,flush=True)
            if choice!='unresolved':break
    trial=dict(choice=choice,interval=[lo,hi],draws=n,history=history)
    row=summarize_record(trial,cache,a,b,plan,lam,final,g,
        dict(pool=pool,method='zero_local_cache',initial_epsilon=initial,final_epsilon=eps),0.)
    row['comparison_seconds']=time.perf_counter()-start
    target=next(v['target'] for v in audits if v['pool']==pool)
    row.update(target=target,acceptable=bool(target<=g if row['choice']=='A' else target>=-g),
        covered=bool(row['interval'][0]<=target<=row['interval'][1]))
    rows.append(row);(ROOT/'zero_dynamic_records.json').write_text(json.dumps(rows,indent=2))
    print('DONE POOL',pool,row['total_labels'],row['fallback'],flush=True)
    del prepared,poly,cache;gc.collect()
print('DONE',len(rows),round(time.perf_counter()-began,1),flush=True)

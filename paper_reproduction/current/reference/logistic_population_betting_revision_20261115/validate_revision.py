"""Independent inversion and bookkeeping audit of the expanded population run."""
from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'logistic_population_comparators_20261114'
G=np.array([1/64,1/32,1/16,1/8,1/4,1/2,.75,.9,.99])
def read(p): return json.loads(p.read_text('utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def log_cap(z,m,M,side):
    if side==1:
        if m==-M: return 0. if np.all(z==-M) else np.inf
        factors=1-G[None,:]+G[None,:]*(M+z[:,None])/(M+m)
    else:
        if m==M: return 0. if np.all(z==M) else np.inf
        factors=1-G[None,:]+G[None,:]*(M-z[:,None])/(M-m)
    assert np.all(factors>0)
    return float(logsumexp(np.log(factors).sum(axis=0))-math.log(len(G)))
def interval(z,M):
    logcrit=math.log(100)
    f=lambda m:log_cap(z,m,M,1)-logcrit
    g=lambda m:log_cap(z,m,M,-1)-logcrit
    lo=-M if f(-M)<=0 else brentq(f,-M,M,xtol=1e-14,rtol=1e-13)
    hi=M if g(M)<=0 else brentq(g,-M,M,xtol=1e-14,rtol=1e-13)
    return np.array([lo,hi])
def choose(v,g):
    return 'A' if v[1]<=g else ('B' if v[0]>=-g else 'unresolved')

prefixes=0;maxdiff=0.;maxidentity=0.;unchanged=0;queries=0
for path in sorted((ROOT/'results').glob('case_*.json')):
    d=read(path);z=np.load(ROOT/'draws'/(path.stem+'.npz'))['z'];M=d['range']+1e-8
    assert np.max(np.abs(z),initial=0)<=M
    run=np.array([-M,M]);guard=1e-9*(1+M);previous=0;eps=d['epsilon0'];labels=0
    by_n={}
    for h in d['phase_records']:
        n=h['blocks'];v=interval(z[:n],M)+np.array([-guard,guard])
        run=np.array([max(run[0],v[0]),min(run[1],v[1])])
        diff=np.max(np.abs(run-h['raw_interval']));maxdiff=max(maxdiff,float(diff));assert diff<2e-10
        assert h['epsilon']<=eps+1e-13 and h['labels']>=labels
        assert h['labels']-labels<=2*(n-previous)
        maxidentity=max(maxidentity,h['identity_error']);assert h['identity_error']<1e-12
        assert h['max_observed_envelope']<=M
        by_n[n]=run.copy();prefixes+=1;previous=n;eps=h['epsilon'];labels=h['labels']
    queries+=len(z)
    for r in d['records']:
        assert sum(r['stage_labels'].values())==r['labels']
        assert r['pilot_labels']==r['calibration_labels']==0 and r['anchors']=='fixed_zero'
        assert r['error_probability']<=.05
        for h in r['history']:
            iv=by_n[h['blocks']]+np.array([-h['epsilon'],h['epsilon']])
            assert np.max(abs(iv-h['interval']))<2e-10
            assert choose(h['interval'],d['case']['g'])==h['choice']
        assert not r['fallback'] and r['decision_issued']
        assert r['history'][-1]['choice']==r['choice']
        assert all(h['choice']=='unresolved' for h in r['history'][:-1])
        gap=r['audit_gap'];lo,hi=r['interval'];g=d['case']['g']
        assert r['acceptable_audit_mean']==(gap<=g if r['choice']=='A' else gap>=-g)
        assert r['covered_audit_mean']==(lo<=gap<=hi)
    assert d['records'][0]['labels']<=d['records'][1]['labels']
    if d['prior_interval_experiment_parity']: unchanged+=1

certs={}
for ci in [3,6]:
    p=ROOT.parent/'logistic_population_paired_local_20261115'/f'integration_{ci}.json'
    d=read(p);rows={int(r[0]):r[1] for r in d['records']};out=[]
    for c in d['certificates']:
        K=c['n'];v=np.minimum([rows[i] for i in range(K)],c['cap'])
        rad=math.sqrt(2*np.var(v,ddof=1)*math.log(2/c['alpha'])/K)+7*c['cap']*math.log(2/c['alpha'])/(3*(K-1))
        assert abs(rad-c['public_mc_error'])<1e-13
        assert abs(np.mean(v)+rad-c['epsilon'])<1e-13
        out.append(c)
    certs[str(ci)]=dict(file_sha256=sha(p),retained=min(out,key=lambda x:x['epsilon']))

original=read(OLD/'records.json');new=read(ROOT/'records.json')
frozen=[r for r in original if r['method']!='lec'];reused=[r for r in new if r['method']!='lec']
assert frozen==reused and len(frozen)==288
baseline_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
fixed_files=['algorithm.tex','method.tex','general_bounds_proofs.tex','simulations.tex','simulation_details.tex']
fixed={}
for name in fixed_files:
    x,y=OLD/'paper'/name,ROOT/'paper'/name
    assert x.read_bytes()==y.read_bytes(),name
    fixed[name]=sha(y)

summary=read(ROOT/'summary.json');assert summary['method_paths']==384
for ci,code in enumerate(['T1','T2','T3','T4','V1','V2','V3','V4']):
    s=summary['cases'][code]
    for method in ['direct_refits','active_testing','ams','lec']:
        rr=[r for r in new if r['method']==method and r['case_index']==ci]
        assert abs(np.mean([r['labels'] for r in rr])-s[method]['mean'])<1e-10
    saving=100*(1-s['lec']['mean']/min(s[m]['mean'] for m in ['direct_refits','active_testing','ams']))
    assert abs(saving-s['saving_vs_best_baseline'])<1e-12

out=dict(passed=True,paired_paths=96,method_paths=192,query_draws=queries,independent_interval_prefixes=prefixes,
         maximum_endpoint_difference=maxdiff,maximum_conditional_identity_error=maxidentity,
         prior_24_path_parity=unchanged,external_baseline_records_unchanged=288,external_records_sha256=baseline_hash,
         zero_pilot_and_zero_paid_calibration=True,certificates=certs,algorithm1_files_unchanged=fixed,
         audit_interpretation='Acceptability and coverage refer to the frozen independent Monte Carlo audit mean, not an exactly known population risk.')
(ROOT/'validation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
(ROOT/'paper/evidence/population_betting_validation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('certificates','algorithm1_files_unchanged')},indent=2))

from pathlib import Path
import json,sys,hashlib
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent
METHODS=['paired_weighted','paired_uniform','separate_weighted','active_cv']
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 cfg=json.loads((ROOT/'protocol.json').read_text('utf-8'));n=json.loads((ROOT/'formal_registration.json').read_text('utf-8'))['pools']
 groups=[];records=[]
 for i in range(n):
  x=json.loads((ROOT/'results'/f'pool_{i:03}.json').read_text('utf-8'))
  assert len(x['groups'])==12 and len(x['records'])==48
  groups.extend(x['groups']);records.extend(x['records'])
 assert len({(r['group'],r['name'],r['method']) for r in records})==n*48
 assert all(abs(x['actual_error'])<=x['epsilon']+2e-9 for x in groups)
 assert max(x['identity_error'] for x in groups)<2e-10
 summaries=[];rng=np.random.default_rng(202611019991)
 for alpha in cfg['alphas']:
  for lam in cfg['penalties']:
   for signal in cfg['signals']:
    name=f'a{alpha}_l{lam}_s{signal}'
    gg=sorted([x for x in groups if x['name']==name],key=lambda x:x['group'])
    audit=dict(name=name,alpha=alpha,lam=lam,signal=signal,regimes={k:sum(x['regime']==k for x in gg) for k in ['both','A_only','B_only']},
       target_mean=float(np.mean([x['target'] for x in gg])),target_min=min(x['target'] for x in gg),target_max=max(x['target'] for x in gg),
       mean_abs_error=float(np.mean([abs(x['actual_error']) for x in gg])),max_abs_error=max(abs(x['actual_error']) for x in gg),
       epsilon_mean=float(np.mean([x['epsilon'] for x in gg])),epsilon_min=min(x['epsilon'] for x in gg),epsilon_max=max(x['epsilon'] for x in gg),
       separate_mean=float(np.mean([x['separate_epsilon'] for x in gg])),methods={})
    counts={}
    for method in METHODS:
     rr=sorted([r for r in records if r['name']==name and r['method']==method],key=lambda r:r['group']);assert len(rr)==n
     x=np.array([r['labels_total'] for r in rr]);counts[method]=x
     cover=sum(r['covered'] for r in rr);errors=sum(not r['acceptable'] for r in rr)
     audit['methods'][method]=dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),se=float(x.std(ddof=1)/np.sqrt(n)),
       coverage=cover,errors=errors,coverage_lower95=float(beta.ppf(.025,cover,n-cover+1)) if cover else 0.,
       simultaneous_coverage=sum(r['simultaneous_query_coverage'] for r in rr),fallback=sum(r['fallback'] for r in rr),public_stops=sum(r['branch']=='public' for r in rr),
       concurrent_seconds_mean=float(np.mean([r['seconds'] for r in rr])),mean_draws=float(np.mean([r['draws'] for r in rr])))
    inds=rng.integers(0,n,(5000,n));a=counts['paired_weighted'];b=counts['active_cv'];boot=100*(1-a[inds].mean(axis=1)/b[inds].mean(axis=1))
    audit['saving_active']=float(100*(1-a.mean()/b.mean()));audit['saving_active_ci95']=[float(v) for v in np.quantile(boot,[.025,.975])]
    audit['saving_separate']=float(100*(1-a.mean()/counts['separate_weighted'].mean()));audit['saving_uniform']=float(100*(1-a.mean()/counts['paired_uniform'].mean()))
    summaries.append(audit)
 checks=dict(pools=n,grid_cells=12,method_runs=len(records),errors=sum(not r['acceptable'] for r in records),
   final_coverage=sum(r['covered'] for r in records),simultaneous_coverage=sum(r['simultaneous_query_coverage'] for r in records),
   max_identity_error=max(x['identity_error'] for x in groups),all_realized_errors_bounded=True,
   raw_file_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'results').glob('pool_*.json'))})
 save('summary.json',summaries);save('validation.json',checks)
 for s in summaries:
  print(s['name'],s['regimes'],'labels', {k:round(v['mean'],1) for k,v in s['methods'].items()},'fall',{k:v['fallback'] for k,v in s['methods'].items()},'save',round(s['saving_active'],1))
 print({k:v for k,v in checks.items() if k!='raw_file_hashes'})
if __name__=='__main__':main()

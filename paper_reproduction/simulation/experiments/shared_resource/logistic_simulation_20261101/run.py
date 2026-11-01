"""Controlled fixed-CV experiment. Audit labels never enter acquisition policies."""
from pathlib import Path
import os,sys,json,time,argparse,hashlib,gc
from datetime import datetime,timezone,timedelta
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_cv_attribution_20261031'))
from run import base,previous,np,paired_preconditioned,active_cv,zero_helper,finish,moment_trial
from scipy.special import expit,ndtr

METHODS=['paired_weighted','paired_uniform','separate_weighted','active_cv']
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def save(p,x):base.save(p,x)
def configuration():
 return dict(schema='controlled-shared-logistic-fixed-cv-v1',registered=now(),pool_size=2048,attributes=4,dimension=8,
  train_size=64,test_size=64,splits=16,penalties=[.125,.5,2.],alphas=[.1,.6],signals=[0.,4.],tolerance=.01,delta=.05,
  query_checks=[128*2**k for k in range(9)],active_initial_draws=32,active_max_draws_per_fold=4096,
  inner_lambdas=[.005,.01,.02,.05,.1,.2,.4,.6,.8,.95],outer_betting_fractions=[.05,.1,.2,.4,.6,.8,.95],
  base_seed=202611010101,smoke_seed=202611010901,methods=METHODS,
  generator='U and V have four independent Rademacher coordinates divided by 2. Hadamard row h_a codes attribute a. Z_A[a]=(h_a*U,e_a)/sqrt(2); Z_B[a]=(h_a*((1-alpha)*U+alpha*V),e_a)/sqrt(2). Row norms at most 1. Shared true coefficient theta_s=s*(1,1/2,-1/2,1/4,0,0,0,0). Conditional marginal p_a=sigmoid(Z_A[a] theta_s). Y_a=1{Phi(sqrt(.3) C+sqrt(.7) E_a)<=p_a}, with independent standard normal C,E_a. Objects independent; labels within an object correlated.',
  design='Sixteen fixed independent random train/test splits, disjoint within a split. One preassigned final subset per data pool. All configurations share U,V,latent uniforms and splits within a replicate. Different base replicates independent.',
  policies='Zero-anchor shared logistic polynomial, min(paired,separate) allowance for paired methods, same public entry check and nine inspection times. Uniform changes only term probabilities and corresponding range. Separate-weighted changes only the allowance. Active CV uses exact training fits and corrected selective attribute evaluation with valid inner and outer intervals.',
  target='Exact paired optimized BCE difference averaged over the fixed sixteen-split plan. Fully labeled audit is separate from all acquisition policies.',
  accounting='Unique object-attribute labels across comparison, exact-CV fallback, and the shared final 64-object training subset. No pilot or calibration labels. Every method is charged its own matrix setup, bounds, queries and fitting. Common synthetic data generation and hidden audits excluded from decision time.',
  fallback='Moment paths unresolved at 32768 draws complete exact CV with their cache. Active CV retains its verified numerical and interval fallback.',
  retention='All twelve grid cells and all formal pools retained. Smoke pools excluded from formal summaries. Formal replication count selected only from smoke runtime.',
  confidence='delta=.05 per method and configuration; not simultaneous across configurations. Formal summaries use independent pool replicates.',
  planned_pools=100,paths_per_pool=1)

def progress(done,total,note,state='running'):
 if os.environ.get('SIM_WORKER')=='1':return
 obj=dict(done=done,total=total,percent=100*done/max(total,1),note=note,state=state,updated=now(),pid=os.getpid())
 save(ROOT/'progress.json',obj)
 (ROOT/'进程.txt').write_text('共享 Logistic 控制模拟\n'+obj['updated']+f'\n状态：{state}\n方法运行：{done}/{total}（{obj["percent"]:.1f}%）\nPID：{os.getpid()}\n{note}\n',encoding='utf-8-sig')

def generate(cfg,seed):
 rng=np.random.default_rng(seed);m=cfg['pool_size'];r=cfg['attributes']
 u=(2*rng.integers(0,2,(m,4))-1)/2.;v=(2*rng.integers(0,2,(m,4))-1)/2.
 h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
 eye=np.broadcast_to(np.eye(r),(m,r,r))
 a=np.concatenate((u[:,None,:]*h[None,:,:],eye),axis=2)/np.sqrt(2)
 noise=ndtr(np.sqrt(.3)*rng.normal(size=(m,1))+np.sqrt(.7)*rng.normal(size=(m,r)))
 orders=[rng.permutation(m) for _ in range(cfg['splits'])];n=cfg['train_size'];q=cfg['test_size']
 splits=[(o[:n],o[n:n+q]) for o in orders];final=rng.choice(m,n,replace=False)
 return a,u,v,h,eye,noise,splits,final

def uniform_plan(poly):
 start=time.perf_counter();k=poly['terms'];q=np.full(k,1/k);cdf=np.arange(1,k+1,dtype=float)/k
 ratio=poly['coef']/q;radius=float(abs(ratio).max())
 return dict(pred=np.zeros(k),q=q,cdf=cdf,base=float(poly['constant']),low=-radius,high=radius,width=2*radius,seconds=time.perf_counter()-start)

def public_trial(poly,plan,oracle,seed,cfg,epsilon):
 start=time.perf_counter();mass=float(abs(poly['coef']).sum());center=float(poly['constant']);guard=2e-9
 lo,hi=center-mass-epsilon-guard,center+mass+epsilon+guard
 choice=base.decision(lo,hi,cfg['tolerance'])
 if choice!='unresolved':
  return dict(choice=choice,lower=lo,upper=hi,estimate=center,draws=0,seconds=time.perf_counter()-start,history=[],branch='public')
 return moment_trial(poly,plan,oracle,seed,cfg['query_checks'],cfg['tolerance'],cfg['delta'],epsilon)

def run(smoke=False,start_pool=0,stop_pool=None):
 cfg=json.loads((ROOT/'protocol.json').read_text('utf-8'))
 if not smoke:
  assert (ROOT/'formal_registration.json').exists(),'Freeze runtime-based repeat count first.'
  pools=json.loads((ROOT/'formal_registration.json').read_text('utf-8'))['pools']
 else:pools=1
 end=pools if stop_pool is None else min(pools,stop_pool)
 dest=ROOT/('smoke' if smoke else 'results');dest.mkdir(exist_ok=True)
 total=pools*12*len(METHODS);begin=time.perf_counter()
 for group in range(start_pool,end):
  if (dest/f'pool_{group:03}.json').exists():continue
  seed=(cfg['smoke_seed'] if smoke else cfg['base_seed'])+group*100
  tick=time.perf_counter();a,u,v,h,eye,noise,splits,final=generate(cfg,seed);generation_seconds=time.perf_counter()-tick
  assert np.max(np.linalg.norm(a,axis=-1))<=1+1e-12
  np.savez_compressed(dest/f'pool_{group:03}.npz',u=u,v=v,latent_uniforms=noise,train=np.array([x[0] for x in splits]),test=np.array([x[1] for x in splits]),final=final)
  groups=[];rows=[];histories={};cell=0
  for alpha in cfg['alphas']:
   b=np.concatenate((((1-alpha)*u+alpha*v)[:,None,:]*h[None,:,:],eye),axis=2)/np.sqrt(2)
   assert np.max(np.linalg.norm(b,axis=-1))<=1+1e-12
   for lam in cfg['penalties']:
    progress(group*48+len(rows),total,f'数据池 {group+1}/{pools}：alpha={alpha}, lambda={lam}，计算公开误差界')
    public=previous.public_bounds(a,b,splits,lam)
    t=time.perf_counter();pair=[paired_preconditioned(a[tr],b[tr],a[te],b[te],lam) for tr,te in splits]
    pair_seconds=time.perf_counter()-t;pair_eps=float(np.mean([p['epsilon'] for p in pair]));eps=min(pair_eps,public['epsilon'])
    poly=base.polynomial(a,b,splits,lam);poly['epsilon']=eps
    weighted=base.query_plan(poly,zero_helper(len(a),cfg['attributes']),cfg['attributes']);uniform=uniform_plan(poly)
    assert np.all(poly['left']!=poly['right'])
    for signal in cfg['signals']:
     name=f'a{alpha}_l{lam}_s{signal}';cell+=1
     theta=signal*np.array([1,.5,-.5,.25,0,0,0,0]);prob=expit(np.einsum('ird,d->ir',a,theta));y=(noise<=prob).astype(float)
     audit=base.exact_cv(a,b,base.Oracle(y),splits,lam);target=audit['delta']
     polynomial_audit=base.audit_polynomial(poly,y,splits,a,b,lam);surrogate=polynomial_audit['surrogate_delta'];err=target-surrogate
     assert abs(err)<=eps+2e-9 and abs(err)<=public['epsilon']+2e-9
     # Directly check the finite-query expectation under both probability rules.
     signs=(2*y-1).ravel();truth=poly['coef']*signs[poly['left']]*signs[poly['right']]
     for plan in [weighted,uniform]:
      expectation=plan['base']+float(plan['q']@(truth/plan['q']))
      assert abs(expectation-surrogate)<2e-10
     g=cfg['tolerance'];regime='both' if abs(target)<=g else 'A_only' if target<0 else 'B_only'
     meta=dict(group=group,name=name,alpha=alpha,lam=lam,signal=signal,g=g,target=target,regime=regime)
     detail=dict(meta,surrogate=surrogate,actual_error=err,paired_epsilon=pair_eps,separate_epsilon=public['epsilon'],epsilon=eps,
       coefficient_mass=float(abs(poly['coef']).sum()),uniform_radius=uniform['width']/2,terms=poly['terms'],identity_error=polynomial_audit['identity_error'],
       bound_seconds=pair_seconds,public_seconds=public['seconds'],poly_seconds=poly['seconds'])
     groups.append(detail);qs=seed+10000000+cell*101;method_hist={}
     for method in METHODS:
      oracle=base.Oracle(y)
      if method=='active_cv':
       trial=active_cv(a,b,oracle,splits,lam,g,public,cfg,qs,qs+1000000);setup=public['seconds']
       first=trial['choice'];used_eps=None;fallback=bool(trial['empty_protection'])
      else:
       plan=uniform if method=='paired_uniform' else weighted;used_eps=public['epsilon'] if method=='separate_weighted' else eps
       trial=public_trial(poly,plan,oracle,qs+2000000,cfg,used_eps);first=trial['choice'];fallback=first=='unresolved'
       if fallback:
        ex=base.exact_cv(a,b,oracle,splits,lam);trial['seconds']+=ex['seconds'];trial['branch']='exact_fallback'
        trial.update(choice=base.decision(ex['delta']-2e-9,ex['delta']+2e-9,g),lower=ex['delta']-2e-9,upper=ex['delta']+2e-9)
       setup=public['seconds']+poly['seconds']+plan['seconds']+(0 if method=='separate_weighted' else pair_seconds)
      row=finish(method,oracle,trial,final,a,b,lam,target,g,setup,meta)
      hist=trial['history'];row.update(epsilon=used_eps,fallback=fallback,first_choice=first,
       simultaneous_query_coverage=all(v['lower']-1e-10<=target<=v['upper']+1e-10 for v in hist))
      rows.append(row);method_hist[method]=hist
     histp=method_hist['paired_weighted'];hists=method_hist['separate_weighted']
     for x,z in zip(histp,hists):assert x['estimate']==z['estimate'] and x['statistical_radius']==z['statistical_radius']
     histories[name]=method_hist
     save(dest/f'pool_{group:03}_partial.json',dict(groups=groups,records=rows))
     progress(group*48+len(rows),total,f'池 {group+1}/{pools}, {name}, {regime}：标签 '+str({r['method']:r['labels_total'] for r in rows[-4:]}))
     print(json.dumps(dict(pool=group,cell=name,regime=regime,target=target,error=err,eps=eps,separate=public['epsilon'],labels={r['method']:r['labels_total'] for r in rows[-4:]})),flush=True)
    del poly,weighted,uniform;gc.collect()
  save(dest/f'pool_{group:03}.json',dict(seed=seed,groups=groups,records=rows,generation_seconds=generation_seconds))
  save(dest/f'history_{group:03}.json',histories)
  (dest/f'pool_{group:03}_partial.json').unlink()
 save(dest/f'run_{start_pool}_{end}.json',dict(seconds=time.perf_counter()-begin,start_pool=start_pool,stop_pool=end,smoke=smoke,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 progress(end*48,total,'小规模计时完成' if smoke else '正式模拟运行完成；正在汇总结果和修订文稿','smoke_complete' if smoke else 'experiment_complete')

def worker(group):
 import contextlib
 os.environ['SIM_WORKER']='1'
 with (ROOT/f'worker_{group:03}.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f):run(False,group,group+1)
 return group

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
 if args.register:
  assert not (ROOT/'protocol.json').exists();save(ROOT/'protocol.json',configuration());print('Registered 12 cells.')
 elif args.workers>1:
  from concurrent.futures import ProcessPoolExecutor,as_completed
  pools=json.loads((ROOT/'formal_registration.json').read_text('utf-8'))['pools'];start=time.perf_counter()
  pending=[i for i in range(pools) if not (ROOT/'results'/f'pool_{i:03}.json').exists()];done=pools-len(pending)
  progress(done*48,pools*48,f'正式方案：{pools} 个独立数据池，{args.workers} 个单线程工作进程')
  with ProcessPoolExecutor(max_workers=args.workers) as executor:
   futures=[executor.submit(worker,i) for i in pending]
   for f in as_completed(futures):
    group=f.result();done+=1
    progress(done*48,pools*48,f'完成 {done}/{pools} 个独立数据池；刚完成池 {group+1}')
  save(ROOT/'results'/'run_info.json',dict(pools=pools,runs=pools*48,workers=args.workers,seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  progress(pools*48,pools*48,'4,800 次方法比较完成；汇总与文稿修订中','experiment_complete')
 else:run(args.smoke,args.start,args.stop)

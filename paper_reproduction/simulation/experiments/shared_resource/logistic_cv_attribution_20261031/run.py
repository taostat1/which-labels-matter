"""Prespecified tolerance grid and matched acquisition ablations."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,gc,copy
from datetime import datetime,timezone,timedelta
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_cv_focus_20261030'))
from run_validation import base,previous,study,np,paired_preconditioned,active_cv,zero_helper,finish
from check_geometry import geometry

METHODS=['active_cv','selective_paired','whole_object_purchases','no_early_stop','separate_bound']
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def save(p,o):base.save(p,o)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol():
    return dict(schema='explicit-image-text-tolerance-attribution-v1',specified=now(),
      domains=[dict(name='image',data='ADE20K',feature_domain='backbone',attributes=6,dimension=22,
                    candidate_A='DINOv2-S/14',candidate_B='DINOv2-B/14',interpolation=False,
                    train_size=128,test_size=128,lam=.5,tolerances=[.0025,.00375,.005]),
               dict(name='text',data='GoEmotions',feature_domain='text',attributes=8,dimension=20,
                    candidate_A='mean pooling',candidate_B='0.9 mean pooling + 0.1 first token',
                    interpolation=True,alpha=.1,train_size=64,test_size=128,lam=.125,tolerances=[.005])],
      pools_per_domain=20,paths_per_pool=3,pool_size=4096,splits=32,delta=.05,
      query_checks=[128*2**k for k in range(9)],
      active_initial_draws=32,active_max_draws_per_fold=4096,
      inner_lambdas=[.005,.01,.02,.05,.1,.2,.4,.6,.8,.95],
      outer_betting_fractions=[.05,.1,.2,.4,.6,.8,.95],
      pool_seed=202610313101,split_seed=202610313102,query_seed=202610313103,
      methods=METHODS,
      target='Exact mean optimized paired binary-cross-entropy difference on the prescribed 32-split plan, conditional on the pool and hidden labels.',
      final_training='A common prespecified subset with the same size as each split training sample; all previously purchased labels are reused.',
      comparison='Active CV fits complete training subsets, actively evaluates attributes, reuses labels and stops early. All moment variants use the same zero anchor, polynomial, public coefficient probabilities, queries, confidence radius, and label cache rule.',
      ablations=dict(whole_object_purchases='When a sampled product requests an attribute, buy every attribute on that object. Keep the same estimator and decision times. This isolates purchase granularity; it is a matched ablation, not an optimized all-label competitor.',
                    no_early_stop='Use selective purchases and the paired allowance, but make no decision until the same prespecified 32768-product cap. Keep the same nine-look radius, so only the stopping policy is removed.',
                    separate_bound='Use the separate-candidate allowance in place of min(paired,separate). Every other component is unchanged.'),
      fallback='An unresolved moment variant at the cap completes exact CV with its own paid-label cache. Numerical tolerance is included in all intervals.',
      error_control='delta=.05 for each prespecified configuration and method. No simultaneous claim across configurations. All interval look counts remain nine, including the no-early-stop ablation.',
      pairing='Within a pool, all tolerances and moment variants share product-query seeds and final subsets. The active CV fold order and evaluation-query seeds are also paired across tolerances.',
      retention='Every prespecified pool, tolerance and method is reported, regardless of target, saving, or fallback. New independent pool and query seeds; pools can overlap within the same fixed corpus.',
      audit='Full-label targets and actual approximation errors are computed in a separate audit. Acquisition and stopping functions never receive these values.',
      accounting='Distinct object-attribute union across comparison and final training. Zero pilot and calibration labels. Feature calculations and fitting seconds are reported separately.',
      analysis='Means of complete labels, SD of pool means, paired pool-bootstrap intervals, coverage and acceptable decisions, first-moment decisions and fallback counts. Ablation effects are one-component comparisons and are not added together.',
      relation_to_prior='The grid halves the existing tolerance and includes its midpoint. Choices use earlier study results; all current pools and query paths are new. The weak-penalty text case is inherited unchanged to assess paired-error cancellation.')

def register():
    p=ROOT/'protocol.json'
    if not p.exists():save(p,protocol())
    return json.loads(p.read_text('utf-8'))

def progress(done,total,note,state='running'):
    obj=dict(done=done,total=total,percent=100*done/max(1,total),note=note,state=state,updated=now(),pid=os.getpid())
    save(ROOT/'progress.json',obj)
    (ROOT/'进程.txt').write_text('容忍度网格、成对消融与论文修订\n'+obj['updated']+'\n'
        f'实验方法运行：{done}/{total}（{obj["percent"]:.1f}%）\n状态：{state}\nPID：{os.getpid()}\n{note}\n'
        '进度表示计算完成比例；覆盖率、合格率和回退次数在结果中单独记录。\n',encoding='utf-8-sig')

class WholeObjectOracle(base.Oracle):
    def query_keys(self,keys,stage):
        keys=np.asarray(keys,dtype=np.int64)
        ids=np.unique(keys//self.r)
        allkeys=(ids[:,None]*self.r+np.arange(self.r)).ravel()
        base.Oracle.query_keys(self,allkeys,stage)
        return self._y.reshape(-1)[keys]

def moment_trial(poly,plan,oracle,seed,checks,g,delta,epsilon,early=True):
    tick=time.perf_counter();rng=np.random.default_rng(seed);su=0.;sq=0.;n=0;history=[]
    t=np.log(4*len(checks)/delta)
    for end in checks:
        idx=np.searchsorted(plan['cdf'],rng.random(end-n),side='right')
        xl=2*oracle.query_keys(poly['left'][idx],'comparison')-1
        xr=2*oracle.query_keys(poly['right'][idx],'comparison')-1
        residual=poly['coef'][idx]/plan['q'][idx]*(xl*xr-plan['pred'][idx])
        assert residual.min()>=plan['low']-1e-10 and residual.max()<=plan['high']+1e-10
        su+=float(residual.sum());sq+=float(residual@residual);n=end
        variance=max(0.,(sq-su*su/n)/(n-1))
        stat=float(np.sqrt(2*variance*t/n)+7*plan['width']*t/(3*(n-1)))
        center=float(plan['base']+su/n);half=stat+epsilon+2e-9
        lo,hi=center-half,center+half;choice=base.decision(lo,hi,g)
        history.append(dict(draws=n,estimate=center,statistical_radius=stat,approximation_radius=epsilon,
            lower=lo,upper=hi,labels=oracle.count,supported_choice=choice))
        if early and choice!='unresolved':break
    return dict(choice=choice,lower=lo,upper=hi,estimate=center,draws=n,
                seconds=time.perf_counter()-tick,history=history,branch='moments')

def main(smoke=False):
    cfg=register();dest=ROOT/('smoke' if smoke else 'results');dest.mkdir(exist_ok=True)
    assert not (dest/'records.json').exists(),'Run exists. Do not overwrite formal evidence.'
    npools=1 if smoke else cfg['pools_per_domain'];npaths=1 if smoke else cfg['paths_per_pool']
    total=npools*npaths*sum(len(x['tolerances']) for x in cfg['domains'])*len(METHODS)
    (dest/'executed_run.py').write_bytes(Path(__file__).read_bytes())
    save(dest/'protocol.json',cfg);save(dest/'source_hash.json',dict(sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.json')))
    tick=time.perf_counter();done=0;rows=[];groups=[];load_seconds={}
    progress(done,total,'方案已冻结；开始独立的小规模耗时检查' if smoke else '正式方案已冻结；开始新数据池验证')
    for domain_idx,case in enumerate(cfg['domains']):
        start=time.perf_counter();a0,b0,y0,pools=study.load_domain(case['feature_domain'])
        a0=a0.astype(float);b0=b0.astype(float)
        a0/=np.maximum(1,np.linalg.norm(a0,axis=-1,keepdims=True));b0/=np.maximum(1,np.linalg.norm(b0,axis=-1,keepdims=True))
        if case['interpolation']:b0=(1-case['alpha'])*a0+case['alpha']*b0
        assert y0.shape[1]==case['attributes'] and a0.shape[-1]==case['dimension']
        load_seconds[case['name']]=time.perf_counter()-start
        for group in range(npools):
            offset=domain_idx*100000+(10000000 if smoke else 0)
            ids=np.random.default_rng(cfg['pool_seed']+offset+group).choice(pools['training'],cfg['pool_size'],replace=False)
            a,b,y=a0[ids],b0[ids],y0[ids];r=y.shape[1];n=case['train_size'];m=case['test_size'];lam=case['lam']
            rng=np.random.default_rng(cfg['split_seed']+offset+group)
            orders=[rng.permutation(len(ids)) for _ in range(cfg['splits'])]
            splits=[(x[:n],x[n:n+m]) for x in orders]
            finals=[rng.choice(len(ids),n,replace=False) for _ in range(npaths)]
            tag=f'{case["name"]}_{group:02}'
            np.savez_compressed(dest/f'schedule_{tag}.npz',pool_ids=ids,train=np.asarray([x[0] for x in splits]),
                                test=np.asarray([x[1] for x in splits]),final=np.asarray(finals))
            progress(done,total,f'{case["name"]} 第 {group+1}/{npools} 池：计算公开系数与误差界')
            public=previous.public_bounds(a,b,splits,lam)
            start=time.perf_counter()
            pair=[paired_preconditioned(a[tr],b[tr],a[te],b[te],lam) for tr,te in splits]
            pair_eps=float(np.mean([x['epsilon'] for x in pair]));pair_seconds=time.perf_counter()-start
            start=time.perf_counter()
            geom=[geometry(a[tr].reshape(-1,a.shape[-1]),b[tr].reshape(-1,b.shape[-1]),
                           a[te].reshape(-1,a.shape[-1]),b[te].reshape(-1,b.shape[-1]),lam) for tr,te in splits]
            geometry_seconds=time.perf_counter()-start
            geom_eps=float(np.mean([x['epsilon'] for x in geom])) if all(x['valid'] for x in geom) else None
            poly=base.polynomial(a,b,splits,lam);poly['epsilon']=min(pair_eps,public['epsilon'])
            plan=base.query_plan(poly,zero_helper(len(a),r),r)
            exact=base.exact_cv(a,b,base.Oracle(y),splits,lam)
            audit=base.audit_polynomial(poly,y,splits,a,b,lam);target=exact['delta']
            err=target-audit['surrogate_delta'];assert abs(err)<=poly['epsilon']+1e-10
            if geom_eps is not None:assert abs(err)<=geom_eps+1e-10
            gi=dict(domain=case['name'],group=group,n=n,lam=lam,target=target,surrogate=audit['surrogate_delta'],actual_error=err,
                    paired_epsilon=pair_eps,separate_epsilon=public['epsilon'],epsilon=poly['epsilon'],geometry_epsilon=geom_eps,
                    gamma_positive=sum(x['valid'] for x in geom),geometry_seconds=geometry_seconds,
                    poly_seconds=poly['seconds'],bound_seconds=pair_seconds,public_seconds=public['seconds'],
                    terms=poly['terms'],range=plan['width'],pool_hash=hashlib.sha256(ids.tobytes()).hexdigest(),
                    public_geometry=geom)
            groups.append(gi);save(dest/'groups.json',groups)
            for g in case['tolerances']:
                regime='both' if abs(target)<=g else 'A_only' if target<0 else 'B_only'
                name=case['name']+'-g'+str(g)
                for rep in range(npaths):
                    seed=cfg['query_seed']+offset+10000*group+rep
                    meta=dict(name=name,domain=case['name'],group=group,rep=rep,n=n,lam=lam,target=target,g=g,regime=regime)
                    histories={};moment_records={}
                    oa=base.Oracle(y)
                    active=active_cv(a,b,oa,splits,lam,g,public,cfg,seed,seed+1000000)
                    ra=finish('active_cv',oa,active,finals[rep],a,b,lam,target,g,public['seconds'],meta)
                    rows.append(ra);histories['active_cv']=active['history'];done+=1
                    for method in METHODS[1:]:
                        oracle=WholeObjectOracle(y) if method=='whole_object_purchases' else base.Oracle(y)
                        epsilon=public['epsilon'] if method=='separate_bound' else poly['epsilon']
                        trial=moment_trial(poly,plan,oracle,seed+2000000,cfg['query_checks'],g,cfg['delta'],epsilon,method!='no_early_stop')
                        first_choice=trial['choice'];labels_before_fallback=oracle.count
                        if first_choice=='unresolved':
                            fallback=base.exact_cv(a,b,oracle,splits,lam)
                            trial['seconds']+=fallback['seconds'];trial['branch']='exact_fallback'
                            trial.update(choice=base.decision(fallback['delta']-2e-9,fallback['delta']+2e-9,g),
                                         lower=fallback['delta']-2e-9,upper=fallback['delta']+2e-9)
                        setup=public['seconds']+poly['seconds']+plan['seconds']+(0 if method=='separate_bound' else pair_seconds)
                        rm=finish(method,oracle,trial,finals[rep],a,b,lam,target,g,setup,meta)
                        rm.update(moment_choice=first_choice,labels_before_fallback=labels_before_fallback,epsilon=epsilon)
                        rows.append(rm);moment_records[method]=rm;histories[method]=trial['history'];done+=1
                    ref=histories['selective_paired'];whole=histories['whole_object_purchases']
                    assert len(ref)==len(whole)
                    assert all(x['estimate']==z['estimate'] and x['lower']==z['lower'] and x['upper']==z['upper'] for x,z in zip(ref,whole))
                    assert moment_records['whole_object_purchases']['labels_total']>=moment_records['selective_paired']['labels_total']
                    for method in ['no_early_stop','separate_bound']:
                        alt=histories[method]
                        assert all(x['estimate']==z['estimate'] and x['statistical_radius']==z['statistical_radius'] for x,z in zip(ref,alt))
                    save(dest/f'path_{tag}_{g}_{rep}.json',dict(meta=meta,active=ra,moments=moment_records,histories=histories))
                    save(dest/'records.json',rows)
                    counts={k:v['labels_total'] for k,v in moment_records.items()}
                    progress(done,total,f'{name} 池 {group+1} 路径 {rep+1}：主动 CV {ra["labels_total"]}；本方法 {counts["selective_paired"]}；{regime}')
                    print(json.dumps(dict(done=done,total=total,name=name,pool=group,rep=rep,active=ra['labels_total'],**counts,regime=regime)),flush=True)
            del poly,plan,pair,geom;gc.collect()
        del a0,b0,y0;gc.collect()
    save(dest/'run_info.json',dict(seconds=time.perf_counter()-tick,load_seconds=load_seconds,runs=len(rows),groups=len(groups),smoke=smoke))
    progress(done,total,'小规模耗时检查完成' if smoke else '正式实验全部完成；统计汇总和文稿修订进行中','smoke_complete' if smoke else 'experiment_complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.register:print(json.dumps(register(),ensure_ascii=False,indent=2))
    else:main(args.smoke)

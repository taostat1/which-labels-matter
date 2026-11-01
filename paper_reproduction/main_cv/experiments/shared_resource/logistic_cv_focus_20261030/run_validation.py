"""Prospective specification of pool-level repeated-split validation."""
from pathlib import Path
import os, sys, time, json, argparse, gc, hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_t4_paired_bound_20261030'))
from paired import paired_preconditioned, base, previous, study, np
from active_core import active_cv, zero_helper

def progress(done,total,note,state='running'):
    obj=dict(done=done,total=total,percent=100*done/max(1,total),note=note,state=state,
             pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    base.save(ROOT/'progress.json',obj)
    (ROOT/'进程.txt').write_text('重复随机划分：独立数据池验证\n'
       f'更新时间：{obj["updated"]}\n实验进度：{done}/{total}（{obj["percent"]:.1f}%）\n'
       f'状态：{state}\nPID：{os.getpid()}\n{note}\n'
       '进度为完成的计算比例；决策成功率另行报告。\n',encoding='utf-8-sig')

def finish(method,oracle,trial,final_ids,a,b,lam,target,g,setup,meta):
    before=oracle.count;start=time.perf_counter()
    labels=oracle.query_objects(final_ids,'final_training')
    z=a if trial['choice']=='A' else b
    theta,fit_info=base.fit(z[final_ids],labels,lam)
    final_seconds=time.perf_counter()-start
    ok=(target<=g if trial['choice']=='A' else target>=-g)
    row=dict(meta,method=method,choice=trial['choice'],covered=bool(trial['lower']<=target<=trial['upper']),
       acceptable=bool(ok),lower=trial['lower'],upper=trial['upper'],labels_total=oracle.count,
       labels_before_final=before,stage_labels=dict(oracle.stage_counts),
       seconds=setup+trial['seconds']+final_seconds,final_fit_seconds=final_seconds,
       final_fit_iterations=fit_info['iterations'],branch=trial.get('branch','active'),
       draws=trial.get('draws',0),opened_folds=trial.get('opened_folds',0))
    assert row['labels_total']==sum(row['stage_labels'].values())
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=json.loads((ROOT.parent/'logistic_active_cv_budget_20261030/protocol.json').read_text('utf-8'))
    cfg.update(pools=1 if args.smoke else 20,paths_per_pool=1 if args.smoke else 3,
               pool_size=4096,splits=32,train_size=64,test_size=128,
               pool_seed=202610310101 if args.smoke else 202610310201,
               split_seed=202610310102 if args.smoke else 202610310202,
               query_seed=202610310103 if args.smoke else 202610310203)
    cases=[dict(name='T4-near',alpha=.1,lam=.125,g=.005),
           dict(name='T4-separated',alpha=.2,lam=.125,g=.005)]
    dest=ROOT/('smoke' if args.smoke else 'results');dest.mkdir(exist_ok=True)
    protocol=dict(configuration=cfg,cases=cases,
       design='Independent draws of 4096-object pools from the fixed training corpus. Pools can overlap. Pool means are the units for variability; query paths are nested within pools.',
       target='Exact finite CV score for 32 prespecified repeated random splits; no population-risk interpretation.',
       policies='Same-permission active CV and zero-anchor coefficient-sampled moments. All labels reused. Identical final 64-object subset for paired policies.',
       continuation='Run the same finite-grid moment interval even if its deterministic allowance exceeds g. That condition is sufficient at a tie, not necessary for a directional decision. The prescribed draw cap is unchanged. If unresolved, complete exact CV using the same cache.',
       preregistration='Both interpolation values, penalty, tolerance, pool count, seeds, and query caps are fixed before loading labels. No pool or configuration is selected or removed using its risk or saving.',
       audit='All-label fits and exact polynomial values are computed separately and are never inputs to a query or stopping policy.',
       accounting='Unique object-attribute labels, including final training. No auxiliary predictor or label calibration. Feature/matrix computation timed separately and charged to each independent decision.',
       reporting='Primary counts are means across pools. SD describes pool means, with three paths per pool. Query-seed dispersion remains in raw records. Coverage, choice, fallback, and actual decision regime reported for every pool.')
    base.save(dest/'protocol.json',protocol)
    base.save(dest/'source_hash.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    start=time.perf_counter();total=cfg['pools']*len(cases)*cfg['paths_per_pool']*2;done=0
    progress(done,total,'载入固定特征；方案与随机种子已保存')
    a0,o0,y0,pools=study.load_domain('text');a0=a0.astype(float);o0=o0.astype(float)
    a0/=np.maximum(1,np.linalg.norm(a0,axis=-1,keepdims=True))
    o0/=np.maximum(1,np.linalg.norm(o0,axis=-1,keepdims=True))
    rows=[];groups=[]
    for group in range(cfg['pools']):
        ids=np.random.default_rng(cfg['pool_seed']+group).choice(pools['training'],cfg['pool_size'],replace=False)
        a=a0[ids];o=o0[ids];y=y0[ids];r=y.shape[1]
        rng=np.random.default_rng(cfg['split_seed']+group)
        orders=[rng.permutation(len(ids)) for _ in range(cfg['splits'])]
        splits=[(x[:64],x[64:192]) for x in orders]
        finals=[rng.choice(len(ids),64,replace=False) for _ in range(cfg['paths_per_pool'])]
        np.savez_compressed(dest/f'schedule_{group:02}.npz',pool_ids=ids,train=np.array([x[0] for x in splits]),
                            test=np.array([x[1] for x in splits]),final=np.asarray(finals))
        for ci,case in enumerate(cases):
            name=case['name'];lam=case['lam'];g=case['g'];b=(1-case['alpha'])*a+case['alpha']*o
            progress(done,total,f'池 {group+1}/{cfg["pools"]}，{name}：公开特征界及风险多项式')
            public=previous.public_bounds(a,b,splits,lam)
            tick=time.perf_counter()
            bounds=[paired_preconditioned(a[tr],b[tr],a[te],b[te],lam) for tr,te in splits]
            paired=float(np.mean([x['epsilon'] for x in bounds]));bound_seconds=time.perf_counter()-tick
            poly=base.polynomial(a,b,splits,lam);poly['epsilon']=min(paired,public['epsilon'])
            plan=base.query_plan(poly,zero_helper(len(a),r),r)
            target=base.exact_cv(a,b,base.Oracle(y),splits,lam)['delta']
            audit=base.audit_polynomial(poly,y,splits,a,b,lam)
            assert abs(target-audit['surrogate_delta'])<=poly['epsilon']+1e-10
            regime='both' if abs(target)<=g else 'A_only' if target<0 else 'B_only'
            gi=dict(group=group,name=name,alpha=case['alpha'],lam=lam,g=g,target=target,
                    surrogate=audit['surrogate_delta'],actual_error=target-audit['surrogate_delta'],
                    paired_epsilon=paired,separate_epsilon=public['epsilon'],epsilon=poly['epsilon'],
                    regime=regime,bound_seconds=bound_seconds,poly_seconds=poly['seconds'],
                    terms=poly['terms'],range=plan['width'],pool_hash=hashlib.sha256(ids.tobytes()).hexdigest())
            groups.append(gi);base.save(dest/'groups.json',groups)
            for rep in range(cfg['paths_per_pool']):
                seed=cfg['query_seed']+10000*group+1000*ci+rep
                oa=base.Oracle(y);active=active_cv(a,b,oa,splits,lam,g,public,cfg,seed,seed+1000000)
                meta=dict(group=group,rep=rep,name=name,target=target,g=g,regime=regime)
                ra=finish('active_cv',oa,active,finals[rep],a,b,lam,target,g,public['seconds'],meta)
                rows.append(ra);done+=1
                om=base.Oracle(y)
                trial=base.query_trial(poly,plan,om,np.random.default_rng(seed+2000000),cfg['query_checks'],g,cfg['delta'])
                trial['branch']='moments'
                if trial['choice']=='unresolved':
                    exact=base.exact_cv(a,b,om,splits,lam)
                    trial['seconds']+=exact['seconds'];trial['branch']='exact_fallback'
                    trial.update(choice=base.decision(exact['delta']-2e-9,exact['delta']+2e-9,g),
                                 lower=exact['delta']-2e-9,upper=exact['delta']+2e-9)
                rm=finish('selective_moments',om,trial,finals[rep],a,b,lam,target,g,
                          bound_seconds+poly['seconds']+plan['seconds'],meta)
                rows.append(rm);done+=1
                base.save(dest/f'path_{group:02}_{ci}_{rep}.json',dict(active=ra,moments=rm,history=trial['history']))
                base.save(dest/'records.json',rows)
                progress(done,total,f'池 {group+1}，{name}，路径 {rep+1}：主动 CV {ra["labels_total"]}；选择性矩 {rm["labels_total"]}；{regime}')
                print(json.dumps(dict(done=done,group=group,name=name,rep=rep,active=ra['labels_total'],moments=rm['labels_total'],regime=regime,epsilon=poly['epsilon'],target=target)),flush=True)
            del poly,plan,bounds;gc.collect()
    base.save(dest/'run_info.json',dict(seconds=time.perf_counter()-start,records=len(rows)))
    progress(done,total,'实验计算完成；正在核对标签记账、池间变异和论文更新','experiment_complete')

if __name__=='__main__':main()

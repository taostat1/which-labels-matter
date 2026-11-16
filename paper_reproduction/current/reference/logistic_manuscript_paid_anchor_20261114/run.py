"""Paid nonzero anchors on the unchanged manuscript fixed-CV configurations."""
from anchor_core import *
import argparse,gc,hashlib,copy
from types import SimpleNamespace
from whichlabels import prepare_comparison,compare
from whichlabels.api import _exact_score

SOURCE=ROOT.parent/'logistic_cp_revision_20261106/real'
CASES={'image':dict(cache='cross_backbone_20261028',n=128,r=6,d=22,lam=.5,g=[.0025,.00375,.005]),
       'text':dict(cache='goemotions_comparison_20261028',n=64,r=8,d=20,lam=.125,g=[.005])}
CHECKS=[128,256,512,1024,2048,4096,8192,16384,32768]

def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def register():
    protocol=dict(date='2026-11-14',purpose='Compare paid nonzero Newton anchors against the released zero-anchor LEC on unchanged manuscript models.',
      studies=['attribution','coverage'],cases=CASES,pools=[0,1,2],paths_per_pool=3,pilot_objects=[0,16,32,64,128],
      pilot_seed=202611140201,query_seed='Reuse each recorded study query_seed + domain_offset + 10000*pool + rep + 2000000',
      final_subsets='Exactly the three final subsets saved in each published schedule.',
      plans='Exactly the first three recorded pools and their fixed 32 splits; no filtering or retuning.',
      tolerance_grid='All original tolerances retained.',delta=.05,inspections=CHECKS,
      anchor='Candidate-specific all-coordinate L2 logistic fit to paid pilot objects. Nested uniform pilot subsets independent of labels. Pilot labels are reused across all splits within one decision.',
      conditional_target='The same exact fixed-plan optimized BCE difference; only the approximation changes.',
      conditional_algebra='Condition on paid labels. Retain constant, single-label and pair terms. Substitute known labels exactly. Compute conditional paired error certificate.',
      bound='Minimum of the native conditional paired certificate and a valid triangle transfer from the feature-only zero-anchor certificate. No oracle errors enter decisions.',
      sampling='Fixed coefficient-proportional draws; CP delta/9 plus numeric mapping guard; public entry check; no impossibility screen or post-query deterministic-cache stopping.',
      zero_control='The unchanged public whichlabels.prepare_comparison and compare API. Compare every control path with its frozen manuscript record.',
      fallback='Complete only the union of the fixed-plan training and evaluation labels, then numerically certified exact CV through the same paid cache.',
      accounting='Distinct pilot, comparison, exact-CV fallback, and final-training labels. No duplicated development charges or unbought oracle labels.',
      audit='Only after acquisition decisions, exact fits diagnose the unchanged target, actual approximation error, polynomial identity, and sampling variance. Audit labels excluded from acquisition.',
      replication='Three data pools and three query paths per pool; pool means are the independent aggregation units. All five pilot budgets are reported.',
      manuscript='Separate exploration; no released code, manuscript or prior result is overwritten.')
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text('utf-8'))==protocol,'Protocol mismatch; use a new folder.'
    else:save(path,protocol)
    paths=[Path(__file__),ROOT/'anchor_core.py']+list((WORK/'output/github/which-labels-logistic/src/whichlabels').glob('*.py'))
    source_hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest=ROOT/'source_hashes.json'
    if manifest.exists():assert json.loads(manifest.read_text('utf-8'))==source_hashes,'Source changes detected after registration.'
    else:save(manifest,source_hashes)

def load_domain(domain):
    folder=ROOT.parent/CASES[domain]['cache']/'prepared'
    f=np.load(folder/'features_public.npz');l=np.load(folder/'labels_oracle.npz')
    assert np.array_equal(f['ids'],l['ids'])
    a=np.array(f['a'],dtype=float);other=np.array(f['other'],dtype=float)
    a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    other/=np.maximum(1,np.linalg.norm(other,axis=-1,keepdims=True))
    b=other if domain=='image' else .9*a+.1*other
    y=((l['y']+1)/2).astype(float)
    assert np.isin(y,[0,1]).all()
    return a,b,y

def exact_fits(a,b,y,plan,lam):
    parameters=[];values=[]
    for tr,te in plan:
        row=[];risks=[]
        for z in (a,b):
            theta,info=fit(z[tr],y[tr],lam);row.append(theta);risks.append(risk(z[te],y[te],theta))
        parameters.append(row);values.append(risks[0]-risks[1])
    return parameters,float(np.mean(values))

def summarize_record(trial,cache,a,b,plan,lam,final,g,metadata,setup):
    began=time.perf_counter();fallback=trial['choice']=='unresolved'
    if fallback:
        exact,error=_exact_score(SimpleNamespace(a=a,b=b,splits=plan,penalty=lam),cache,'exact_cv_fallback')
        trial.update(choice=choose(exact-error,exact+error,g),interval=[exact-error,exact+error])
    assert trial['choice']!='unresolved'
    before=cache.count;labels=cache.query_objects(final,'final_training')
    z=a if trial['choice']=='A' else b
    theta,info=fit(z[final],labels,lam)
    assert cache.count==sum(cache.stage_counts.values())
    assert cache.seen.reshape(len(a),a.shape[1])[final].all()
    return dict(metadata,**trial,fallback=fallback,total_labels=cache.count,labels_before_final=before,
      new_final_training_labels=cache.count-before,stage_labels=dict(cache.stage_counts),
      final_parameter=theta.tolist(),final_fit=info,setup_seconds=setup,
      finish_seconds=time.perf_counter()-began)

def run(study,domain,pools,budgets,reps):
    register();cfg=CASES[domain];lam=cfg['lam'];offset=0 if domain=='image' else 100000
    source=SOURCE/study
    source_cfg=json.loads((source/'protocol.json').read_text('utf-8'))
    archived=json.loads((source/'results/records.json').read_text('utf-8'))
    a0,b0,y0=load_domain(domain)
    for pool in pools:
        folder=ROOT/'results'/study/domain/f'pool_{pool:02d}';folder.mkdir(parents=True,exist_ok=True)
        sch=np.load(source/'results'/f'schedule_{domain}_{pool:02d}.npz')
        ids=sch['pool_ids'];a=a0[ids];b=b0[ids];y=y0[ids]
        plan=tuple(zip(sch['train'],sch['test']));finals=sch['final'][:reps]
        order=np.random.default_rng(202611140201+offset+pool).permutation(len(ids))
        np.savez_compressed(folder/'schedule.npz',**{k:sch[k] for k in sch.files},pilot_order=order)
        print(study,domain,pool,'prepare original zero anchor',flush=True)
        zero=prepare_comparison(a,b,plan,lam)
        zc=float(zero.polynomial['constant']);zm=float(np.abs(zero.polynomial['coef']).sum());ze=zero.epsilon
        save(folder/'zero_public.json',dict(epsilon=ze,constant=zc,mass=zm,setup_seconds=zero.setup_seconds))
        exact_parameters=None;target=None
        for m in sorted(budgets):
            dest=folder/f'pilot_{m:03d}';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():
                print(study,domain,pool,m,'resume: complete',flush=True);continue
            tick=time.perf_counter();pilot=order[:m];rows=[];masks={}
            if m==0:
                for g in cfg['g']:
                    for rep,final in enumerate(finals):
                        seed=source_cfg['query_seed']+offset+10000*pool+rep+2000000
                        cache=LabelCache(len(ids),cfg['r'],ArrayLabelProvider(y))
                        row=compare(zero,cache,final,tolerance=g,delta=.05,inspections=CHECKS,seed=seed)
                        row.update(study=study,domain=domain,pool=pool,pilot_objects=0,rep=rep,g=g,
                                   epsilon=ze,mass=zm,pilot_labels=0,seed=seed)
                        reference=next(x for x in archived if x['domain']==domain and x['group']==pool and x['rep']==rep and x['g']==g and x['method']=='selective_paired')
                        row['archive_parity']={k:row[new]==reference[k] for k,new in [('choice','choice'),('labels_total','total_labels'),('draws','draws')]}
                        assert all(row['archive_parity'].values()),row['archive_parity']
                        rows.append(row);masks[f'{g}_{rep}']=np.packbits(cache.seen)
                save(dest/'decisions_before_audit.json',rows)
                if exact_parameters is None:exact_parameters,target=exact_fits(a,b,y,plan,lam)
                x=(2*y-1).ravel();poly=zero.polynomial
                surrogate=zc+float(poly['coef']@(x[poly['left']]*x[poly['right']]))
                au=dict(target=target,surrogate=surrogate,actual_error=abs(target-surrogate),
                    sampling_variance=max(0.,zm**2-(surrogate-zc)**2))
                assert au['actual_error']<=ze+2e-8
                setup=zero.setup_seconds
                save(dest/'public.json',dict(epsilon=ze,native_epsilon=ze,constant=zc,mass=zm,pilot_labels=0,terms=len(poly['coef']),setup_seconds=setup))
                del poly
            else:
                # Retain only the public scalar baseline certificate, not its large sparse arrays.
                if zero is not None:del zero;zero=None;gc.collect()
                paid=LabelCache(len(ids),cfg['r'],ArrayLabelProvider(y));yp=paid.query_objects(pilot,'pilot')
                anchors=[fit(z[pilot],yp,lam)[0] for z in (a,b)]
                np.savez_compressed(dest/'anchors.npz',a=anchors[0],b=anchors[1],pilot=pilot)
                print(study,domain,pool,m,'prepare conditional polynomial',flush=True)
                poly=prepare(a,b,plan,anchors,lam,paid._values.copy())
                native=poly['epsilon'];transfer=ze+abs(poly['constant']-zc)+poly['mass']+zm
                poly['epsilon']=min(native,transfer);setup=time.perf_counter()-tick
                public=dict(pilot_objects=m,pilot_labels=paid.count,epsilon=poly['epsilon'],native_epsilon=native,
                    zero_transfer_epsilon=transfer,constant=poly['constant'],mass=poly['mass'],
                    single_terms=poly['single_terms'],terms=poly['terms'],bounds=poly['bounds'],setup_seconds=setup,
                    anchor_norms=[float(np.linalg.norm(t)) for t in anchors])
                save(dest/'public.json',public)
                print(study,domain,pool,m,'epsilon',poly['epsilon'],'mass',poly['mass'],flush=True)
                assert not np.intersect1d(poly['left'],poly['known_keys']).size
                assert not np.intersect1d(poly['right'][poly['right']>=0],poly['known_keys']).size
                for g in cfg['g']:
                    for rep,final in enumerate(finals):
                        seed=source_cfg['query_seed']+offset+10000*pool+rep+2000000
                        cache=LabelCache(len(ids),cfg['r'],ArrayLabelProvider(y));cache.query_objects(pilot,'pilot')
                        qt=time.perf_counter();trial=sample(poly,cache,seed,g);qs=time.perf_counter()-qt
                        meta=dict(study=study,domain=domain,pool=pool,pilot_objects=m,rep=rep,g=g,seed=seed,
                            epsilon=poly['epsilon'],mass=poly['mass'],pilot_labels=m*cfg['r'],query_seconds=qs)
                        row=summarize_record(trial,cache,a,b,plan,lam,final,g,meta,setup)
                        rows.append(row);masks[f'{g}_{rep}']=np.packbits(cache.seen)
                save(dest/'decisions_before_audit.json',rows)
                if exact_parameters is None:exact_parameters,target=exact_fits(a,b,y,plan,lam)
                au=audit(poly,a,b,y,plan,anchors,lam,exact_parameters)
                del poly,paid;gc.collect()
            reference=next(x for x in archived if x['domain']==domain and x['group']==pool)
            assert abs(au['target']-reference['target'])<1e-9,(au['target'],reference['target'])
            for row in rows:
                lo,hi=row['interval'];g=row['g']
                row.update(target=target,actual_error=au['actual_error'],sampling_variance=au['sampling_variance'],
                    acceptable=bool(target<=g if row['choice']=='A' else target>=-g),coverage=bool(lo<=target<=hi))
            save(dest/'records.json',rows);save(dest/'audit.json',au)
            np.savez_compressed(dest/'masks.npz',**masks)
            save(dest/'complete.json',dict(paths=len(rows),seconds=time.perf_counter()-tick))
            print(study,domain,pool,m,'DONE labels',[(g,float(np.mean([x['total_labels'] for x in rows if x['g']==g]))) for g in cfg['g']],flush=True)
        del a,b,y,zero;gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['attribution','coverage'],required=True)
    p.add_argument('--domain',choices=list(CASES),required=True);p.add_argument('--pools',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--budgets',nargs='+',type=int,default=[0,16,32,64,128]);p.add_argument('--reps',type=int,default=3)
    args=p.parse_args();assert all(x in [0,1,2] for x in args.pools)
    assert all(x in [0,16,32,64,128] for x in args.budgets) and args.reps in [1,2,3]
    run(args.study,args.domain,args.pools,args.budgets,args.reps)

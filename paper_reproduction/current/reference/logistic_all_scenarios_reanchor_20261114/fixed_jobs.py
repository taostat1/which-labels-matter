from pathlib import Path
import os,sys,json,gc,time,argparse,contextlib,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parent
import engine as e
import numpy as np
from scipy.special import expit,ndtr
from whichlabels import prepare_comparison
from whichlabels._intervals import endpoints,mapping_guard
CP=SHARED/'logistic_cp_revision_20261106';SIM=SHARED/'logistic_simulation_refinement_20261106/simulation'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def study_module():
    sys.path.insert(0,str(SHARED/'shared_logistic_full_20261029'))
    import study
    return study
def jobs():
    out=[]
    for family in ['attribution','coverage']:
        cfg=read(CP/'real'/family/'protocol.json')
        for dom in cfg['domains']:
            for g in dom['tolerances']:
                for pool in range(3):out.append(dict(family=family,domain=dom['name'],case=dom['name'],g=g,lam=dom['lam'],pool=pool,alpha=dom.get('alpha',1.)))
    for c in read(CP/'cache/protocol.json')['cases']:
        for g in c['tolerances']:
            for pool in range(3):out.append(dict(family='cache',domain=c['domain'],case=c['name'],g=g,lam=c['lam'],pool=pool,alpha=c['alpha']))
    cases=study_module().CASES
    for si,scenario in enumerate(['KFold_n64','KFold_n256']):
        for ci,c in enumerate(cases[:4]):
            for pool in range(3):out.append(dict(family='fivefold',domain='text',case=f'T{ci+1}',scenario=scenario,si=si,ci=ci,g=c['g'],lam=c['lam'],alpha=c['alpha'],pool=pool))
    for family,folder,lams in [('simulation',SIM,[.25,.5,2.]),('boundary',CP/'simulation',[.125])]:
        cfg=read(folder/'protocol.json')
        for alpha in cfg['alphas']:
            for lam in lams:
                for signal in cfg['signals']:
                    for pool in range(3):out.append(dict(family=family,domain='synthetic',case=f'a{alpha}_l{lam}_s{signal}',alpha=alpha,lam=lam,signal=signal,g=.01,pool=pool))
    for i,j in enumerate(out):j['id']=i
    return out
def load(j):
    family=j['family'];pool=j['pool'];domain=j['domain'];alpha=j['alpha']
    if family in ['attribution','coverage']:
        folder=CP/'real'/family;cfg=read(folder/'protocol.json')
        sch=np.load(folder/'results'/f'schedule_{domain}_{pool:02d}.npz')
        a,b,y=e.load_domain(domain);ids=sch['pool_ids'];a,b,y=a[ids],b[ids],y[ids]
        seed=cfg['query_seed']+(0 if domain=='image' else 100000)+10000*pool+2000000
        refs=[r for r in read(folder/'results/records.json') if r['domain']==domain and r['group']==pool and r['rep']==0 and r['g']==j['g']]
        plan=list(zip(sch['train'],sch['test']));final=sch['final'][0]
    elif family=='cache':
        folder=CP/'cache';cfg=read(folder/'protocol.json');a,o,y,_=study_module().load_domain(domain)
        a=a.astype(float);o=o.astype(float);a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True));o/=np.maximum(1,np.linalg.norm(o,axis=-1,keepdims=True))
        sch=np.load(folder/'results'/f'schedule_{j["case"]}_p{pool:02d}.npz');ids=sch['pool_ids'];a,o,y=a[ids],o[ids],y[ids];b=(1-alpha)*a+alpha*o
        seed=cfg['query_seed']+['text','image','backbone'].index(domain)*100000+10000*pool
        refs=[r for r in read(folder/'results/records.json') if r['case']==j['case'] and r['pool']==pool and r['rep']==0 and r['g']==j['g']]
        plan=list(zip(sch['train'],sch['test']));final=sch['final'][0]
    elif family=='fivefold':
        a,o,y,_=study_module().load_domain('text');a=a.astype(float);o=o.astype(float)
        a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True));o/=np.maximum(1,np.linalg.norm(o,axis=-1,keepdims=True))
        sch=np.load(SHARED/'logistic_cv_earlystop_kfold_20261030/results'/f'schedule_{j["scenario"]}_{pool}.npz');ids=sch['pool_ids'];a,o,y=a[ids],o[ids],y[ids];b=(1-alpha)*a+alpha*o
        seed=20261030901+100000*j['si']+10000*pool+1000*j['ci']+10000000
        refs=[r for r in read(CP/'fivefold/kfold_results/records.json') if r['scenario']==j['scenario'] and r['tag']==j['case'] and r['group']==pool and r['rep']==0]
        plan=list(zip(sch['train'],sch['test']));final=sch['final'][0]
    else:
        folder=SIM if family=='simulation' else CP/'simulation';cfg=read(folder/'protocol.json')
        data=np.load(folder/'results'/f'pool_{pool:03d}.npz');u,v,noise=data['u'],data['v'],data['latent_uniforms'];m=len(u);r=4
        h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]);eye=np.broadcast_to(np.eye(r),(m,r,r))
        a=np.concatenate((u[:,None,:]*h[None,:,:],eye),2)/np.sqrt(2)
        b=np.concatenate((((1-alpha)*u+alpha*v)[:,None,:]*h[None,:,:],eye),2)/np.sqrt(2)
        theta=j['signal']*np.array([1,.5,-.5,.25,0,0,0,0]);y=(noise<=expit(np.einsum('ird,d->ir',a,theta))).astype(float)
        plan=list(zip(data['train'],data['test']));final=data['final'];cell=0
        for al in cfg['alphas']:
            for la in cfg['penalties']:
                for si in cfg['signals']:
                    cell+=1
                    if (al,la,si)==(alpha,j['lam'],j['signal']):match=cell
        seed=cfg['base_seed']+100*pool+12000000+match*101
        refs=[r for r in read(folder/'results'/f'pool_{pool:03d}.json')['records'] if r['name']==j['case']]
    return a,b,y,plan,final,seed,refs

def fixed_dynamic(a,b,plan,lam,g,cache,seed,prepared):
    poly=prepared.polynomial;cdf=prepared.proposal['cdf'] if prepared.proposal else np.array([])
    c=poly['constant'];mass=float(abs(poly['coef']).sum());eps=prepared.epsilon
    anchors=[[np.zeros(a.shape[-1]),np.zeros(a.shape[-1])] for _ in plan]
    L=c-mass-eps-2e-9;U=c+mass+eps+2e-9;choice=e.decision(L,U,g);n=k=0;history=[]
    if choice!='unresolved':return dict(choice=choice,interval=[L,U],draws=n,history=history)
    guard=2e-9+mapping_guard(poly['coef'],cdf,mass);rng=np.random.default_rng(seed)
    for end in e.CHECKS:
        ix=np.searchsorted(cdf,rng.random(end-n),side='right');x=2*cache.query_keys(poly['left'][ix],'comparison')-1
        kr=poly['right'][ix];pair=kr>=0;xr=np.ones(len(ix));xr[pair]=2*cache.query_keys(kr[pair],'comparison')-1
        k+=int(np.sum(np.sign(poly['coef'][ix])*x*xr>0));n=end;p,q=endpoints(k,n,.05/len(e.CHECKS))
        eps=min(eps,e.conditional_bound(a,b,plan,anchors,lam,cache._values))
        L=c+mass*(2*p-1)-eps-guard;U=c+mass*(2*q-1)+eps+guard;choice=e.decision(L,U,g)
        history.append(dict(draws=n,labels=cache.count,epsilon=eps,interval=[L,U],choice=choice))
        if choice!='unresolved':break
    return dict(choice=choice,interval=[L,U],draws=n,history=history)

def worker(j):
    dest=ROOT/'results'/f'fixed_{j["id"]:03d}.json'
    if dest.exists():return j['id']
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs'/f'fixed_{j["id"]:03d}.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:
            start=time.perf_counter();a,b,y,plan,final,seed,refs=load(j);lam=j['lam'];g=j['g'];r=y.shape[1]
            assert len(refs)>0,'No original comparator records'
            prepared=prepare_comparison(a,b,plan,lam);initial_epsilon=prepared.epsilon;rows=[];snapshot_audits=[]
            cache=e.LabelCache(len(a),r,e.ArrayLabelProvider(y))
            trial=fixed_dynamic(a,b,plan,lam,g,cache,seed,prepared)
            row=e.summarize_record(trial,cache,a,b,plan,lam,final,g,dict(method='fixed_zero_local'),0.)
            rows.append(row);del prepared;gc.collect()
            for mode in ['zero','midpoint']:
                cache=e.LabelCache(len(a),r,e.ArrayLabelProvider(y));trial,snapshots=e.run_path(a,b,plan,lam,g,cache,seed,mode,initial_epsilon)
                row=e.summarize_record(trial,cache,a,b,plan,lam,final,g,dict(method='phase_'+mode),0.)
                rows.append(row);snapshot_audits.append((row,snapshots))
            # Full-label fits are only supplied to the retrospective audit.
            _,target=e.exact_fits(a,b,y,plan,lam)
            for row,snapshots in snapshot_audits:row['approximation_errors']=e.audit_snapshots(a,b,y,plan,lam,snapshots,target)
            for row in rows:
                row.update(target=target,acceptable=bool(target<=g if row['choice']=='A' else target>=-g),covered=bool(row['interval'][0]<=target<=row['interval'][1]))
                assert row['total_labels']==sum(row['stage_labels'].values())
            dump(dest,dict(job=j,seed=seed,pool_objects=len(a),attributes=r,train_size=len(plan[0][0]),splits=len(plan),initial_epsilon=initial_epsilon,
                target=target,regime='both' if abs(target)<=g else 'A_only' if target<0 else 'B_only',records=rows,original=refs,seconds=time.perf_counter()-start))
            return j['id']
        except Exception:
            traceback.print_exc();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--job',type=int);p.add_argument('--family');args=p.parse_args()
    alljobs=jobs();dump(ROOT/'fixed_protocol.json',dict(pools=[0,1,2],paths_per_pool=1,methods=['fixed_zero_local','phase_zero','phase_midpoint'],
        source='Current manuscript, including all fixed-plan appendix settings and retained weak-penalty simulations.',
        pairing='Original pool/split/final subset and query seed, path 0. Published comparators reused as frozen records.',
        changes='New conditional local bounds; phase 0 retains original zero-anchor public certificate; reanchoring uses conditional midpoint.',
        confidence='delta=.05 per configuration/method; phase CP uses delta/9. Failure events are recorded, not discarded.',
        scope='All-configuration exploratory screen, three pools per cell, not the original full replication count.',jobs=alljobs))
    pending=[j for j in alljobs if (args.family is None or j['family']==args.family) and (args.job is None or j['id']==args.job) and not (ROOT/'results'/f'fixed_{j["id"]:03d}.json').exists()]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in pending}
        for fu in as_completed(fs):
            try:i=fu.result();status='done'
            except Exception as exc:i=fs[fu]['id'];status='error: '+repr(exc)
            done=len(list((ROOT/'results').glob('fixed_*.json')));dump(ROOT/'fixed_progress.json',dict(done=done,total=len(alljobs),last_job=i,status=status,elapsed=time.perf_counter()-began))
            print(done,'/',len(alljobs),'job',i,status,flush=True)
    print('BATCH COMPLETE',flush=True)

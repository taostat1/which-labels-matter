from pathlib import Path
import os,sys,json,time,hashlib,argparse,contextlib,traceback,gc
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;SHARED=ROOT.parents[1]
sys.path.insert(0,str(SHARED/'logistic_all_scenarios_reanchor_20261114'))
import fixed_jobs as f
e=f.e;np=f.np
from local_core import candidate_geometry,certificate
from whichlabels import prepare_comparison
from whichlabels._intervals import endpoints,mapping_guard

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');tmp.replace(path)

class Whole(e.LabelCache):
    def query_keys(self,keys,stage):
        keys=np.asarray(keys,dtype=int);ids=np.unique(keys//self.r)
        super().query_keys((ids[:,None]*self.r+np.arange(self.r)).ravel(),stage)
        return self._values[keys]

def configs():
    templates=[j for j in f.jobs() if j['pool']==0];out=[]
    for j in templates:
        family=j['family'];pools=20 if family in ['attribution','coverage'] else 8 if family=='cache' else 3 if family=='fivefold' else 100
        for pool in range(pools):
            d=dict(j,pool=pool,id=len(out));out.append(d)
    return out

def data(j):
    a,b,y,plan,_,seed,_=f.load(j);family=j['family'];pool=j['pool']
    if family in ['attribution','coverage']:
        folder=f.CP/'real'/family;refs=[r for r in f.read(folder/'results/records.json') if r['domain']==j['domain'] and r['group']==pool and r['g']==j['g']]
        sch=np.load(folder/'results'/f'schedule_{j["domain"]}_{pool:02d}.npz');finals=sch['final']
        methods=['selective_paired','whole_object_purchases']+(['separate_bound'] if family=='attribution' else [])
    elif family=='cache':
        folder=f.CP/'cache';refs=[r for r in f.read(folder/'results/records.json') if r['case']==j['case'] and r['pool']==pool and r['g']==j['g']]
        sch=np.load(folder/'results'/f'schedule_{j["case"]}_p{pool:02d}.npz');finals=sch['final'];methods=['current']
    elif family=='fivefold':
        refs=[r for r in f.read(f.CP/'fivefold/kfold_results/records.json') if r['scenario']==j['scenario'] and r['tag']==j['case'] and r['group']==pool]
        sch=np.load(SHARED/'logistic_cv_earlystop_kfold_20261030/results'/f'schedule_{j["scenario"]}_{pool}.npz');finals=sch['final'];methods=['paired_moments']
    else:
        folder=f.SIM if family=='simulation' else f.CP/'simulation';refs=[r for r in f.read(folder/'results'/f'pool_{pool:03d}.json')['records'] if r['name']==j['case']]
        sch=np.load(folder/'results'/f'pool_{pool:03d}.npz');finals=sch['final'][None,:];methods=['paired_weighted','paired_uniform','separate_weighted']
    return a,b,y,plan,finals,seed,refs,methods

def bound(a,b,plan,lam,values):
    r=a.shape[1];x=np.where(np.isfinite(values),2*values-1,np.nan);pairs=[];seps=[]
    for tr,te in plan:
        tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
        aa=candidate_geometry(a[tr],a[te],np.zeros(a.shape[-1]),lam,x[tk]);bb=candidate_geometry(b[tr],b[te],np.zeros(b.shape[-1]),lam,x[tk])
        bd=certificate(aa,bb,lam,x[qk])[1];pairs.append(bd['epsilon']);seps.append(bd['separate'])
    return float(np.mean(pairs)),float(np.mean(seps))

def trial(prepared,cache,seed,g,kind,entry,dynamic=True):
    poly=prepared.polynomial;coef=poly['coef'];mass=float(abs(coef).sum());c=float(poly['constant'])
    separate=kind in ['separate_bound','separate_weighted'];uniform=kind=='paired_uniform'
    eps=prepared.separate_epsilon if separate else prepared.epsilon;initial=eps
    cdf=np.arange(1,len(coef)+1)/len(coef) if uniform else prepared.proposal['cdf']
    if not uniform:assert mapping_guard(coef,cdf,mass)<=2e-9
    L=c-mass-eps-2e-9;U=c+mass+eps+2e-9;choice=e.decision(L,U,g) if entry else 'unresolved'
    n=k=0;su=sq=0.;history=[]
    if choice=='unresolved':
        rng=np.random.default_rng(seed)
        for end in e.CHECKS:
            ix=np.searchsorted(cdf,rng.random(end-n),side='right')
            xx=(2*cache.query_keys(poly['left'][ix],'comparison')-1)*(2*cache.query_keys(poly['right'][ix],'comparison')-1)
            k+=int(np.sum(np.sign(coef[ix])*xx>0));n=end
            if uniform:
                v=coef[ix]*len(coef)*xx;su+=float(v.sum());sq+=float(v@v);var=max(0.,(sq-su*su/n)/(n-1))
                log=np.log(4*len(e.CHECKS)/.05);rad=np.sqrt(2*var*log/n)+14*abs(coef).max()*len(coef)*log/(3*(n-1));lo=c+su/n-rad;hi=c+su/n+rad
            else:
                pl,pu=endpoints(k,n,.05/len(e.CHECKS));lo=c+mass*(2*pl-1);hi=c+mass*(2*pu-1)
            choice=e.decision(lo-eps-2e-9,hi+eps+2e-9,g);refined=False
            # Refinement cannot support a decision while the sampling interval
            # itself is unresolved; retain the previous valid certificate then.
            if dynamic and choice=='unresolved' and e.decision(lo-2e-9,hi+2e-9,g)!='unresolved':
                pair,sep=bound(prepared.a,prepared.b,prepared.splits,prepared.penalty,cache._values)
                eps=min(eps,sep if separate else pair);refined=True
            L=lo-eps-2e-9;U=hi+eps+2e-9;choice=e.decision(L,U,g)
            history.append(dict(draws=n,labels=cache.count,epsilon=eps,interval=[L,U],choice=choice,refined=refined,successes=k))
            if choice!='unresolved':break
    return dict(choice=choice,interval=[L,U],draws=n,history=history,epsilon=eps,initial_epsilon=initial)

def worker(j):
    dest=ROOT/'results'/f'job_{j["id"]:04d}.json'
    if dest.exists():return j['id']
    with (ROOT/'logs'/f'job_{j["id"]:04d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            start=time.perf_counter();a,b,y,plan,finals,seed,refs,methods=data(j);prep=prepare_comparison(a,b,plan,j['lam'])
            entry=j['family'] in ['coverage','simulation','boundary']
            rows=[];baselines=[];masks={};target=float(refs[0].get('target',refs[0].get('cv_delta')))
            xx=(2*y-1).ravel();pol=prep.polynomial;surrogate=float(pol['constant']+pol['coef']@(xx[pol['left']]*xx[pol['right']]));actual=abs(target-surrogate)
            for rep,final in enumerate(finals):
                for method in methods:
                    ref=next(r for r in refs if r['method']==method and r.get('rep',0)==rep)
                    # Preserve the older fivefold algorithm's preparation gate.
                    if j['family']=='fivefold' and prep.epsilon>=j['g']:
                        row=dict(ref,total_labels=ref['labels_total'],coverage=ref['covered'],acceptable=ref['cv_acceptable'],legacy_active_gate=True)
                        rows.append(row);baselines.append(ref);continue
                    cls=Whole if method=='whole_object_purchases' else e.LabelCache
                    cache=cls(len(a),y.shape[1],e.ArrayLabelProvider(y));old=trial(prep,cache,seed+rep,j['g'],method,entry,False)
                    oldrow=e.summarize_record(old,cache,a,b,plan,j['lam'],final,j['g'],dict(method=method),0.)
                    assert oldrow['total_labels']==ref['labels_total'],('baseline labels',j,rep,method,oldrow['total_labels'],ref['labels_total'])
                    assert oldrow['choice']==ref['choice'],('baseline choice',j,rep,method)
                    if 'draws' in ref:assert oldrow['draws']==ref['draws'],('baseline draws',j,rep,method,oldrow['draws'],ref['draws'])
                    cache=cls(len(a),y.shape[1],e.ArrayLabelProvider(y));tr=trial(prep,cache,seed+rep,j['g'],method,entry,True)
                    assert all(actual<=h['epsilon']+2e-8 for h in tr['history'])
                    row=e.summarize_record(tr,cache,a,b,plan,j['lam'],final,j['g'],dict(method=method,rep=rep,g=j['g'],seed=seed+rep),0.)
                    assert row['total_labels']<=oldrow['total_labels'],('monotonic labels',j,rep,method)
                    row.update(target=target,actual_error=actual,covered=bool(row['interval'][0]<=target<=row['interval'][1]),acceptable=bool(target<=j['g'] if row['choice']=='A' else target>=-j['g']),baseline_labels=oldrow['total_labels'],saved_labels=oldrow['total_labels']-row['total_labels'])
                    byobj=cache.seen.reshape(len(a),-1).sum(1);row['cache_total']=dict(labels=cache.count,touched_objects=int(sum(byobj>0)),fully_labeled_objects=int(sum(byobj==y.shape[1])),label_count_histogram=np.bincount(byobj,minlength=y.shape[1]+1).tolist())
                    rows.append(row);baselines.append(ref);masks[f'{method}_{rep}']=np.packbits(cache.seen)
                    print(method,rep,row['total_labels'],'old',oldrow['total_labels'],'refinements',sum(h['refined'] for h in row['history']),flush=True)
            np.savez_compressed(dest.with_suffix('.npz'),**masks)
            save(dest,dict(job=j,records=rows,baselines=baselines,original_comparators=[r for r in refs if r['method'] not in methods],target=target,surrogate=surrogate,actual_error=actual,initial_epsilon=prep.epsilon,separate_epsilon=prep.separate_epsilon,seconds=time.perf_counter()-start))
            return j['id']
        except Exception:traceback.print_exc();raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--job',type=int);ap.add_argument('--family');args=ap.parse_args()
    jobs=configs();save(ROOT/'protocol.json',dict(jobs=jobs,delta=.05,checks=e.CHECKS,methods='Original moment methods, with cache-uniform epsilon refinement only. No anchor/proposal changes.',entry='Preserve historical entry/gate per family; do not attribute entry-check additions to local bounds.',lazy='Refine only when the old widened interval is unresolved but the uninflated sampling interval could support a choice; otherwise retain previous epsilon.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pending=[j for j in jobs if (args.job is None or j['id']==args.job) and (args.family is None or j['family']==args.family) and not (ROOT/'results'/f'job_{j["id"]:04d}.json').exists()]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in pending}
        for fu in as_completed(fs):
            try:i=fu.result();status='done'
            except Exception as err:i=fs[fu]['id'];status=repr(err)
            done=len(list((ROOT/'results').glob('job_*.json')));save(ROOT/'progress.json',dict(done=done,total=len(jobs),last=i,status=status));print(done,'/',len(jobs),i,status,flush=True)

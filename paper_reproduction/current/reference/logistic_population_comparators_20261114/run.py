"""Exact-refit baselines for the frozen zero-preparation population study.

Training sets, tolerance, pools and distinct-label accounting match LEC.
AT and AMS allocate eight evaluation-attribute queries per fitted pair.
Both use the pair's mean predicted probability, requiring no extra pilot.
Acquisition rules depend on independently fitted pairs and public test panels,
not on previous query outcomes. Cumulative bounded-mean intervals are valid.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, hashlib, argparse, contextlib, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
SOURCE = SHARED / 'logistic_population_cache_only_20261114'
sys.path.insert(0, str(SHARED / 'logistic_all_scenarios_reanchor_20261114'))
import population_jobs as p
s = p.s
METHODS = ('direct_refits','active_testing','ams')
GROUPS = tuple(range(6,18))
CHECKS = tuple(p.CHECKS)
DELTA = .04
PANEL = 64
QUERIES = 8


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf-8')


def allocation(score, floors):
    """q=max(floors,t*score), with t chosen to normalize each row."""
    assert np.all(floors >= 0) and np.max(floors.sum(1)) <= 1+1e-10
    score = np.maximum(score, 1e-15)
    lo = np.zeros(len(score)); hi = 1 / score.sum(1)
    for _ in range(48):
        mid = (lo+hi)/2
        too_large = np.maximum(floors, mid[:,None]*score).sum(1) > 1
        hi = np.where(too_large, mid, hi)
        lo = np.where(too_large, lo, mid)
    q = np.maximum(floors, ((lo+hi)/2)[:,None]*score)
    q /= q.sum(1)[:,None]
    assert np.max(np.abs(q.sum(1)-1)) < 1e-12
    assert np.min(q-floors) > -1e-12
    return q


def fitted_residual(z, y, theta, lam):
    v = np.einsum('bnrd,bd->bnr', z, theta)
    g = np.einsum('bnrd,bnr->bd', z, expit(v)-y)/(z.shape[1]*z.shape[2]) + lam*theta
    return np.linalg.norm(g, axis=1)


def run_path(ci, group, a, b, y, pools):
    case = s.CASES[ci]; tick=time.perf_counter()
    original = json.loads((SOURCE/'results'/f'case_{ci:02d}_{group:02d}.json').read_text('utf-8'))
    reference = original['records'][0]
    R = reference['refit_range']
    seed = 202611141701 + ci*10000000 + group*10000
    rng_train = np.random.default_rng(seed+6000000)
    rng_test = np.random.default_rng(seed+6100000)
    rng_query = np.random.default_rng(seed+6200000)
    caches = {m:p.Cache(y) for m in METHODS}
    stats = {m:dict(sum=0.,square=0.,n=0,L=-R,U=R,choice=s.decision(-R,R,case['g']),history=[],conflict=False) for m in METHODS}
    active = {m for m in METHODS if stats[m]['choice']=='unresolved'}
    r=y.shape[1]; d=a.shape[-1]; start=np.zeros(d)
    numeric = 2e-9/case['lam']
    worst_residual=0.; max_range_excess=-float('inf'); max_floor_shortfall=0.
    trace=hashlib.sha256()
    for pos in range(0,CHECKS[-1],64):
        if not active:break
        bs=min(64,CHECKS[-1]-pos)
        ids=rng_train.choice(pools['training'],(bs,case['n']))
        panel=rng_test.choice(pools['comparison'],(bs,PANEL))
        uniforms=rng_query.random((bs,QUERIES))
        trace.update(ids.tobytes());trace.update(panel.tobytes());trace.update(uniforms.tobytes())
        for m in sorted(active):
            yy=caches[m].objects(ids,'comparison_training')
        za,zb=a[ids],b[ids]
        ta,_=s.batch_fit(za,yy,case['lam'],start)
        tb,_=s.batch_fit(zb,yy,case['lam'],start)
        res=max(float(fitted_residual(za,yy,ta,case['lam']).max()),float(fitted_residual(zb,yy,tb,case['lam']).max()))
        worst_residual=max(worst_residual,res)
        assert res<1e-9
        va=np.einsum('bprd,bd->bpr',a[panel],ta).reshape(bs,-1)
        vb=np.einsum('bprd,bd->bpr',b[panel],tb).reshape(bs,-1)
        spa=np.logaddexp(0,va);spb=np.logaddexp(0,vb)
        d0=spa-spb;d1=d0-va+vb;w=-(va-vb)/2
        c=((d0+d1)/2).mean(1)
        assert np.max(abs(c)+abs(w).mean(1)-R)<1e-9
        pred=(expit(va)+expit(vb))/2
        scores={'active_testing':.5*(spa+spb-pred*(va+vb)),
                'ams':(1-pred)*abs(d0)+pred*abs(d1)}
        floors=abs(w)/(PANEL*r*np.maximum(R-abs(c[:,None]),1e-15))
        for m in sorted(active):
            if m=='direct_refits':
                yt=caches[m].objects(panel[:,0],'comparison_evaluation')
                values=(d0[:,:r]-yt*(va[:,:r]-vb[:,:r])).mean(1)
            else:
                q=allocation(scores[m],floors)
                max_floor_shortfall=max(max_floor_shortfall,float(np.max(floors-q)))
                cdf=np.cumsum(q,axis=1);cdf[:,-1]=1.
                ix=(cdf[:,None,:]<uniforms[:,:,None]).sum(2)
                bi=np.arange(bs)[:,None]
                obj=panel[bi,ix//r];keys=obj*r+ix%r
                signed=2*caches[m].query(keys,'comparison_evaluation')-1
                values=c+np.mean(w[bi,ix]*signed/(PANEL*r*q[bi,ix]),axis=1)
            excess=float(np.max(abs(values)-R));max_range_excess=max(max_range_excess,excess)
            assert excess<1e-9
            st=stats[m];st['sum']+=float(values.sum());st['square']+=float(values@values);st['n']+=bs
        N=pos+bs
        if N not in CHECKS:continue
        for m in sorted(active):
            st=stats[m]
            lo,hi,h=s.interval(st['sum'],st['square'],st['n'],R,numeric,DELTA,len(CHECKS))
            st['L']=max(st['L'],lo);st['U']=min(st['U'],hi)
            st['conflict']=st['L']>st['U']
            st['choice']='unresolved' if st['conflict'] else s.decision(st['L'],st['U'],case['g'])
            st['history'].append(dict(blocks=N,labels=caches[m].count,mean=st['sum']/N,interval=[st['L'],st['U']],choice=st['choice'],radius=h))
            if st['choice']!='unresolved' or st['conflict']:active.remove(m)
    final_ids=np.random.default_rng(s.SEED+9000000+ci*100000+40000+group).choice(pools['training'],case['n'])
    out=[];gap=reference['audit_gap'];se=reference['audit_se']
    for m in METHODS:
        st=stats[m];cache=caches[m];before=cache.count
        issued=st['choice']!='unresolved';fit_info=None
        if issued:
            yf=cache.objects(final_ids,'final_training')
            _,fit_info=s.fit((a if st['choice']=='A' else b)[final_ids],yf,case['lam'])
        row=dict(method=m,case=case['id'],case_index=ci,group=group,seed=seed,labels=cache.count,
                 stage_labels=cache.stages,final_new_labels=cache.count-before,
                 pilot_labels=0,calibration_labels=0,decision_issued=issued,
                 acceptable_audit_mean=bool(issued and (gap<=case['g'] if st['choice']=='A' else gap>=-case['g'])),
                 covered_audit_mean=bool(st['L']<=gap<=st['U']),audit_gap=gap,audit_se=se,
                 range=R,error_probability=DELTA,final_fit=fit_info,**st)
        assert row['labels']==sum(row['stage_labels'].values())
        out.append(row)
    return dict(case=case,group=group,records=out,seconds=time.perf_counter()-tick,
                validation=dict(max_solver_residual=worst_residual,max_range_excess=max_range_excess,
                                max_floor_shortfall=max_floor_shortfall),shared_fit_stream_sha256=trace.hexdigest())


def worker(job):
    ci,groups=job;case=s.CASES[ci]
    pending=[g for g in groups if not (ROOT/'results'/f'case_{ci:02d}_{g:02d}.json').exists()]
    if not pending:return ci
    with (ROOT/'logs'/f'case_{ci:02d}.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            a,o,y,pools=s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
            for g in pending:
                data=run_path(ci,g,a,b,y,pools)
                save(ROOT/'results'/f'case_{ci:02d}_{g:02d}.json',data)
                print(ci,g,round(data['seconds'],2),[(v['method'],v['labels'],v['choice']) for v in data['records']],flush=True)
        except Exception:
            traceback.print_exc();raise
    return ci


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--group',type=int);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    for name in ('results','logs'): (ROOT/name).mkdir(parents=True,exist_ok=True)
    save(ROOT/'protocol.json',dict(target='expected test risk over iid random training sets from the same declared finite pools',
        cases=s.CASES,groups=list(GROUPS),methods=list(METHODS),checkpoints=list(CHECKS),delta=DELTA,
        panel_size=PANEL,attribute_queries_per_fit=QUERIES,preparation_labels=0,
        surrogate='mean probability of the two fully fitted candidates on the public evaluation panel',
        cache='distinct training, evaluation and final labels; all acquired labels are reusable',
        ci='cumulative empirical Bernstein, intersection across eight prescribed checkpoints',
        LEC='frozen 96-path zero-preparation public-entry policy, total error allocation 0.04',
        design_fixed_before_results=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    cases=range(8) if args.case is None else [args.case]
    groups=GROUPS if args.group is None else [args.group]
    jobs=[(ci,groups) for ci in cases]
    if args.workers==1:
        for job in jobs:print('completed case',worker(job),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(worker,job) for job in jobs]):print('completed case',f.result(),flush=True)


if __name__=='__main__':main()

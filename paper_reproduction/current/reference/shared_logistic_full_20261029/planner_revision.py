"""Independent-pilot planning with finite-pool label reuse.

The Gaussian distribution is a working forecast, not a confidence guarantee.
Decisions use the unchanged finite-sample intervals on fresh comparison data.
"""
from label_study import *
from scipy.special import ndtri
import copy


def forecast_revision(rec, st, pools):
    tick = time.perf_counter()
    case = rec['case']; stats = rec['preparation']['statistics']
    r = 8 if case['domain'] == 'text' else 6
    sizes = np.array([len(pools['training']), len(pools['comparison'])], float)
    def missing(keys):
        keys = np.asarray(keys, dtype=np.int64)
        return np.array([r*len(pools[s])-np.isin(keys//r,pools[s]).sum()
                         for s in ('training','comparison')], float)
    remaining = missing(rec['initial']); initial_remaining = missing(st['initial'])
    ns = np.asarray(CHECKS, float)
    labels = {}
    for m, multiplicity in [('refit', case['n']), ('complete', 2)]:
        survive = np.exp(ns[:,None] * np.array([multiplicity,1])[None,:]
                         * np.log1p(-1/sizes)[None,:])
        labels[m] = remaining[None,:]*(1-survive)
    # A uniform occupancy approximation extrapolates the observed partial-query rate.
    rate = stats['weighted']['new_labels']/stats['weighted']['blocks']
    hazard = rate/max(1., initial_remaining.sum())
    labels['weighted'] = remaining[None,:]*(-np.expm1(-ns[:,None]*hazard))
    mu = stats['refit']['mean']; se = stats['refit']['se']
    grid = mu + max(se,1e-15)*ndtri((np.arange(101)+.5)/101)
    def radii(m, delta):
        t = np.log(4*len(CHECKS)/delta)
        eps = 2e-10/case['lam'] if m=='refit' else st['epsilon']
        bound = st['R'] if m=='refit' else st['M']
        return eps+np.sqrt(2*stats[m]['variance']*t/ns)+14*bound*t/(3*(ns-1))
    def first_indices(rad):
        qualified = rad[None,:] <= case['g']+abs(grid[:,None])
        return np.where(qualified.any(1), qualified.argmax(1), len(ns)-1)
    standalone_ix = first_indices(radii('refit',.04))
    fallback_ix = first_indices(radii('refit',.02))
    fallback_labels = labels['refit'][fallback_ix]
    refit_forecast = float(labels['refit'][standalone_ix].sum(1).mean())
    best = dict(selected='refit', expected_labels=refit_forecast,
                trial_label_cap=None, max_blocks=int(ns[-1]))
    options = []
    if st['accepted']:
        for m in ('weighted','complete'):
            rad = radii(m,.02)
            for j in range(len(ns)):
                end_labels = labels[m][j]
                cap = min(2*end_labels.sum(),remaining.sum())
                if end_labels.sum()>cap+1e-10: continue
                good = rad[None,:j+1] <= case['g']+abs(grid[:,None])
                success = good.any(1); ix = good.argmax(1)
                costs = labels[m][ix].sum(1)
                # Independent uniform occupancy within each pool approximates reuse.
                overlap = end_labels[None,:]*fallback_labels/np.maximum(remaining,1)[None,:]
                costs[~success] = (end_labels[None,:]+fallback_labels-overlap).sum(1)[~success]
                option = dict(selected=m,expected_labels=float(costs.mean()),
                              trial_label_cap=max(1,int(np.ceil(cap))),max_blocks=int(ns[j]),
                              forecast_success=float(success.mean()))
                options.append(option)
                if option['expected_labels']<best['expected_labels']-1e-9: best=dict(option)
    best.update(refit_expected_labels=refit_forecast, options=options,
                pilot_mean=mu,pilot_se=se,remaining_labels=remaining.tolist(),
                planning_seconds=time.perf_counter()-tick)
    return best


def revised_deferred(rec, plan):
    m=plan['selected']
    if m=='refit':
        ans=static(rec,m,'adaptive')
        ans.update(method='revised',switched=False,selected=m)
        ans['seconds']+=plan['planning_seconds']; return ans
    cap=plan['trial_label_cap'];keys=set(rec['initial']);base=len(keys)
    secs=rec['preparation']['seconds']+plan['planning_seconds']
    checks={(v['method'],v['blocks']):v for v in rec['checkpoints']}
    row=None;spent=0;maxbatch=0
    for batch in rec['batches']:
        v=batch['methods'].get(m)
        if v is None: continue
        keys.update(v['new']);secs+=v['seconds'];spent=len(keys)-base
        maxbatch=max(maxbatch,len(v['new']))
        row=checks.get((m,batch['blocks']))
        if row is not None and row['protected']['decision']!='unresolved':break
        if spent>=cap or batch['blocks']>=plan['max_blocks']:break
    switched = row is None or row['protected']['decision']=='unresolved'
    if switched:
        assert spent<=cap+maxbatch
        row=first_row(rec,'refit','protected')
        for batch in rec['batches']:
            if row is None or batch['blocks']>row['blocks']:break
            v=batch['methods'].get('refit')
            if v is not None:keys.update(v['new']);secs+=v['seconds']
        standalone=0 if row is None else row['labels']-base
        assert len(keys)-base<=cap+maxbatch+standalone
    return finish(rec,keys,secs,row,'protected','revised',
                  dict(selected=m,switched=switched,trial_labels=spent,trial_cap=cap,
                       max_blocks=plan['max_blocks']))


def evaluate(rec,st,pools,input_seconds,audit):
    old,old_plan=analyze_record(rec,st,pools,input_seconds,audit)
    plan=forecast_revision(rec,st,pools);ans=revised_deferred(rec,plan)
    ans['seconds']+=input_seconds;ans['input_setup_seconds']=input_seconds
    eligible=[static(rec,m,'protected') for m in METHODS]
    eligible=[v for v in eligible if v['decision']!='unresolved']
    best=min(eligible,key=lambda v:v['labels'])['method'] if eligible else 'none'
    ans.update(cheapest_protected=best,plan_correct_protected=plan['selected']==best)
    assess(ans,audit,rec['case']['g'])
    return old+[ans],plan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true')
    ap.add_argument('--timing',action='store_true');ap.add_argument('--case',type=int)
    ap.add_argument('--group',type=int)
    args=ap.parse_args();inputs=json.loads((ROOT/'input_costs.json').read_text('utf-8'))['cases']
    audits=json.loads((ROOT/'audit_results.json').read_text('utf-8'))['cases']
    dest=ROOT/('planner_final_timing' if args.timing else 'planner_final_checked');dest.mkdir(exist_ok=True)
    rows=[];plans=[]
    for ci,case in enumerate(CASES):
        if args.case is not None and args.case!=ci: continue
        a,o,y,pools=load_domain(case['domain'])
        sts={g:json.loads((ROOT/'results'/f"setup_{case['id']}_{g}.json").read_text('utf-8')) for g in range(4)}
        if args.replay:
            paths=sorted((ROOT/'label_results').glob(case['id']+'_[0-9][0-9][0-9].json.gz'))
            for p in paths:
                with gzip.open(p,'rt',encoding='utf-8') as f: rec=json.load(f)
                rr,pl=evaluate(rec,sts[rec['group']],pools,inputs[case['id']]['seconds'],audits[case['id']])
                rows.extend(rr);plans.append(dict(case=case['id'],rep=rec['rep'],**pl))
        else:
            reps=1 if args.timing else 20
            for group in range(1 if args.timing else 4):
                if args.group is not None and group!=args.group:continue
                indices=range(group,reps,1 if args.timing else 4)
                if all((dest/f"{case['id']}_{i:03d}.json.gz").exists() for i in indices):continue
                b,ts,tcs,dc,now=setup(case,a,o,y,pools,group);st=sts[group]
                for key in ('epsilon','M','R'): assert abs(now[key]-st[key])<1e-12
                for i in indices:
                    path=dest/f"{case['id']}_{i:03d}.json.gz"
                    if path.exists():continue
                    tick=time.perf_counter();rep=(39000 if args.timing else 40000)+i
                    rec=run_trial(case,dict(planning_blocks=64,batch=128,checkpoints=CHECKS),rep,a,b,y,pools,ts,tcs,dc,st)
                    rr,pl=evaluate(rec,st,pools,inputs[case['id']]['seconds'],audits[case['id']])
                    rec['revision_analysis']=rr;rec['revision_plan']=pl;rec['new_index']=i
                    tmp=path.with_suffix('.partial')
                    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(rec,f,separators=(',',':'))
                    tmp.replace(path);done=len(list(dest.glob('*.json.gz')))
                    progress('先导选路修订：最终独立验证',done,1 if args.timing else 160,
                             f"{case['id']}，第 {i+1} 条路径；本次 {time.perf_counter()-tick:.1f} 秒")
                    print(case['id'],i,round(time.perf_counter()-tick,2),[(v['method'],v['labels'],v['decision']) for v in rr if v['method'] in ('deferred','revised','refit_direct')],flush=True)
        print('DONE',case['id'],flush=True)
    if args.replay:
        save(ROOT/f"planner_development_{args.case if args.case is not None else 'all'}.json",dict(rows=rows,plans=plans))
    elif not args.timing:
        rows=[];plans=[]
        for p in sorted(dest.glob('*.json.gz')):
            with gzip.open(p,'rt',encoding='utf-8') as f: rec=json.load(f)
            rows.extend(rec['revision_analysis']);plans.append(dict(case=rec['case']['id'],rep=rec['rep'],**rec['revision_plan']))
        save(ROOT/'planner_final_summary.json',dict(rows=rows,plans=plans,trajectories=len(plans)))


if __name__=='__main__':main()

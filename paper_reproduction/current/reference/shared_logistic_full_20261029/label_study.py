"""New, independently seeded paths for label-count planning and deferred refits.

All prices are absent. Original observations and preparations remain immutable.
The adaptation sees planning data and already revealed prefixes only.
"""
from core import *
from study import CASES, METHODS, load_domain, setup, run_trial
from label_metrics import first_row, assess, adaptive_refit_intervals
import gzip, argparse

CHECKS=[256,512,1024,2048,4096,8192,16384,32768]


def label_forecast(case,st,stats):
    h=case['g']+max(0.,abs(stats['refit']['mean'])-stats['refit']['se'])
    plans={}
    for m in METHODS:
        eps=0. if m=='refit' else st['epsilon'];M=st['R'] if m=='refit' else st['M']
        t=np.log(4*len(CHECKS)/.02);v=stats[m]['variance'];blocks=None
        if m=='refit' or st['accepted']:
            for N in CHECKS:
                if eps+np.sqrt(2*v*t/N)+14*M*t/(3*(N-1))<=h:
                    blocks=N;break
        labels=None if blocks is None else blocks*stats[m]['new_labels']/stats[m]['blocks']
        plans[m]=dict(blocks=blocks,labels=labels)
    chosen=min(METHODS,key=lambda m:float('inf') if plans[m]['labels'] is None else plans[m]['labels'])
    if plans[chosen]['labels'] is None:chosen='refit'
    cap=None
    if chosen!='refit':
        cap=2*plans[chosen]['labels']
        if plans['refit']['labels'] is not None:cap=min(cap,plans['refit']['labels']/4)
        cap=max(1,int(np.ceil(cap)))
    return dict(selected=chosen,plans=plans,radius=h,trial_label_cap=cap)


def finish(rec,keys,seconds,row,mode,method,extra=None):
    choice='unresolved' if row is None else row[mode]['decision'];before=len(keys)
    if choice!='unresolved':keys.update(rec['final_keys']);seconds+=rec['final_fits'][choice]['seconds']
    r=8 if rec['case']['domain']=='text' else 6
    out=dict(case=rec['case']['id'],rep=rec['rep'],group=rec['group'],method=method,decision=choice,
             labels=len(keys),documents=len({k//r for k in keys}),seconds=seconds,
             final_new_labels=len(keys)-before,lower=None if row is None else row[mode]['lower'],
             upper=None if row is None else row[mode]['upper'],both=False if row is None else row[mode]['both'])
    if extra:out.update(extra)
    return out


def static(rec,m,mode='single'):
    row=first_row(rec,m,mode);keys=set(rec['initial']);secs=rec['preparation']['seconds']
    if row is not None:
        for batch in rec['batches']:
            if batch['blocks']>row['blocks']:break
            if m in batch['methods']:
                v=batch['methods'][m];keys.update(v['new']);secs+=v['seconds']
    return finish(rec,keys,secs,row,mode,m)


def deferred(rec,plan):
    m=plan['selected']
    if m=='refit':
        ans=static(rec,m,'adaptive');ans.update(method='deferred',switched=False,selected=m);return ans
    cap=plan['trial_label_cap'];keys=set(rec['initial']);base=len(keys);secs=rec['preparation']['seconds']
    checks={(v['method'],v['blocks']):v for v in rec['checkpoints']};row=None;switched=False;spent=0;maxbatch=0
    for batch in rec['batches']:
        v=batch['methods'].get(m)
        if v is None:continue
        keys.update(v['new']);secs+=v['seconds'];spent=len(keys)-base;maxbatch=max(maxbatch,len(v['new']))
        row=checks.get((m,batch['blocks']))
        if row is not None and row['protected']['decision']!='unresolved':break
        if spent>=cap:switched=True;break
    else:switched=True
    if row is None or row['protected']['decision']=='unresolved':switched=True
    if switched:
        assert spent<=cap+maxbatch
        row=first_row(rec,'refit','protected')
        for batch in rec['batches']:
            if row is None or batch['blocks']>row['blocks']:break
            v=batch['methods'].get('refit')
            if v is not None:keys.update(v['new']);secs+=v['seconds']
        standalone=0 if row is None else row['labels']-base
        assert len(keys)-base<=cap+maxbatch+standalone
    return finish(rec,keys,secs,row,'protected','deferred',dict(selected=m,switched=switched,trial_labels=spent,trial_cap=cap))


def parallel(rec,plan):
    m=plan['selected']
    if m=='refit':
        ans=static(rec,m,'adaptive');ans.update(method='parallel',selected=m);return ans
    paths=[m,'refit'];w={m:.75,'refit':.25};spend={p:0 for p in paths};ix={p:0 for p in paths}
    streams={p:[(b['blocks'],b['methods'][p]) for b in rec['batches'] if p in b['methods']] for p in paths}
    checks={(v['method'],v['blocks']):v for v in rec['checkpoints']}
    keys=set(rec['initial']);secs=rec['preparation']['seconds'];row=None
    while any(ix[p]<len(streams[p]) for p in paths):
        p=min([p for p in paths if ix[p]<len(streams[p])],key=lambda p:spend[p]/w[p])
        blocks,v=streams[p][ix[p]];ix[p]+=1;spend[p]+=len(v['new']);keys.update(v['new']);secs+=v['seconds']
        row=checks.get((p,blocks))
        if row is not None and row['protected']['decision']!='unresolved':break
    else:row=None
    return finish(rec,keys,secs,row,'protected','parallel',dict(selected=m))


def analyze_record(rec,st,pools,input_seconds,audit):
    from label_metrics import refit_requests
    adaptive_refit_intervals(rec)
    plan=label_forecast(rec['case'],st,rec['preparation']['statistics'])
    results=[static(rec,m) for m in METHODS]
    chosen=static(rec,'refit','adaptive') if plan['selected']=='refit' else next(v for v in results if v['method']==plan['selected'])
    single=dict(chosen,method='pilot_only',selected=plan['selected'])
    results.extend([single,deferred(rec,plan),parallel(rec,plan)])
    r=8 if rec['case']['domain']=='text' else 6
    keys,row=refit_requests(rec,pools,r,st)
    direct=finish(rec,keys,st['anchor_seconds']+row['seconds'],row,'single','refit_direct')
    results.append(direct)
    eligible=[v for v in results[:3] if v['decision']!='unresolved']
    cheapest=min(eligible,key=lambda v:v['labels'])['method'] if eligible else 'none'
    for out in results:
        out['seconds']+=input_seconds;out['input_setup_seconds']=input_seconds
        out['selected']=plan['selected'];out['plan_correct']=plan['selected']==cheapest;out['cheapest_fixed']=cheapest
        assess(out,audit,rec['case']['g'])
    return results,plan


def progress_update(stage,n,total,note):
    d=dict(stage=stage,completed=n,total=total,percent=100*n/total,note=note,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'label_optimization_progress.json',d)
    (ROOT/'标签优化进程.txt').write_text(f"阶段：{stage}\n进度：{n}/{total}（{100*n/total:.1f}%）\n{note}\n更新时间：{d['updated']}\n",encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--timing',action='store_true');ap.add_argument('--replay',action='store_true');ap.add_argument('--case',type=int);args=ap.parse_args()
    if args.replay:
        audits=json.loads((ROOT/'audit_results.json').read_text('utf-8'))['cases']
        inputs=json.loads((ROOT/'input_costs.json').read_text('utf-8'))['cases'];allrows=[]
        for case in CASES:
            _,_,y,pools=load_domain(case['domain'])
            sts={g:json.loads((ROOT/'results'/f"setup_{case['id']}_{g}.json").read_text('utf-8')) for g in range(4)}
            for path in sorted((ROOT/'results').glob(case['id']+'_[0-9][0-9][0-9].json.gz')):
                with gzip.open(path,'rt',encoding='utf-8') as f:rec=json.load(f)
                rows,plan=analyze_record(rec,sts[rec['group']],pools,inputs[case['id']]['seconds'],audits[case['id']])
                allrows.extend(rows)
            print('LABEL REPLAY',case['id'],len(allrows)//7,flush=True)
        save(ROOT/'label_replay_summary.json',dict(rows=allrows,trajectories=len(allrows)//7,methods_per_trajectory=7,
             description='Label-only analysis of original paths; algorithm revision was developed after these observations.'))
        return
    reps=1 if args.timing else 20;groups=1 if args.timing else 4
    dest=ROOT/('label_timing' if args.timing else 'label_results');dest.mkdir(exist_ok=True)
    cfg=dict(planning_blocks=64,batch=128,checkpoints=CHECKS)
    audit=json.loads((ROOT/'audit_results.json').read_text('utf-8'))['cases']
    inputs=json.loads((ROOT/'input_costs.json').read_text('utf-8'))['cases']
    if not args.timing:
        save(ROOT/'label_protocol.json',dict(repetitions=20,groups=4,conditional_paths_per_group=5,rep_offset=10000,
             preparation='Four existing independent preparations; charged fully; all subsequent planning and comparison draws are new.',
             forecast='Expected distinct new labels per planning block times forecast blocks, at sampling error .02.',
             trial_cap='Minimum of twice selected-plan label forecast and one quarter refit label forecast, rounded upward.',
             error_budgets=dict(calibration=.01,single_moment=.04,direct_refit=.05,pilot_selected_refit=.04,deferred_moment=.02,deferred_refit=.02),
             primary_resource='Distinct object-attribute labels including preparation and final fitting.',**cfg))
    for ci,case in enumerate(CASES):
        if args.case is not None and ci!=args.case:continue
        if all((dest/f"{case['id']}_{rep:03d}.json.gz").exists() for rep in range(reps)):continue
        a,o,y,pools=load_domain(case['domain'])
        for group in range(groups):
            if all((dest/f"{case['id']}_{rep:03d}.json.gz").exists() for rep in range(group,reps,groups)):continue
            b,ts,tcs,dc,now=setup(case,a,o,y,pools,group)
            st=json.loads((ROOT/'results'/f"setup_{case['id']}_{group}.json").read_text('utf-8'))
            for key in ['epsilon','M','R']:assert abs(now[key]-st[key])<1e-12
            for rep in range(group,reps,groups):
                path=dest/f"{case['id']}_{rep:03d}.json.gz"
                if path.exists():continue
                tick=time.perf_counter();newrep=(9000 if args.timing else 10000)+rep
                rec=run_trial(case,cfg,newrep,a,b,y,pools,ts,tcs,dc,st)
                results,plan=analyze_record(rec,st,pools,inputs[case['id']]['seconds'],audit[case['id']])
                # Replace the obsolete stored forecast; only this label forecast is used.
                rec['preparation']['plan']=plan;rec['label_analysis']=results;rec['new_index']=rep
                tmp=path.with_suffix('.partial')
                with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(rec,f,separators=(',',':'))
                tmp.replace(path);done=len(list(dest.glob('*.json.gz')))
                progress_update('小规模计时' if args.timing else '独立标签验证',done,8*reps,f"{case['id']}，准备组 {group+1}；本次 {time.perf_counter()-tick:.1f} 秒")
                print(case['id'],rep,round(time.perf_counter()-tick,2),[(v['method'],v['labels'],v['decision']) for v in results if v['method'] in ['pilot_only','deferred','refit_direct']],flush=True)
    if not args.timing:
        rows=[]
        for p in sorted(dest.glob('*.json.gz')):
            with gzip.open(p,'rt',encoding='utf-8') as f:rows.extend(json.load(f)['label_analysis'])
        save(ROOT/'label_results_summary.json',dict(rows=rows,trajectories=len(rows)//7,methods_per_trajectory=7))


if __name__=='__main__':main()

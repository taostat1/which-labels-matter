"""Prefix-only cost replay; audit risks are read only after all choices are fixed."""
from core import *
from study import CASES, METHODS, SEED, load_domain, forecast
import gzip, argparse


def first_row(rec, method, mode='single'):
    rows=[v for v in rec['checkpoints'] if v['method']==method]
    if not rows:return None
    return next((v for v in rows if v[mode]['decision']!='unresolved'),rows[-1])


def finish(rec,keys,seconds,row,mode,reading,r,method,blocks):
    choice='unresolved' if row is None else row[mode]['decision']
    before_cost=seconds+.01*(len(keys)+reading*len({k//r for k in keys}))
    nfinal=0
    if choice!='unresolved':
        before=len(keys);keys.update(rec['final_keys']);nfinal=len(keys)-before
        seconds+=rec['final_fits'][choice]['seconds']
    docs=len({k//r for k in keys});cost=seconds+.01*(len(keys)+reading*docs)
    return dict(method=method,decision=choice,both=False if row is None else bool(row[mode]['both']),
                blocks=blocks,labels=len(keys),documents=docs,seconds=seconds,cost=cost,
                annotation_cost=.01*(len(keys)+reading*docs),
                final_cost=cost-before_cost,
                final_new_labels=nfinal,lower=None if row is None else row[mode]['lower'],
                upper=None if row is None else row[mode]['upper'])


def fixed(rec,method,reading,r,mode='single'):
    row=first_row(rec,method,mode);keys=set(rec['initial']);seconds=rec['preparation']['seconds']
    if row is not None:
        for batch in rec['batches']:
            if batch['blocks']>row['blocks']:break
            v=batch['methods'].get(method)
            if v:keys.update(v['new']);seconds+=v['seconds']
    return finish(rec,keys,seconds,row,mode,reading,r,method,0 if row is None else row['blocks'])


def protected(rec,reading,r):
    selected=rec['preparation']['plan']['selected']
    if selected=='refit':
        out=fixed(rec,'refit',reading,r);out['method']='protected';out['winner_path']='refit';return out
    paths=(selected,'refit');share={selected:.75,'refit':.25}
    streams={p:[(b['blocks'],b['methods'][p]) for b in rec['batches'] if p in b['methods']] for p in paths}
    indices=dict.fromkeys(paths,0);spend=dict.fromkeys(paths,0.);maxbatch=dict.fromkeys(paths,0.)
    lastdocs=dict.fromkeys(paths,rec['preparation']['documents'])
    checks={(v['method'],v['blocks']):v for v in rec['checkpoints']}
    keys=set(rec['initial']);seconds=rec['preparation']['seconds'];row=None;winner=None
    while any(indices[p]<len(streams[p]) for p in paths):
        active=[p for p in paths if indices[p]<len(streams[p])]
        p=min(active,key=lambda k:spend[k]/share[k]);blocks,v=streams[p][indices[p]];indices[p]+=1
        inc=v['seconds']+.01*(len(v['new'])+reading*(v['documents']-lastdocs[p]))
        lastdocs[p]=v['documents'];spend[p]+=inc;maxbatch[p]=max(maxbatch[p],inc)
        keys.update(v['new']);seconds+=v['seconds']
        row=checks.get((p,blocks))
        if row is not None and row['protected']['decision']!='unresolved':winner=p;break
    if winner is None:row=None
    out=finish(rec,keys,seconds,row,'protected',reading,r,'protected',sum(indices.values())*128)
    out['winner_path']=winner;out['path_batches']=indices;out['virtual_spending']=spend
    # Deterministic theorem audit, before the common preparation and final fit.
    bounds=[]
    for p in paths:
        cp=first_row(rec,p,'protected')
        if cp is None or cp['protected']['decision']=='unresolved':continue
        standalone=cp['seconds']+.01*((cp['labels']-rec['preparation']['labels'])+reading*(cp['documents']-rec['preparation']['documents']))
        bounds.append(standalone/share[p])
    out['scheduling_lhs']=sum(spend.values())
    out['scheduling_rhs']=None if not bounds else min(bounds)+sum(maxbatch.values())
    if bounds:assert out['scheduling_lhs']<=out['scheduling_rhs']+1e-7
    return out


def refit_requests(rec,pools,r,st):
    """An equally informed refit baseline buys no moment calibration/planning."""
    case=rec['case'];ci=next(i for i,c in enumerate(CASES) if c['id']==case['id'])
    row=first_row(rec,'refit');rng=np.random.default_rng(SEED+7000000+ci*100000+rec['rep'])
    rng.choice(pools['training'],(64,case['n']));rng.choice(pools['comparison'],64);rng.random(64)
    seen=np.zeros(max(int(pools['training'].max()),int(pools['comparison'].max()))+1,bool)
    seen[np.asarray(st['pilot'])]=True
    for start in range(0,row['blocks'],128):
        ids=rng.choice(pools['training'],(128,case['n']));test=rng.choice(pools['comparison'],128);rng.random(128)
        seen[ids.ravel()]=True;seen[test]=True
    docs=np.flatnonzero(seen)
    keys=set((docs[:,None]*r+np.arange(r)).ravel().tolist())
    return keys,row


def assess(out,audit,g):
    if audit is None:return out
    delta=audit['difference']['mean'];se=audit['difference']['se']
    lo,hi=delta-1.96*se,delta+1.96*se
    out['audit_gap']=delta;out['audit_se']=se
    out['covered_audit_mean']=out['lower'] is not None and out['lower']<=delta<=out['upper']
    out['excludes_audit_band']=out['lower'] is not None and (out['lower']>hi or out['upper']<lo)
    out['unacceptable_audit_mean']=(out['decision']=='A' and delta>g) or (out['decision']=='B' and delta<-g)
    out['unacceptable_audit_band']=(out['decision']=='A' and lo>g) or (out['decision']=='B' and hi<-g)
    return out


def summarize(rows):
    ans=[]
    for ci,case in enumerate(CASES):
        for reading in (0,5,20):
            for method in ('weighted','complete','refit_common','pilot_single','protected','refit_direct'):
                rr=[v for v in rows if v['case']==case['id'] and v['reading']==reading and v['method']==method]
                if not rr:continue
                dec=[v for v in rr if v['decision']!='unresolved']
                item=dict(case=case['id'],reading=reading,method=method,reps=len(rr),decisions=len(dec),both=sum(v['both'] for v in rr))
                for key in ('cost','labels','documents','seconds','blocks'):
                    a=np.array([v[key] for v in rr]);item[key+'_median']=float(np.median(a));item[key+'_mean']=float(a.mean())
                for key in ('unacceptable_audit_mean','unacceptable_audit_band','excludes_audit_band'):
                    item[key]=sum(v.get(key,False) for v in dec)
                ans.append(item)
    return ans


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--available',action='store_true');args=ap.parse_args()
    auditfile=ROOT/'audit_results.json';audits=json.loads(auditfile.read_text('utf-8'))['cases'] if auditfile.exists() else {}
    costfile=ROOT/'input_costs.json';inputcost=json.loads(costfile.read_text('utf-8'))['cases'] if costfile.exists() else {}
    if not args.partial:assert all(c['id'] in inputcost for c in CASES)
    rows=[];planrows=[];domain=None
    cachepath=ROOT/'analysis_available.json'
    cache=json.loads(cachepath.read_text('utf-8')) if cachepath.exists() and not (args.partial or args.available) else dict(rows=[],planning=[])
    for case in CASES:
        files=sorted((ROOT/'results').glob(case['id']+'_[0-9][0-9][0-9].json.gz'))
        if not files:continue
        if args.available and len(files)!=100:continue
        cached=[v for v in cache['rows'] if v['case']==case['id']]
        cp=[v for v in cache['planning'] if v['case']==case['id']]
        expected={(m,p,i) for m in ('weighted','complete','refit_common','pilot_single','protected','refit_direct') for p in (0,5,20) for i in range(100)}
        if len(files)==100 and len(cached)==1800 and len(cp)==100 and {(v['method'],v['reading'],v['rep']) for v in cached}==expected:
            common=inputcost[case['id']]['seconds']
            assert all(abs(v['input_setup_seconds']-common)<1e-12 for v in cached)
            rows.extend(assess(dict(v),audits.get(case['id']),case['g']) for v in cached);planrows.extend(cp)
            print(case['id'],'100 previously analyzed immutable records retained; risk audit refreshed',flush=True)
            continue
        if case['domain']!=domain:
            _,_,y,pools=load_domain(case['domain']);r=y.shape[1];domain=case['domain']
        setups={g:json.loads((ROOT/'results'/f"setup_{case['id']}_{g}.json").read_text('utf-8')) for g in range(4) if (ROOT/'results'/f"setup_{case['id']}_{g}.json").exists()}
        for path in files:
            with gzip.open(path,'rt',encoding='utf-8') as f:rec=json.load(f)
            directkeys,directrow=refit_requests(rec,pools,r,setups[rec['group']]) if not args.partial else (None,None)
            for reading in (0,5,20):
                rec['preparation']['plan']=forecast(case,setups[rec['group']],rec['preparation']['statistics'],len(rec['initial']),reading=reading)
                out={m:fixed(rec,m,reading,r) for m in METHODS}
                selected=rec['preparation']['plan']['selected'];out['pilot_single']=dict(out[selected],method='pilot_single')
                out['protected']=protected(rec,reading,r);out['refit_common']=out.pop('refit');out['refit_common']['method']='refit_common'
                if directrow is not None:
                    # Public feature bounds are granted free to this strong baseline.
                    seconds=setups[rec['group']]['anchor_seconds']+directrow['seconds']
                    out['refit_direct']=finish(rec,set(directkeys),seconds,directrow,'single',reading,r,'refit_direct',directrow['blocks'])
                for v in out.values():
                    v.update(case=case['id'],rep=rec['rep'],group=rec['group'],reading=reading,selected_plan=selected)
                    if v['method']=='refit_direct':
                        count=len(set(setups[rec['group']]['pilot']))
                        v['preparation_cost']=setups[rec['group']]['anchor_seconds']+.01*count*(r+reading)
                    else:
                        p=rec['preparation'];v['preparation_cost']=p['seconds']+.01*(p['labels']+reading*p['documents'])
                    common=inputcost.get(case['id'],{}).get('seconds',0.)
                    v['preparation_cost']+=common;v['seconds']+=common;v['cost']+=common;v['input_setup_seconds']=common
                    rows.append(assess(v,audits.get(case['id']),case['g']))
                if reading==5:
                    candidates={k:v for k,v in out.items() if k in ('weighted','complete','refit_common') and v['decision']!='unresolved'}
                    cheapest=min(candidates,key=lambda k:candidates[k]['cost']) if candidates else 'none'
                    selectedname='refit_common' if selected=='refit' else selected
                    planrows.append(dict(case=case['id'],rep=rec['rep'],group=rec['group'],selected=selected,cheapest=cheapest,
                        correct=selectedname==cheapest,selected_decides=out['pilot_single']['decision']!='unresolved'))
        print(case['id'],len(files),'analyzed',flush=True)
    dest='analysis_partial.json' if args.partial else ('analysis_available.json' if args.available else 'analysis_results.json')
    save(ROOT/dest,dict(rows=rows,summary=summarize(rows),planning=planrows,
        cost_convention='seconds + 0.01*(unique labels + reading ratio*unique objects); preparation and final fitting included'))


if __name__=='__main__':main()

"""Focused batch-size ablation on the same paid pilots and independent data prefixes."""
from study import *
from analyze import fixed,finish,protected
import gzip


def run(rec,a,b,y,pools,ts,st):
    selected=rec['preparation']['plan']['selected'];r=y.shape[1];case=rec['case'];reading=5
    if selected=='refit':
        ans=fixed(rec,'refit',reading,r);ans['selected_plan']=selected;return ans
    ci=next(i for i,c in enumerate(CASES) if c['id']==case['id'])
    rng=np.random.default_rng(SEED+7000000+ci*100000+rec['rep'])
    rng.choice(pools['training'],(64,case['n']));rng.choice(pools['comparison'],64);rng.random(64)
    feature_chunks=[]
    def feature_part(start):
        chunk=start//128;offset=start%128
        while len(feature_chunks)<=chunk:
            ii=rng.choice(pools['training'],(128,case['n']));tt=rng.choice(pools['comparison'],128);rng.random(128)
            feature_chunks.append((ii,tt))
        ii,tt=feature_chunks[chunk];return ii[offset:offset+8],tt[offset:offset+8]
    moments=[(v['blocks'],v['methods'][selected]) for v in rec['batches'] if selected in v['methods']]
    checks={v['blocks']:v for v in rec['checkpoints'] if v['method']==selected}
    cp=[256,512,1024,2048,4096,8192,16384,32768]
    shares={selected:.75,'refit':.25};spend={selected:0.,'refit':0.};qmax={selected:0.,'refit':0.}
    keys=set(rec['initial']);seconds=rec['preparation']['seconds'];ledger=Ledger(y,rec['initial'])
    olddocs=dict.fromkeys(shares,rec['preparation']['documents']);mi=0;rn=0;su=sq=0.;row=None;winner=None
    while mi<len(moments) or rn<32768:
        active=([selected] if mi<len(moments) else [])+(['refit'] if rn<32768 else [])
        path=min(active,key=lambda p:spend[p]/shares[p])
        if path==selected:
            blocks,v=moments[mi];mi+=1;new=v['new'];sec=v['seconds'];docs=v['documents'];row=checks.get(blocks)
        else:
            ids,tests=feature_part(rn);before=ledger.seen.copy()
            values,work,_=batch_values(ids,tests,None,a,b,y,ts,None,None,case,st,['refit'],{'refit':ledger})
            values=values['refit'];su+=float(values.sum());sq+=float(values@values);rn+=8
            new=np.flatnonzero(ledger.seen&~before).tolist();_,docs=ledger.counts();sec=work['refit'];row=None
            if rn in cp:
                reference=next((v for v in rec['checkpoints'] if v['method']=='refit' and v['blocks']==rn),None)
                if reference is not None:assert abs(su/rn-reference['mean'])<1e-10
                lo,hi,rad=interval(su,sq,rn,st['R'],2e-10/case['lam'],.02,8)
                dd=decision(lo,hi,case['g']);row=dict(method='refit',blocks=rn,protected=dict(lower=float(lo),upper=float(hi),radius=float(rad),decision=dd,both=bool(lo>=-case['g'] and hi<=case['g'])))
        inc=sec+.01*(len(new)+reading*(docs-olddocs[path]));olddocs[path]=docs
        spend[path]+=inc;qmax[path]=max(qmax[path],inc);seconds+=sec;keys.update(new)
        if row is not None and row['protected']['decision']!='unresolved':winner=path;break
    if winner is None:row=None
    ans=finish(rec,keys,seconds,row,'protected',reading,r,'small_batch',mi*128+rn)
    ans.update(selected_plan=selected,winner_path=winner,refit_blocks=rn,moment_blocks=mi*128,virtual_spend=spend,max_batches=qmax)
    return ans


def main():
    dest=ROOT/'small_batch_results.json'
    allrows=json.loads(dest.read_text('utf-8'))['rows'] if dest.exists() else []
    for ci,case in enumerate(CASES):
        if sum(v['case']==case['id'] for v in allrows)==100:continue
        if len(list((ROOT/'results').glob(case['id']+'_[0-9][0-9][0-9].json.gz')))<100:continue
        a,o,y,pools=load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        preparations={}
        for g in range(4):
            st=json.loads((ROOT/'results'/f"setup_{case['id']}_{g}.json").read_text('utf-8'));pilot=np.asarray(st['pilot'])
            preparations[g]=st,[fit(z[pilot],y[pilot],case['lam'])[0] for z in [a,b]]
        for rep in range(100):
            with gzip.open(ROOT/'results'/f"{case['id']}_{rep:03d}.json.gz",'rt',encoding='utf-8') as f:rec=json.load(f)
            st,ts=preparations[rec['group']];out=run(rec,a,b,y,pools,ts,st)
            coarse=protected(rec,5,y.shape[1]);out.update(case=case['id'],rep=rep,group=rec['group'],coarse=coarse)
            allrows.append(out)
        save(ROOT/'small_batch_results.json',dict(rows=allrows,refit_batch=8,moment_batch=128,cases=sorted(set(v['case'] for v in allrows)),
            note='Focused ablation on existing independent data prefixes. Refit computation is executed in batches of eight; moment prefix statistics and their measured work are retained.'))
        rr=[v for v in allrows if v['case']==case['id']]
        print(case['id'],'decisions',sum(v['decision']!='unresolved' for v in rr),'median fine/coarse',np.median([v['cost']/v['coarse']['cost'] for v in rr]),flush=True)


if __name__=='__main__':main()

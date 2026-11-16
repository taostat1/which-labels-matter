"""Independent Monte Carlo audit, unavailable to planning and interval stopping."""
from study import *


def risk_many(z,y,t):
    zz=z.reshape(-1,z.shape[-1]);yy=y.reshape(-1);out=np.empty(len(t))
    for i in range(0,len(t),32):
        v=zz@t[i:i+32].T
        out[i:i+32]=(np.logaddexp(0,v)-yy[:,None]*v).mean(0)
    return out


def mean_se(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/np.sqrt(len(a))),replicates=len(a))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--available',action='store_true');ap.add_argument('--prepared',action='store_true');args=ap.parse_args()
    dest=ROOT/'audit_results.json';results=json.loads(dest.read_text('utf-8'))['cases'] if dest.exists() else {};reps=2048
    for ci,case in enumerate(CASES):
        if case['id'] in results and results[case['id']]['reps']==reps:continue
        if args.available and len(list((ROOT/'results').glob(case['id']+'_[0-9][0-9][0-9].json.gz')))<100:continue
        if args.prepared and not all((ROOT/'results'/f"setup_{case['id']}_{g}.json").exists() for g in range(4)):continue
        tick=time.perf_counter();a,o,y,pools=load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        zs=(a,b);r,d=a.shape[1:];groups=[]
        for group in range(4):
            st=json.loads((ROOT/'results'/f"setup_{case['id']}_{group}.json").read_text('utf-8'))
            pilot=np.asarray(st['pilot']);ts=[fit(z[pilot],y[pilot],case['lam'])[0] for z in zs]
            tcs=[test_constants(z[pools['comparison']],t) for z,t in zip(zs,ts)]
            dc=[cache_design(z,t,case['lam']) for z,t in zip(zs,ts)]
            groups.append(dict(st=st,ts=ts,tcs=tcs,dc=dc,approx=[[],[]]))
        my=[np.einsum('ird,ir->d',z[pools['comparison']],y[pools['comparison']])/y[pools['comparison']].size for z in zs]
        rng=np.random.default_rng(SEED+11000000+ci*100000);exact=[[],[]];maxresid=0.
        for pos in range(0,reps,128):
            ids=rng.choice(pools['training'],(min(128,reps-pos),case['n']));yy=y[ids];bs=len(ids)
            for j,z in enumerate(zs):
                fitted,_=batch_fit(z[ids],yy,case['lam'],groups[0]['ts'][j])
                pp=expit(np.einsum('bnrd,bd->bnr',z[ids],fitted))
                gr=np.einsum('bnrd,bnr->bd',z[ids],pp-yy)/(case['n']*r)+case['lam']*fitted
                maxresid=max(maxresid,float(np.linalg.norm(gr,axis=1).max()))
                exact[j].extend(risk_many(z[pools['comparison']],y[pools['comparison']],fitted).tolist())
                for gg in groups:
                    t=gg['ts'][j];tc=gg['tcs'][j];dc=gg['dc'][j]
                    H=dc['h'][ids].mean(1)+case['lam']*np.eye(d)
                    rhs=dc['u'][ids].mean(1)+np.einsum('bnrd,bnr->bd',z[ids],yy-.5)/(case['n']*r)
                    s=np.linalg.solve(H,rhs[...,None])[...,0]
                    value=tc['q']-my[j]@t+s@(tc['b']-my[j])+.5*np.einsum('bi,ij,bj->b',s,tc['C'],s)
                    gg['approx'][j].extend(value.tolist())
        ea,eb=np.array(exact[0]),np.array(exact[1]);delta=ea-eb;approximations=[]
        for group,gg in enumerate(groups):
            qa,qb=np.array(gg['approx'][0]),np.array(gg['approx'][1]);err=qa-qb-delta
            eps=gg['st']['epsilon'];eacheps=gg['st']['looks'][-1]['details']
            aa=dict(group=group,epsilon=eps,difference=mean_se(qa-qb),error=mean_se(err),
                    error_A=mean_se(qa-ea),error_B=mean_se(qb-eb),epsilon_A=eacheps[0]['used'],epsilon_B=eacheps[1]['used'])
            assert abs(aa['error']['mean'])<=eps+6*aa['error']['se']+1e-10
            approximations.append(aa)
        results[case['id']]=dict(case=case,reps=reps,A=mean_se(ea),B=mean_se(eb),difference=mean_se(delta),
            approximation=approximations,max_gradient_residual=maxresid,seconds=time.perf_counter()-tick,
            target='Expected exact fitted BCE on the declared comparison pool; independent future-training draws')
        save(ROOT/'audit_results.json',dict(cases=results,seed=SEED+11000000))
        if not (args.available or args.prepared):progress('独立目标风险审计',ci+1,len(CASES),f"800/800 条正式比较已完成；{case['id']} 的 2048 次未来拟合审计完成。")
        print('AUDIT',case['id'],results[case['id']]['difference'],'seconds',round(time.perf_counter()-tick,1),flush=True)


if __name__=='__main__':main()

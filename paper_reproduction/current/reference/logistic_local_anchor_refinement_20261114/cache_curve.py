from local_core import *
from anchor_core import fit,risk
from run import load_domain,SOURCE
import json,csv,time
cases={'text':[.05,.01],'image':[.125,.01]}
fractions=[0,.25,.5,.75,.9,1.]
config=dict(pools=[0,1,2],splits=8,pilot_objects=32,fractions=fractions,cases=cases,
    seed=202611141501,design='Uniform attribute acquisition; diagnostic error-bound curve, not an executed stopping rule.',
    valid='Anchor frozen; approximation fixed. Cache-dependent deterministic bound only.')
(ROOT/'cache_curve_protocol.json').write_text(json.dumps(config,indent=2))
out=[];began=time.perf_counter()
for domain,lams in cases.items():
    a0,b0,y0=load_domain(domain)
    for pool in config['pools']:
        sch=np.load(SOURCE/'coverage/results'/f'schedule_{domain}_{pool:02d}.npz')
        ids=sch['pool_ids'];a,b,y=a0[ids],b0[ids],y0[ids];r=y.shape[1]
        plan=list(zip(sch['train'][:8],sch['test'][:8]))
        pilot=np.random.default_rng(202611141401+pool+(100 if domain=='image' else 0)).permutation(len(ids))[:32]
        known0=np.full(y.size,np.nan);known0.reshape(y.shape)[pilot]=2*y[pilot]-1
        union=np.unique(np.concatenate([(np.r_[tr,te,None] if False else np.r_[tr,te])[:,None]*r+np.arange(r) for tr,te in plan]).ravel())
        # Only labels used in this diagnostic plan are queried; pilot labels all charged.
        unseen=union[~np.isfinite(known0[union])]
        order=np.random.default_rng(config['seed']+pool+(100 if domain=='image' else 0)).permutation(unseen)
        for lam in lams:
            anchors=[fit(z[pilot],y[pilot],lam)[0] for z in (a,b)]
            exact=[[fit(z[tr],y[tr],lam)[0] for z in (a,b)] for tr,te in plan]
            for fraction in fractions:
                known=known0.copy();keys=order[:int(np.ceil(fraction*len(order)))];known[keys]=(2*y-1).ravel()[keys]
                vals=[]
                for si,(tr,te) in enumerate(plan):
                    tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
                    gg=[candidate_geometry(z[tr],z[te],t,lam,known[tk]) for z,t in zip((a,b),anchors)]
                    old,new,ga,gb=certificate(*gg,lam,known[qk]);xx=(2*y[tr]-1).ravel();delta=[]
                    for j,g in enumerate((ga,gb)):
                        s=g['c']+g['L']@xx;v=g['vq'];u=g['q']@s;yy=y[te].ravel()
                        approx=np.mean(np.logaddexp(0,v)-yy*v+(expit(v)-yy)*u+.5*sigmoid_weight(v)*u*u)
                        delta.append((risk((a,b)[j][te],y[te],exact[si][j]),approx))
                    target=delta[0][0]-delta[1][0];surrogate=delta[0][1]-delta[1][1]
                    assert abs(target-surrogate)<=new['epsilon']+2e-8
                    vals.append([old['epsilon'],new['epsilon'],target,surrogate])
                old,new,target,surrogate=np.mean(vals,axis=0)
                row=dict(domain=domain,pool=pool,lam=lam,fraction=fraction,plan_labels=len(union),pilot_labels=32*r,
                    purchased=int(np.isfinite(known).sum()),old=float(old),new=float(new),target=float(target),surrogate=float(surrogate),
                    actual_error=float(abs(target-surrogate)),g=.005,epsilon_below_g=bool(new<.005),directional_room=float(.005+abs(surrogate)-new))
                out.append(row);(ROOT/'cache_curve.json').write_text(json.dumps(out,indent=2))
                print(domain,pool,lam,fraction,'bound',round(new,7),'purchased',row['purchased'],'room',round(row['directional_room'],7),flush=True)
with (ROOT/'cache_curve.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
print('DONE',len(out),round(time.perf_counter()-began,1),flush=True)

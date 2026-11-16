from local_core import *
from anchor_core import fit,risk
from run import load_domain,SOURCE
import json
out=[]
for domain,lam in [('text',.05),('text',.01),('image',.125),('image',.01)]:
    a0,b0,y0=load_domain(domain);sch=np.load(SOURCE/'coverage/results'/f'schedule_{domain}_00.npz')
    ids=sch['pool_ids'];a,b,y=a0[ids],b0[ids],y0[ids];r=y.shape[1]
    pilot=np.random.default_rng(202611141401+(100 if domain=='image' else 0)).permutation(len(ids))[:32]
    anchors=[fit(z[pilot],y[pilot],lam)[0] for z in (a,b)]
    plan=list(zip(sch['train'][:8],sch['test'][:8]));geoms=[]
    for tr,te in plan:
        known=np.full(y.size,np.nan);known.reshape(y.shape)[pilot]=2*y[pilot]-1
        tk=(tr[:,None]*r+np.arange(r)).ravel()
        geoms.append([candidate_geometry(z[tr],z[te],t,lam,known[tk]) for z,t in zip((a,b),anchors)])
    candidates=[('observed_audit',y.copy()),('all_unknown_zero',np.zeros_like(y)),('all_unknown_one',np.ones_like(y))]
    for sign in [-1,1]:
        latent=sign*(a[:,:,0]-np.median(a[:,:,0]));candidates.append((f'feature_sign_{sign}',(latent>0).astype(float)))
    for name,yy in candidates:
        yy[pilot]=y[pilot];delta=[]
        for (tr,te),gg in zip(plan,geoms):
            pair=[]
            for z,t,g in zip((a,b),anchors,gg):
                hat=fit(z[tr],yy[tr],lam)[0];s=g['c']+g['L']@(2*yy[tr]-1).ravel();u=g['q']@s;v=g['vq']
                approx=np.mean(np.logaddexp(0,v)-yy[te].ravel()*v+(expit(v)-yy[te].ravel())*u+.5*sigmoid_weight(v)*u*u)
                pair.append([risk(z[te],yy[te],hat),approx])
            delta.append([pair[0][0]-pair[1][0],pair[0][1]-pair[1][1]])
        actual,approx=np.mean(delta,axis=0)
        out.append(dict(domain=domain,lam=lam,completion=name,target=float(actual),surrogate=float(approx),error=float(abs(actual-approx))))
(ROOT/'completion_witnesses.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

from local_core import *
from anchor_core import fit,risk
import json,time,csv
sys.path.insert(0,str(ROOT.parent/'logistic_manuscript_paid_anchor_20261114'))
from run import load_domain,SOURCE

protocol=dict(pools=[0,1,2],splits='first 4 prescribed full-validation splits',pilot_objects=[0,32,128],
    penalty={'text':[.125,.05,.01,.001],'image':[.5,.125,.01,.001]},tolerance=.005,
    pilot_seed=202611141401,scope='Bound diagnosis only; no label-saving claim or tuning by held-out labels.',
    accounting='Pilot labels explicitly counted. Full labels used only for retrospective exact-risk audit.',
    comparisons=['current nonzero bound','local curvature plus exact scalar remainder','released zero bound'],
    collection='No modification to manuscript or released algorithm.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
rows=[];start=time.perf_counter()
for domain in ['text','image']:
    a0,b0,y0=load_domain(domain)
    for pool in protocol['pools']:
        sch=np.load(SOURCE/'coverage/results'/f'schedule_{domain}_{pool:02d}.npz')
        ids=sch['pool_ids'];a,b,y=a0[ids],b0[ids],y0[ids];r=y.shape[1]
        order=np.random.default_rng(protocol['pilot_seed']+pool+(100 if domain=='image' else 0)).permutation(len(ids))
        plan=list(zip(sch['train'][:4],sch['test'][:4]))
        for lam in protocol['penalty'][domain]:
            exact=[]
            for tr,te in plan:
                exact.append([fit(z[tr],y[tr],lam)[0] for z in (a,b)])
            for m in protocol['pilot_objects']:
                known=np.full(y.size,np.nan);pilot=order[:m]
                known.reshape(y.shape)[pilot]=2*y[pilot]-1
                anchors=[fit(z[pilot],y[pilot],lam)[0] if m else np.zeros(z.shape[-1]) for z in (a,b)]
                rr=[]
                for si,(tr,te) in enumerate(plan):
                    tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
                    gg=[candidate_geometry(z[tr],z[te],t,lam,known[tk]) for z,t in zip((a,b),anchors)]
                    old,new,ga,gb=certificate(*gg,lam,known[qk])
                    if m==0:
                        from anchor_core import paired_preconditioned
                        released=paired_preconditioned(a[tr],b[tr],a[te],b[te],lam)['epsilon']
                        old['epsilon']=min(old['epsilon'],released);new['epsilon']=min(new['epsilon'],old['epsilon'])
                    actual=[];approximations=[]
                    for j,(z,t,g) in enumerate(zip((a,b),anchors,(ga,gb))):
                        hat=exact[si][j];s=g['c']+g['L']@(2*y[tr]-1).ravel()
                        u=g['q']@s;v=g['vq'];yy=y[te].ravel()
                        approximations.append(float(np.mean(np.logaddexp(0,v)-yy*v+(expit(v)-yy)*u+.5*sigmoid_weight(v)*u*u)))
                        actual.append(risk(z[te],y[te],hat))
                        assert np.linalg.norm(hat-t-s)<=g['e']+2e-8
                    target=actual[0]-actual[1];surrogate=approximations[0]-approximations[1]
                    assert abs(target-surrogate)<=new['epsilon']+2e-8
                    rr.append(dict(old=old['epsilon'],new=new['epsilon'],paired=new['paired'],separate=new['separate'],target=target,surrogate=surrogate,
                        actual=abs(target-surrogate),old_e=float(np.mean([g['e'] for g in gg])),new_e=float(np.mean([ga['e'],gb['e']])),
                        old_rem=sum(g['remainder'] for g in gg),new_rem=ga['remainder']+gb['remainder']))
                row=dict(domain=domain,pool=pool,lam=lam,pilot=m,pilot_labels=m*r,**{k:float(np.mean([x[k] for x in rr])) for k in rr[0]})
                row['actual_plan_error']=abs(row['target']-row['surrogate']);row['remaining_at_true_center']=.005+abs(row['surrogate'])-row['new']
                rows.append(row)
                (ROOT/'screen_records.json').write_text(json.dumps(rows,indent=2))
                print(domain,pool,lam,m,'old/new/actual',*[f'{row[k]:.5g}' for k in ['old','new','actual_plan_error']],flush=True)
with (ROOT/'screen_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('DONE',len(rows),'conditions',round(time.perf_counter()-start,1),'seconds',flush=True)

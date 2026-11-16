"""Retrospective checks only: no audit value is supplied to acquisition."""
from local_core import *
from anchor_core import fit,risk
from run import load_domain,SOURCE,exact_fits
import json,csv,hashlib

rows=json.loads((ROOT/'chain_records.json').read_text())+json.loads((ROOT/'zero_dynamic_records.json').read_text())
assert len(rows)==15
aa,bb,yy=load_domain('image');audit=[]
for pool in [0,1,2]:
    sch=np.load(SOURCE/'coverage/results'/f'schedule_image_{pool:02d}.npz');ids=sch['pool_ids']
    a,b,y=aa[ids],bb[ids],yy[ids];r=y.shape[1];plan=list(zip(sch['train'],sch['test']));lam=.125
    pilot=np.random.default_rng(202611141401+pool+100).permutation(len(a))[:32]
    fitted,target=exact_fits(a,b,y,plan,lam)
    for mode in ['zero','pilot']:
        anchors=[np.zeros(z.shape[-1]) if mode=='zero' else fit(z[pilot],y[pilot],lam)[0] for z in (a,b)]
        approx=[];parameters=[];individual=[]
        for si,(tr,te) in enumerate(plan):
            pair=[]
            for j,z in enumerate((a,b)):
                zt=z[tr].reshape(-1,z.shape[-1]);q=z[te].reshape(-1,z.shape[-1]);theta0=anchors[j]
                vt=zt@theta0;vq=q@theta0;yt=y[tr].ravel();ye=y[te].ravel()
                H=lam*np.eye(len(theta0))+(zt.T*sigmoid_weight(vt))@zt/len(zt)
                step=-np.linalg.solve(H,zt.T@(expit(vt)-yt)/len(zt)+lam*theta0)
                u=q@step
                value=np.mean(np.logaddexp(0,vq)-ye*vq+(expit(vq)-ye)*u+.5*sigmoid_weight(vq)*u*u)
                pair.append(float(value));parameters.append(float(np.linalg.norm(theta0+step-fitted[si][j])))
                individual.append(float(abs(risk(z[te],y[te],fitted[si][j])-value)))
            approx.append(pair[0]-pair[1])
        surrogate=float(np.mean(approx));error=abs(target-surrogate)
        relevant=[v for v in rows if v['pool']==pool and v['method'].startswith(mode)]
        for v in relevant:
            assert abs(v['target']-target)<1e-10
            assert v['total_labels']==sum(v['stage_labels'].values())
            assert 768<=v['total_labels']<=24576
            assert v['covered'] and v['acceptable']
            e=min([h.get('epsilon',v['initial_epsilon']) for h in v['history']] or [v['initial_epsilon']])
            assert error<=e+2e-9
            v.update(actual_approximation_error=error,surrogate=surrogate,min_audited_epsilon=e)
        audit.append(dict(pool=pool,anchor=mode,target=target,surrogate=surrogate,
            actual_error=error,mean_parameter_error=float(np.mean(parameters)),
            mean_individual_risk_error=float(np.mean(individual)),only_A_acceptable=bool(target<-.005)))

methods=['zero_static','pilot_old_static','pilot_local_static','zero_local_cache','pilot_local_cache']
summary=[]
for method in methods:
    rr=[v for v in rows if v['method']==method]
    summary.append(dict(method=method,pools=len(rr),mean_total_labels=float(np.mean([v['total_labels'] for v in rr])),
        saved_percent=float(100*(1-np.mean([v['total_labels'] for v in rr])/24576)),
        fallback=sum(v['fallback'] for v in rr),acceptable=sum(v['acceptable'] for v in rr),covered=sum(v['covered'] for v in rr),
        pool0=rr[0]['total_labels'],pool1=rr[1]['total_labels'],pool2=rr[2]['total_labels']))
for name,values in [('chain_summary.csv',summary),('chain_audit.csv',audit)]:
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=values[0].keys());writer.writeheader();writer.writerows(values)
(ROOT/'combined_records.json').write_text(json.dumps(rows,indent=2))
(ROOT/'audit_checks.json').write_text(json.dumps(dict(status='passed',runs=len(rows),checks=[
    'Exact target recomputed after acquisition', 'Every final interval covers exact target',
    'Every returned choice acceptable', 'Approximation error below every inspected allowance',
    'Complete label count equals disjoint stage counts', 'Final training labels included'],details=audit),indent=2))
paths=list(ROOT.glob('*.py'))+[ROOT.parent/'logistic_manuscript_paid_anchor_20261114'/s for s in ['anchor_core.py','run.py']]
paths+=list((ROOT.parents[2]/'output/github/which-labels-logistic/src/whichlabels').glob('*.py'))
(ROOT/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(json.dumps(summary,indent=2));print('All audits passed.')

"""Label-free paired error bound for a zero-anchor logistic risk approximation."""
from pathlib import Path
import os,sys,time,json
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_active_cv_budget_20261030'))
from active_core import base,previous,study,np,expit

def op(x):return float(np.linalg.norm(x,2))
def gram(z):return z.T@z/len(z)
def eigmax(x):return max(0.,float(np.linalg.eigvalsh(x)[-1]))
def phi_ratio(t):
    # max |sigma(v)-1/2-v/4|/|v| over |v|<=t.
    return t*t/48 if t<1e-5 else .25-.5*np.tanh(t/2)/t
def phi_lip(t):return .25*np.tanh(t/2)**2
def phi_max(t):return t*phi_ratio(t)
def weight_lip(t):return min(1/(6*np.sqrt(3)),t/8)

def phis(t):
    t=np.asarray(t)
    # The cubic term is an upper bound, avoiding cancellation near zero.
    return np.where(t<1e-4,t**3/48,t/4-.5*np.tanh(t/2))

def mapped_bound(z,bounds):
    """Upper bound for ||mean_i z_i u_i|| for every |u_i|<=bounds_i."""
    if not np.any(bounds):return 0.
    N=len(z);k=eigmax(gram(z));rms=float(np.sqrt(np.mean(bounds**2)))
    weighted=eigmax((z.T*(bounds**2))@z/N)**.5
    component=float(np.linalg.norm(abs(z).T@bounds/N))
    return min(np.sqrt(k)*rms,weighted,component)

def row_geometry(z,q,lam):
    z=z.reshape(-1,z.shape[-1]);q=q.reshape(-1,q.shape[-1]);N=len(z)
    G=gram(z);H=lam*np.eye(z.shape[-1])+G/4;L=np.linalg.solve(H,z.T)/(2*N)
    V=z@L;Vq=q@L
    ell=np.sum(abs(V),axis=1);ellq=np.sum(abs(Vq),axis=1)
    r=np.linalg.norm(z,axis=1);rq=np.linalg.norm(q,axis=1)
    res=mapped_bound(z,phis(ell));d=res/lam
    k=eigmax(G);kq=eigmax(gram(q))
    grad=min(mapped_bound(q,expit(ellq+rq*d)),.5*np.sqrt(kq)+mapped_bound(q,expit(ellq+rq*d)-.5))
    remainder=float(np.mean(ellq**4)/192)
    separate=grad*d+remainder
    return dict(z=z,q=q,H=H,L=L,V=V,Vq=Vq,ell=ell,ellq=ellq,r=r,rq=rq,res=res,d=d,k=k,kq=kq,
                grad=grad,separate=separate)

def cross_bound(u,v,w):
    """||mean u_i v_i' t_i||, uniformly for |t_i|<=w_i."""
    if not np.any(w):return 0.
    return np.sqrt(eigmax((u.T*w)@u/len(u))*eigmax((v.T*w)@v/len(v)))

def precondition(a):
    a=dict(a);a['U']=np.linalg.solve(a['H'],a['z'].T).T
    a['r0']=mapped_bound(a['U'],phis(a['ell']))
    a['initial_d']=a['d']
    for _ in range(8):
        w=.25*np.tanh((a['ell']+a['r']*a['d'])/2)**2
        eta=cross_bound(a['U'],a['z'],w)
        if eta<1:a['d']=min(a['d'],a['r0']/(1-eta))
    a['tau']=a['ell']+a['r']*a['d'];a['w']=.25*np.tanh(a['tau']/2)**2
    a['eta']=cross_bound(a['U'],a['z'],a['w'])
    a['grad']=min(mapped_bound(a['q'],expit(a['ellq']+a['rq']*a['d'])),
                  .5*np.sqrt(a['kq'])+mapped_bound(a['q'],expit(a['ellq']+a['rq']*a['d'])-.5))
    a['separate']=a['grad']*a['d']+float(np.mean(a['ellq']**4)/192)
    return a

def paired_preconditioned(za,zb,qa,qb,lam):
    A=precondition(row_geometry(za,qa,lam));B=precondition(row_geometry(zb,qb,lam))
    dz=A['z']-B['z'];dq=A['q']-B['q'];du=A['U']-B['U']
    rd=np.linalg.norm(dz,axis=1);rdq=np.linalg.norm(dq,axis=1)
    dv=np.sum(abs(A['V']-B['V']),axis=1);dvq=np.sum(abs(A['Vq']-B['Vq']),axis=1)
    lp=.25*np.tanh(np.maximum(A['ell'],B['ell'])/2)**2
    dr=min(mapped_bound(du,phis(A['ell']))+mapped_bound(B['U'],lp*dv),
           mapped_bound(du,phis(B['ell']))+mapped_bound(A['U'],lp*dv))
    tau=np.maximum(A['tau'],B['tau']);lw=np.minimum(1/(6*np.sqrt(3)),tau/8)
    choices=[];details=[]
    for a,b in [(A,B),(B,A)]:
        dgeometry=cross_bound(du,a['z'],a['w'])+cross_bound(b['U'],dz,a['w'])
        dweights=cross_bound(b['U'],b['z'],lw*(dv+rd*b['d']))
        feedback=cross_bound(b['U'],b['z'],lw*a['r'])
        den=1-a['eta']-b['d']*feedback
        if den>0:choices.append((dr+b['d']*(dgeometry+dweights))/den)
        details.append(dict(preconditioned_residual_difference=dr,eta=a['eta'],geometry=dgeometry,
                            weight_constant=dweights,weight_feedback=feedback,denominator=den))
    de=min(choices+[A['d']+B['d']])
    risk=[]
    for a,b in [(A,B),(B,A)]:
        delta_logit=dvq+a['rq']*de+rdq*b['d']
        first=mapped_bound(dq,expit(a['ellq']+a['rq']*a['d']))
        grad_diff=first+mapped_bound(b['q'],.25*delta_logit)
        risk.append(grad_diff*a['d']+b['grad']*de)
    remainder=float(np.mean(phis(np.maximum(A['ellq'],B['ellq']))*dvq))
    paired=min(risk)+remainder
    return dict(epsilon=min(paired,A['separate']+B['separate']),paired=paired,separate=A['separate']+B['separate'],
                risk_increment=min(risk),quadratic_remainder=remainder,parameter_error_difference=de,
                directions=details,candidates=[dict(initial_d=a['initial_d'],d=a['d'],eta=a['eta'],
                    r0=a['r0'],gradient=a['grad'],separate=a['separate']) for a in (A,B)])

def paired_rows(za,zb,qa,qb,lam):
    A=row_geometry(za,qa,lam);B=row_geometry(zb,qb,lam)
    dz=A['z']-B['z'];dq=A['q']-B['q'];rd=np.linalg.norm(dz,axis=1);rdq=np.linalg.norm(dq,axis=1)
    dv=np.sum(abs(A['V']-B['V']),axis=1);dvq=np.sum(abs(A['Vq']-B['Vq']),axis=1)
    ell=np.maximum(A['ell'],B['ell']);lp=.25*np.tanh(ell/2)**2
    dr=min(mapped_bound(dz,phis(A['ell']))+mapped_bound(B['z'],lp*dv),
           mapped_bound(dz,phis(B['ell']))+mapped_bound(A['z'],lp*dv))
    dg0=.25*np.sqrt(eigmax(gram(dz)))*(np.sqrt(A['k'])+np.sqrt(B['k']))
    tau=np.maximum(A['ell']+A['r']*A['d'],B['ell']+B['r']*B['d'])
    lw=np.minimum(1/(6*np.sqrt(3)),tau/8)
    choices=[];feedbacks=[]
    for a,b in [(A,B),(B,A)]:
        constant=dg0+eigmax((b['z'].T*(lw*(dv+rd*b['d'])))@b['z']/len(b['z']))
        lips=eigmax((b['z'].T*(lw*a['r']))@b['z']/len(b['z']))
        feedback=b['d']*lips/lam;feedbacks.append(feedback)
        if feedback<1:choices.append((dr+b['d']*constant)/(lam*(1-feedback)))
    de=min(choices+[A['d']+B['d']])
    risk=[]
    for a,b in [(A,B),(B,A)]:
        delta_logit=dvq+a['rq']*de+rdq*b['d']
        first=mapped_bound(dq,expit(a['ellq']+a['rq']*a['d']))
        grad_diff=first+mapped_bound(b['q'],.25*delta_logit)
        risk.append(grad_diff*a['d']+b['grad']*de)
    remainder=float(np.mean(phis(np.maximum(A['ellq'],B['ellq']))*dvq))
    paired=min(risk)+remainder
    return dict(epsilon=min(paired,A['separate']+B['separate']),paired=paired,separate=A['separate']+B['separate'],
                risk_increment=min(risk),quadratic_remainder=remainder,parameter_error_difference=de,
                training_residual_difference=dr,feedback=feedbacks,
                delta_logit_max=float(dv.max()),delta_eval_logit_max=float(dvq.max()),
                candidates=[dict(d=a['d'],residual=a['res'],gradient=a['grad'],separate=a['separate'],
                                 max_logit=float(a['ell'].max()),max_eval_logit=float(a['ellq'].max()),
                                 rms_logit=float(np.sqrt(np.mean(a['ell']**2)))) for a in (A,B)])

def geometry(z,q,lam):
    z=z.reshape(-1,z.shape[-1]);q=q.reshape(-1,q.shape[-1]);N=len(z)
    G=gram(z);Gq=gram(q);H=lam*np.eye(z.shape[-1])+G/4
    L=np.linalg.solve(H,z.T)/(2*N);s=np.sqrt(N)*op(L)
    k=eigmax(G);kq=eigmax(Gq);row=float(np.linalg.norm(z,axis=1).max());rowq=float(np.linalg.norm(q,axis=1).max())
    residual=k*phi_ratio(row*s)*s
    d=min(residual/lam,k*row*s*s/(12*np.sqrt(3)*lam))
    S=s+d
    separate=(.5*np.sqrt(kq)+kq*s/4)*d+kq*d*d/8+kq*rowq**2*s**4/192
    return dict(z=z,q=q,N=N,G=G,H=H,L=L,s=s,k=k,kq=kq,row=row,rowq=rowq,d=d,S=S,
                residual=residual,separate=separate)

def paired_bound(za,zb,qa,qb,lam):
    A=geometry(za,qa,lam);B=geometry(zb,qb,lam);N=A['N']
    dz=A['z']-B['z'];dq=A['q']-B['q']
    kd=eigmax(gram(dz));kdq=eigmax(gram(dq))
    rowd=float(np.linalg.norm(dz,axis=1).max())
    dt=np.sqrt(N)*op(A['L']-B['L'])
    def logrms(a,b,training,change):
        ka,kb=(a['k'],b['k']) if training else (a['kq'],b['kq'])
        dd=kd if training else kdq
        return min(np.sqrt(ka)*change+np.sqrt(dd)*b['s'],np.sqrt(kb)*change+np.sqrt(dd)*a['s'])
    dv=logrms(A,B,True,dt)
    tau=max(A['row']*A['s'],B['row']*B['s'])
    dr=min(np.sqrt(kd*A['k'])*phi_ratio(A['row']*A['s'])*A['s']+np.sqrt(B['k'])*phi_lip(tau)*dv,
           np.sqrt(kd*B['k'])*phi_ratio(B['row']*B['s'])*B['s']+np.sqrt(A['k'])*phi_lip(tau)*dv)
    dg0=.25*np.sqrt(kd)*(np.sqrt(A['k'])+np.sqrt(B['k']))
    tau_path=max(A['row']*A['S'],B['row']*B['S']);lw=weight_lip(tau_path)
    de_choices=[]
    for a,b in [(A,B),(B,A)]:
        constant=dg0+b['k']*lw*(a['row']*dt+rowd*b['S'])
        feedback=b['d']*b['k']*lw*a['row']/lam
        if feedback<1:
            de_choices.append((dr+b['d']*constant)/(lam*(1-feedback)))
    de=min(de_choices+[A['d']+B['d']])
    # Difference of the exact-vs-one-step risk increments.
    erisk=[]
    for a,b in [(A,B),(B,A)]:
        delta_v=min(np.sqrt(a['kq'])*(dt+de)+np.sqrt(kdq)*b['S'],
                    np.sqrt(b['kq'])*(dt+de)+np.sqrt(kdq)*a['S'])
        first=min(np.sqrt(kdq)*expit(a['rowq']*a['S']),
                  np.sqrt(kdq)*(.5+np.sqrt(a['kq'])*a['S']/4))
        grad_diff=first+.25*np.sqrt(b['kq'])*delta_v
        grad_b=min(np.sqrt(b['kq'])*expit(b['rowq']*b['S']),
                   .5*np.sqrt(b['kq'])+b['kq']*b['S']/4)
        erisk.append(grad_diff*a['d']+grad_b*de)
    dvq=logrms(A,B,False,dt)
    tq=max(A['rowq']*A['s'],B['rowq']*B['s'])
    max_rms=min(np.sqrt(A['kq'])*A['s'],np.sqrt(B['kq'])*B['s'])+dvq
    dqrem=min(phi_max(tq)*dvq,phi_ratio(tq)*max_rms*dvq)
    epaired=min(erisk)+dqrem
    return dict(epsilon=min(epaired,A['separate']+B['separate']),paired=epaired,
                separate=A['separate']+B['separate'],risk_increment=min(erisk),quadratic_remainder=dqrem,
                parameter_error_difference=de,one_step_difference=dt,training_residual_difference=dr,
                gram_difference=kd,evaluation_gram_difference=kdq,
                candidates=[{k:v for k,v in a.items() if k not in ['z','q','G','H','L']} for a in (A,B)])

def actual(za,zb,qa,qb,ytr,yq,lam):
    rows=[]
    for z,q in [(za,qa),(zb,qb)]:
        geo=geometry(z,q,lam);theta,meta=base.fit(z,ytr,lam);t=geo['L']@(2*ytr.ravel()-1)
        v=geo['q']@t
        exact=base.mean_loss(q,yq,theta)
        one=base.mean_loss(q,yq,t)
        quadratic=float(np.mean(np.log(2)+(.5-yq.ravel())*v+v*v/8))
        rows.append(dict(exact=exact,one=one,quadratic=quadratic,error=exact-quadratic,
                         parameter_error=float(np.linalg.norm(theta-t)),fit_residual=meta['residual']))
    return dict(exact_difference=rows[0]['exact']-rows[1]['exact'],
                quadratic_difference=rows[0]['quadratic']-rows[1]['quadratic'],
                paired_error=rows[0]['error']-rows[1]['error'],candidates=rows)

def main():
    tick=time.perf_counter();a0,o0,y0,pools=study.load_domain('text')
    a0=a0.astype(float);o0=o0.astype(float)
    a0/=np.maximum(1,np.linalg.norm(a0,axis=-1,keepdims=True));o0/=np.maximum(1,np.linalg.norm(o0,axis=-1,keepdims=True))
    records=[];summaries=[]
    for group in range(3):
        sch=np.load(ROOT.parent/'logistic_active_cv_budget_20261030'/'results'/f'schedule_{group}.npz')
        ids=sch['pool_ids'];a=a0[ids];b=.9*a+.1*o0[ids];y=y0[ids]
        rr=[]
        for split,(tr,te) in enumerate(zip(sch['train'],sch['test'])):
            result=paired_bound(a[tr],b[tr],a[te],b[te],.125)
            localized=paired_rows(a[tr],b[tr],a[te],b[te],.125)
            preconditioned=paired_preconditioned(a[tr],b[tr],a[te],b[te],.125)
            audit=actual(a[tr],b[tr],a[te],b[te],y[tr],y[te],.125)
            original=sum(base.bound(z[tr],z[te],.125)['epsilon'] for z in (a,b))
            row=dict(group=group,split=split,original=original,bound=result,localized=localized,preconditioned=preconditioned,audit=audit);records.append(row);rr.append(row)
            assert abs(audit['paired_error'])<=result['epsilon']+1e-8
            assert abs(audit['paired_error'])<=localized['epsilon']+1e-8
            assert abs(audit['paired_error'])<=preconditioned['epsilon']+1e-8
        summary=dict(group=group,original=float(np.mean([r['original'] for r in rr])),
                     paired=float(np.mean([r['bound']['paired'] for r in rr])),
                     epsilon=float(np.mean([r['bound']['epsilon'] for r in rr])),
                     separate=float(np.mean([r['bound']['separate'] for r in rr])),
                     localized_paired=float(np.mean([r['localized']['paired'] for r in rr])),
                     localized_epsilon=float(np.mean([r['localized']['epsilon'] for r in rr])),
                     localized_separate=float(np.mean([r['localized']['separate'] for r in rr])),
                     preconditioned=float(np.mean([r['preconditioned']['epsilon'] for r in rr])),
                     actual_paired_error=float(np.mean([r['audit']['paired_error'] for r in rr])),
                     mean_absolute_fold_error=float(np.mean([abs(r['audit']['paired_error']) for r in rr])),
                     exact_difference=float(np.mean([r['audit']['exact_difference'] for r in rr])))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    base.save(ROOT/'diagnostics.json',dict(groups=summaries,folds=records,seconds=time.perf_counter()-tick))

if __name__=='__main__':main()

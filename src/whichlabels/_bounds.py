"""Feature-only separate and paired approximation bounds."""
import numpy as np
from scipy.special import expit

def bound(ztr,zte,lam,t=None):
    d=ztr.shape[-1];zt=ztr.reshape(-1,d);zq=zte.reshape(-1,d)
    t=np.zeros(d) if t is None else t
    pt=expit(zt@t);pq=expit(zq@t)
    H=(zt.T*(pt*(1-pt)))@zt/len(zt)+lam*np.eye(d)
    inv=np.linalg.inv(H);G=zt.T@zt/len(zt);Gq=zq.T@zq/len(zq)
    J=(zt.T*abs(zt@t))@zt/len(zt);Jq=(zq.T*abs(zq@t))@zq/len(zq)
    c=inv@(zt.T@(.5-pt)/len(zt)-lam*t)
    radius=np.linalg.norm(c)+.5*np.sqrt(max(0,np.linalg.eigvalsh(inv@G@inv)[-1]))
    kp=np.linalg.eigvalsh(G)[-1];kq=np.linalg.eigvalsh(Gq)[-1]
    jp=np.linalg.eigvalsh(J)[-1];jq=np.linalg.eigvalsh(Jq)[-1]
    dpar=min((jp*radius**2/16+kp*radius**3/48)/lam,kp*radius**2/(12*np.sqrt(3)*lam))
    gradient=np.linalg.norm(zq.T@(pq-.5)/len(zq))+.5*np.sqrt(kq)
    eps=(gradient+kq*radius/4)*dpar+kq*dpar*dpar/8+jq*radius**3/48+kq*radius**4/192
    return dict(epsilon=float(eps),radius=float(radius),kp=float(kp),kq=float(kq))

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

"""Uniform certificates over binary completions of the acquired-label cache.

The anchor, polynomial, and query distribution remain fixed at zero.
Only the deterministic approximation certificate changes with the cache.
"""
import numpy as np
from scipy.special import expit
from ._bounds import mapped_bound, cross_bound

CURVATURE_LIP=1/(6*np.sqrt(3))

def sigmoid_weight(t):
    e=expit(-np.abs(t))
    return e*(1-e)

def candidate_geometry(z,q,anchor,lam,known_x):
    z=z.reshape(-1,z.shape[-1]);q=q.reshape(-1,q.shape[-1]);N=len(z)
    v=z@anchor;vq=q@anchor;p=expit(v);pq=expit(vq);w=sigmoid_weight(v)
    H=(z.T*w)@z/N+lam*np.eye(z.shape[-1])
    U=np.linalg.solve(H,z.T).T
    L=U.T/(2*N)
    c=np.linalg.solve(H,z.T@(.5-p)/N-lam*anchor)
    known=np.isfinite(known_x);unknown=~known
    ck=c+L[:,known]@known_x[known]
    V=z@L;Vq=q@L
    center=z@ck;centerq=q@ck
    ell=np.abs(center)+np.abs(V[:,unknown]).sum(1)
    ellq=np.abs(centerq)+np.abs(Vq[:,unknown]).sum(1)
    # Extrema of sigma(v+u)-sigma(v)-sigma'(v)u occur at
    # endpoints or u=-2v, because sigma'(v+u)=sigma'(v).
    rem=lambda u:expit(v+u)-p-w*u
    rho=np.maximum(np.abs(rem(-ell)),np.abs(rem(ell)))
    rho=np.maximum(rho,np.where(np.abs(2*v)<=ell,np.abs(rem(-2*v)),0.))
    rho=np.nextafter(rho,np.inf)+2e-15
    rnorm=mapped_bound(z,rho);e=rnorm/lam
    r0=mapped_bound(U,rho)
    rows=np.linalg.norm(z,axis=1);qrows=np.linalg.norm(q,axis=1)
    eta=float('inf')
    for _ in range(8):
        shift=ell+rows*e
        wmin=sigmoid_weight(np.abs(v)+shift)
        wmax=sigmoid_weight(np.maximum(0,np.abs(v)-shift))
        wdev=np.maximum(np.abs(wmin-w),np.abs(wmax-w))
        eta=cross_bound(U,z,wdev)
        if eta<1:e=min(e,r0/(1-eta))
    # Recompute the contraction envelope using the final certified radius.
    shift=ell+rows*e
    wdev=np.maximum(np.abs(sigmoid_weight(np.abs(v)+shift)-w),
                    np.abs(sigmoid_weight(np.maximum(0,np.abs(v)-shift))-w))
    eta=cross_bound(U,z,wdev)
    grad=mapped_bound(q,expit(np.abs(vq)+ellq+qrows*e))
    remainder=np.minimum(ellq**3/(36*np.sqrt(3)),np.abs(vq)*ellq**3/48+ellq**4/192).mean()
    C=(q.T*sigmoid_weight(vq))@q/len(q)
    b=q.T@(pq-.5)/len(q)
    return dict(z=z,q=q,t=anchor,v=v,vq=vq,H=H,U=U,L=L,c=c,ck=ck,V=V,Vq=Vq,
        unknown=unknown,ell=ell,ellq=ellq,rho=rho,e=e,r0=r0,eta=eta,wdev=wdev,
        rows=rows,qrows=qrows,gradient=grad,remainder=float(remainder),
        separate=float(grad*e+remainder),C=C,b=b)

def affine_difference(a,b,test=False):
    matrix=(a['Vq']-b['Vq']) if test else (a['V']-b['V'])
    z=a['q'] if test else a['z'];q=b['q'] if test else b['z']
    return np.abs(z@a['ck']-q@b['ck'])+np.abs(matrix[:,a['unknown']]).sum(1)

def paired_certificate(a,b):
    dz=a['z']-b['z'];dq=a['q']-b['q'];du=a['U']-b['U']
    rd=np.linalg.norm(dz,axis=1);rdq=np.linalg.norm(dq,axis=1)
    delta_anchor=np.abs(a['v']-b['v']);delta_anchor_q=np.abs(a['vq']-b['vq'])
    delta_step=affine_difference(a,b);delta_step_q=affine_difference(a,b,True)
    ell=np.maximum(a['ell'],b['ell'])
    # Derivatives of the training gradient remainder with respect to u and v.
    lu=np.minimum(.25,CURVATURE_LIP*ell)
    lv=np.minimum(ell**2/16,2*CURVATURE_LIP*ell)
    residual_difference=np.minimum(
        mapped_bound(du,a['rho'])+mapped_bound(b['U'],lu*delta_step+lv*delta_anchor),
        mapped_bound(du,b['rho'])+mapped_bound(a['U'],lu*delta_step+lv*delta_anchor))
    candidates=[a['e']+b['e']];denominators=[]
    for x,y in ((a,b),(b,a)):
        geometry=cross_bound(du,x['z'],x['wdev'])+cross_bound(y['U'],dz,x['wdev'])
        weights=cross_bound(y['U'],y['z'],CURVATURE_LIP*(2*delta_anchor+delta_step+rd*y['e']))
        feedback=cross_bound(y['U'],y['z'],CURVATURE_LIP*x['rows'])
        denominator=1-x['eta']-y['e']*feedback
        denominators.append(denominator)
        if denominator>0:candidates.append((residual_difference+y['e']*(geometry+weights))/denominator)
    de=min(candidates)
    if np.array_equal(a['z'],b['z']) and np.array_equal(a['t'],b['t']):de=0.
    increments=[]
    for x,y in ((a,b),(b,a)):
        delta_logits=delta_anchor_q+delta_step_q+x['qrows']*de+rdq*y['e']
        gradient_difference=mapped_bound(dq,expit(np.abs(x['vq'])+x['ellq']+x['qrows']*x['e']))
        gradient_difference+=mapped_bound(y['q'],.25*delta_logits)
        increments.append(gradient_difference*x['e']+y['gradient']*de)
    eq=np.maximum(a['ellq'],b['ellq']);vq=np.maximum(np.abs(a['vq']),np.abs(b['vq']))
    rho_sup=np.minimum(eq**2/(12*np.sqrt(3)),np.minimum(1+.25*eq,vq*eq**2/16+eq**3/48))
    rem=float(np.mean(rho_sup*delta_step_q+(eq**3/48)*delta_anchor_q))
    separate=a['separate']+b['separate'];paired=min(increments)+rem
    return dict(epsilon=float(min(separate,paired)),paired=float(paired),separate=float(separate),
        parameter_errors=[a['e'],b['e']],parameter_error_difference=de,
        contraction=[a['eta'],b['eta']],denominators=denominators,
        risk_increment=float(min(increments)),paired_remainder=rem)


def residual_range(v,lo,hi):
    p=expit(v);w=sigmoid_weight(v)
    fun=lambda s:expit(v+s)-p-w*s
    values=[fun(lo),fun(hi)]
    for s in [np.zeros_like(v),-2*v]:
        values.append(np.where((lo<=s)&(s<=hi),fun(s),np.nan))
    a=np.nanmin(values,axis=0)-3e-14;b=np.nanmax(values,axis=0)+3e-14
    return a,b


def loss_remainder(v,lo,hi):
    p=expit(v);e=expit(-abs(v));w=e*(1-e)
    def der(u):return expit(v+u)-p-w*u
    def rem(u):return np.logaddexp(0,v+u)-np.logaddexp(0,v)-p*u-.5*w*u*u
    knots=[lo,np.clip(np.minimum(0.,-2*v),lo,hi),np.clip(np.maximum(0.,-2*v),lo,hi),hi]
    out=np.maximum.reduce([abs(rem(t)) for t in knots])
    for a,b in zip(knots[:-1],knots[1:]):
        a=a.copy();b=b.copy();fa=der(a);fb=der(b);mask=fa*fb<0
        if not mask.any():continue
        for _ in range(60):
            mid=(a+b)/2;fm=der(mid);left=fa*fm<=0
            b=np.where(mask&left,mid,b);a=np.where(mask&~left,mid,a);fa=np.where(mask&~left,fm,fa)
        out=np.maximum(out,np.where(mask,abs(rem((a+b)/2)),0.))
    return out+3e-13

def norm_box(z,lo,hi):
    center=(lo+hi)/2;rad=(hi-lo)/2
    return min(mapped_bound(z,np.maximum(abs(lo),abs(hi))),
               np.linalg.norm(z.T@center/len(z))+mapped_bound(z,rad))*(1+2e-12)+2e-14

def refine(old,lam,known_eval=None,iterations=12):
    g=dict(old);z,q=g['z'],g['q'];n=len(z);d=z.shape[1]
    rad=abs(g['V'][:,g['unknown']]).sum(1)
    radq=abs(g['Vq'][:,g['unknown']]).sum(1)
    center=z@g['ck'];centerq=q@g['ck']
    slo,shi=center-rad,center+rad
    rlo,rhi=residual_range(g['v'],slo,shi)
    rho=np.maximum(abs(rlo),abs(rhi))
    # Initial global strong-convexity radius is valid for every completion.
    e=min(g['e'],norm_box(z,rlo,rhi)/lam)
    train_error=g['rows']*e;test_error=g['qrows']*e
    tmax=abs(g['v']+center)+rad
    if known_eval is None:known_eval=np.full(len(q),np.nan)
    known=np.isfinite(known_eval);ycenter=np.where(known,(known_eval+1)/2,.5)
    best_fit=float('inf');best_metric=None
    trace=[]
    for it in range(iterations):
        wmin=sigmoid_weight(tmax+train_error)
        M=lam*np.eye(d)+(z.T*wmin)@z/n
        ev,V=np.linalg.eigh(M);invroot=(V/np.sqrt(ev))@V.T
        zr=z@invroot;qr=q@invroot
        R=norm_box(zr,rlo,rhi)
        newe=R/np.sqrt(ev[0])
        et=R*np.linalg.norm(zr,axis=1);eq=R*np.linalg.norm(qr,axis=1)
        e=min(e,newe);train_error=np.minimum(train_error,et);test_error=np.minimum(test_error,eq)
        lower=g['vq']+centerq-radq-test_error;upper=g['vq']+centerq+radq+test_error
        glo=expit(lower)-ycenter-.5*(~known);ghi=expit(upper)-ycenter+.5*(~known)
        G=norm_box(qr,glo,ghi)
        risk_bound=R*G
        row_bound=np.mean(np.maximum(abs(glo),abs(ghi))*test_error)
        best_fit=min(best_fit,risk_bound,row_bound)
        trace.append(dict(iteration=it,parameter_radius=e,residual_metric=R,gradient_metric=G,fit_bound=best_fit,min_curvature=float(ev[0])))
        best_metric=(M,R)
    # Tightened Euclidean radii can also enter the existing paired argument.
    lower=g['v']+center-rad-train_error;upper=g['v']+center+rad+train_error
    lowweight=sigmoid_weight(np.maximum(abs(lower),abs(upper)))
    highweight=sigmoid_weight(np.where((lower<=0)&(upper>=0),0,np.minimum(abs(lower),abs(upper))))
    w0=sigmoid_weight(g['v']);wdev=np.maximum(abs(lowweight-w0),abs(highweight-w0))
    gl=g['vq']+centerq-radq-test_error;gu=g['vq']+centerq+radq+test_error
    glo=expit(gl)-ycenter-.5*(~known);ghi=expit(gu)-ycenter+.5*(~known)
    grad=norm_box(q,glo,ghi)
    rem=float(loss_remainder(g['vq'],centerq-radq,centerq+radq).mean())
    g.update(e=e,rho=rho,r0=norm_box(g['U'],rlo,rhi),wdev=wdev,
             eta=cross_bound(g['U'],z,wdev),gradient=grad,remainder=min(g['remainder'],rem),
             separate=min(g['separate'],best_fit+rem,grad*e+rem),local_trace=trace,
             train_error=train_error,test_error=test_error,residual_lo=rlo,residual_hi=rhi)
    return g

def certificate(a,b,lam,known_eval=None):
    old=paired_certificate(a,b)
    aa=refine(a,lam,known_eval);bb=refine(b,lam,known_eval)
    new=paired_certificate(aa,bb)
    new['epsilon']=min(new['epsilon'],old['epsilon'])
    return old,new,aa,bb


def approximation_bound(prepared, known_values, *, separate=False):
    """Return a feature/cache certificate without acquiring additional labels.

    known_values is a flat array of acquired binary labels and NaNs elsewhere.
    The bound is valid for every completion of those NaNs. This uniformity
    permits the same labels to determine the query interval and this bound.
    """
    values = np.asarray(known_values, dtype=float)
    expected = len(prepared.a) * prepared.a.shape[1]
    if values.shape != (expected,):
        raise ValueError('known_values must have one entry per object-attribute label')
    known = np.isfinite(values)
    if np.any(~np.isin(values[known], [0., 1.])):
        raise ValueError('known values must be binary; use NaN for unknown labels')
    x = np.where(known, 2*values-1, np.nan)
    r = prepared.a.shape[1]
    result = []
    for train, evaluation in prepared.splits:
        tk = (train[:, None]*r + np.arange(r)).ravel()
        ek = (evaluation[:, None]*r + np.arange(r)).ravel()
        designs = [candidate_geometry(z[train], z[evaluation],
                   np.zeros(z.shape[-1]), prepared.penalty, x[tk])
                   for z in (prepared.a, prepared.b)]
        refined = certificate(*designs, prepared.penalty, x[ek])[1]
        result.append(refined['separate' if separate else 'epsilon'])
    return float(np.mean(result))

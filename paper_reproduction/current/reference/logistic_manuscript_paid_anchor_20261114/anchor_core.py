"""Pilot-conditional nonzero-anchor algebra, reused with unchanged formulas.
Original: logistic_paid_anchor_20261108/anchor_core.py.
The quality-repair data imports and optional no-stop screen are removed.
"""
from pathlib import Path
import os,sys,time,json
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'output/github/which-labels-logistic/src'))
import numpy as np
from scipy.special import expit
from scipy import sparse
from whichlabels._bounds import mapped_bound,cross_bound,paired_preconditioned
from whichlabels.solver import fit,risk
from whichlabels._intervals import endpoints,mapping_guard
from whichlabels.labels import LabelCache,ArrayLabelProvider

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

def prepare(a,b,splits,anchors,lam,known_values):
    start=time.perf_counter();r=a.shape[1];total=a.shape[0]*r
    xknown=np.where(np.isfinite(known_values),2*known_values-1,np.nan)
    linear=np.zeros(total);constant=0.;rr=[];cc=[];vv=[];bounds=[];geometries=[]
    for tr,te in splits:
        tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
        u,v=np.triu_indices(len(tk),1)
        quad=np.zeros((len(tk),len(tk)));cross=np.zeros((len(qk),len(tk)));gg=[]
        for sign,z,theta in ((1.,a,anchors[0]),(-1.,b,anchors[1])):
            g=candidate_geometry(z[tr],z[te],theta,lam,xknown[tk]);gg.append(g)
            L,C,c,q,vq=g['L'],g['C'],g['c'],g['q'],g['vq']
            base=np.mean(np.logaddexp(0,vq)-.5*vq)+g['b']@c+.5*c@C@c
            constant+=sign*base/len(splits)
            np.add.at(linear,tk,sign*(L.T@(g['b']+C@c))/len(splits))
            np.add.at(linear,qk,-sign*(q@(theta+c))/(2*len(qk)*len(splits)))
            quad+=sign*.5*(L.T@C@L)
            cross-=sign*(q@L)/(2*len(qk))
        constant+=float(np.trace(quad))/len(splits)
        rr.append(np.minimum(tk[u],tk[v]).astype(np.int32));cc.append(np.maximum(tk[u],tk[v]).astype(np.int32))
        vv.append((quad[u,v]+quad[v,u])/len(splits))
        left=np.repeat(qk,len(tk));right=np.tile(tk,len(qk))
        rr.append(np.minimum(left,right).astype(np.int32));cc.append(np.maximum(left,right).astype(np.int32));vv.append(cross.ravel()/len(splits))
        bd=paired_certificate(*gg)
        if not np.any(anchors[0]) and not np.any(anchors[1]):
            # Recover the original zero-anchor paired certificate as well.
            bd['epsilon']=min(bd['epsilon'],paired_preconditioned(a[tr],b[tr],a[te],b[te],lam)['epsilon'])
        bounds.append(bd)
        geometries.append([(g['L'],g['c'],g['C'],g['b'],g['vq']) for g in gg])
    matrix=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(total,total)).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros();co=matrix.tocoo()
    left=co.row;right=co.col;coef=co.data
    known=np.isfinite(xknown);lk=known[left];rk=known[right]
    both=lk&rk
    constant+=float(np.sum(coef[both]*xknown[left[both]]*xknown[right[both]]))
    mask=lk&~rk;np.add.at(linear,right[mask],coef[mask]*xknown[left[mask]])
    mask=rk&~lk;np.add.at(linear,left[mask],coef[mask]*xknown[right[mask]])
    constant+=float(linear[known]@xknown[known]);linear[known]=0
    remain=~lk&~rk;single=np.flatnonzero(linear!=0)
    left=np.r_[left[remain],single].astype(np.int32)
    right=np.r_[right[remain],np.full(len(single),-1)].astype(np.int32)
    coef=np.r_[coef[remain],linear[single]]
    mass=float(np.abs(coef).sum());prob=np.abs(coef)/mass if mass else np.array([])
    cdf=np.cumsum(prob)
    if mass:cdf[-1]=1.
    return dict(constant=constant,left=left,right=right,coef=coef,mass=mass,cdf=cdf,
        epsilon=float(np.mean([v['epsilon'] for v in bounds])),bounds=bounds,geometry=geometries,
        seconds=time.perf_counter()-start,known_keys=np.flatnonzero(known),single_terms=len(single),
        terms=len(coef),conditional_on_paid_pilot=True)

def audit(poly,a,b,y,splits,anchors,lam,exact_parameters=None):
    x=(2*y-1).ravel();right=np.ones(len(poly['coef']));pair=poly['right']>=0
    right[pair]=x[poly['right'][pair]]
    sparse_value=poly['constant']+float(poly['coef']@(x[poly['left']]*right))
    direct=[];exact=[];step_errors=[];candidate_risk_errors=[]
    for si,(tr,te) in enumerate(splits):
        row=[];truth=[]
        for j,z in enumerate((a,b)):
            L,c,C,bb,vq=poly['geometry'][si][j]
            s=c+L@(2*y[tr]-1).ravel();theta=anchors[j]+s
            q=z[te].reshape(-1,z.shape[-1]);yy=y[te].ravel();p=expit(vq)
            quadratic=np.mean(np.logaddexp(0,vq)-yy*vq)+(bb-q.T@(yy-.5)/len(yy))@s+.5*s@C@s
            # bb=q' (p-.5)/M, so the linear term is q'(p-y)/M.
            row.append(float(quadratic))
            optimum=fit(z[tr],y[tr],lam)[0] if exact_parameters is None else exact_parameters[si][j]
            truth.append(risk(z[te],y[te],optimum))
            step_errors.append(float(np.linalg.norm(theta-optimum)))
            candidate_risk_errors.append(abs(truth[-1]-row[-1]))
        direct.append(row[0]-row[1]);exact.append(truth[0]-truth[1])
    target=float(np.mean(exact));surrogate=float(np.mean(direct))
    assert abs(sparse_value-surrogate)<2e-8,(sparse_value,surrogate)
    assert abs(target-surrogate)<=poly['epsilon']+2e-8
    variance=max(0.,poly['mass']**2-(surrogate-poly['constant'])**2)
    return dict(target=target,surrogate=surrogate,actual_error=abs(target-surrogate),
        polynomial_identity_error=abs(sparse_value-surrogate),sampling_variance=variance,
        standard_error_at_32768=np.sqrt(variance/32768),mean_parameter_error=float(np.mean(step_errors)),
        max_parameter_error=max(step_errors),mean_candidate_risk_error=float(np.mean(candidate_risk_errors)))

def choose(lo,hi,g):return 'A' if hi<=g else 'B' if lo>=-g else 'unresolved'

def sample(poly,cache,seed,g,delta=.05):
    checks=[128,256,512,1024,2048,4096,8192,16384,32768]
    mass=poly['mass'];constant=poly['constant'];eps=poly['epsilon'];history=[]
    initial_half=mass+eps+2e-9
    if choose(constant-initial_half,constant+initial_half,g)!='unresolved':
        return dict(choice=choose(constant-initial_half,constant+initial_half,g),interval=[constant-initial_half,constant+initial_half],
                    draws=0,history=history,screen='public_choice')
    if not mass:
        return dict(choice='unresolved',interval=[constant-initial_half,constant+initial_half],
                    draws=0,history=history,screen='zero_mass')
    rng=np.random.default_rng(seed);n=k=0
    guard=2e-9
    drift=mapping_guard(poly['coef'],poly['cdf'],mass)
    assert drift<guard,(drift,guard)
    for end in checks:
        idx=np.searchsorted(poly['cdf'],rng.random(end-n),side='right')
        xl=2*cache.query_keys(poly['left'][idx],'comparison')-1
        kr=poly['right'][idx];pair=kr>=0;xr=np.ones(len(idx))
        xr[pair]=2*cache.query_keys(kr[pair],'comparison')-1
        signs=np.sign(poly['coef'][idx])*xl*xr
        assert np.isin(signs,[-1.,1.]).all()
        k+=int((signs>0).sum());n=end
        lo,hi=endpoints(k,n,delta/len(checks))
        lo=constant+mass*(2*lo-1)-eps-guard;hi=constant+mass*(2*hi-1)+eps+guard
        choice=choose(lo,hi,g)
        history.append(dict(draws=n,successes=k,interval=[lo,hi],labels=cache.count,choice=choice))
        if choice!='unresolved':break
    return dict(choice=choice,interval=[lo,hi],draws=n,history=history,screen='sampled',mapping_guard=guard,mapping_error_bound=drift)

"""Independent finite checks transcribed from the paper, not experiment code.

Exhaustive checks supplement, and do not replace, the analytic review.
Only files next to this script are written.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
RHO = 1 / (6 * np.sqrt(3))
LABELS = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
CACHES = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=6)))


def eigmax(a):
    return max(0.0, float(np.linalg.eigvalsh(a)[-1]))


def op(a):
    return float(np.linalg.norm(a, 2)) if a.size else 0.0


def vecbound(c, b):
    n = len(c)
    return min(op(c) * np.linalg.norm(b) / n,
               op(b[:, None] * c) / np.sqrt(n),
               np.linalg.norm(np.abs(c).T @ b) / n)


def boxbound(c, lo, hi):
    return min(vecbound(c, np.maximum(np.abs(lo), np.abs(hi))),
               np.linalg.norm(c.T @ ((lo + hi) / 2)) / len(c)
               + vecbound(c, (hi - lo) / 2))


def crossbound(c, d, w):
    n = len(c)
    return np.sqrt(eigmax(c.T @ (w[:, None] * c) / n)
                   * eigmax(d.T @ (w[:, None] * d) / n))


def phi(t):
    return 0.5 * np.tanh(t / 2) - t / 4


def aenv(t):
    return np.tanh(t / 2)**2 / 4


def remainder(t):
    return np.logaddexp(0, t) - np.log(2) - t / 2 - t*t / 8


def loss(z, y, theta):
    logits = z @ theta
    return np.mean(np.logaddexp(0, logits) - y * logits)


def fit(z, y, lam):
    theta = np.zeros(z.shape[1])
    for _ in range(100):
        p = expit(z @ theta)
        grad = z.T @ (p - y) / len(z) + lam * theta
        if np.linalg.norm(grad) < 3e-14:
            break
        h = z.T @ ((p * (1 - p))[:, None] * z) / len(z) + lam * np.eye(z.shape[1])
        step = np.linalg.solve(h, grad)
        value = loss(z, y, theta) + lam * np.dot(theta, theta) / 2
        scale = 1.0
        while loss(z, y, theta-scale*step) + lam*np.dot(theta-scale*step, theta-scale*step)/2 > value - 1e-4*scale*np.dot(grad, step) + 1e-15:
            scale /= 2
            if scale < 2**-40:
                break
        theta -= scale * step
    residual = np.linalg.norm(z.T @ (expit(z @ theta) - y) / len(z) + lam * theta)
    assert residual < 1e-10, residual
    return theta, residual


def single_cache(z, w, lam, cache):
    n = len(z)
    h = lam*np.eye(z.shape[1]) + z.T@z/(4*n)
    l = np.linalg.solve(h, z.T)/(2*n)
    u_mat = z@np.linalg.inv(h)
    known = cache[:n] != 0
    m = l[:, known] @ cache[:n][known]
    aa, bb = z@m, np.abs(z@l[:, ~known]).sum(axis=1)
    aq, bq = w@m, np.abs(w@l[:, ~known]).sum(axis=1)
    low, high = phi(aa+bb), phi(aa-bb)
    d = boxbound(z, low, high)/lam
    ru, rv = np.linalg.norm(z, axis=1)*d, np.linalg.norm(w, axis=1)*d
    for _ in range(12):
        p = expit(np.abs(aa)+bb+ru)
        metric = lam*np.eye(z.shape[1]) + z.T@((p*(1-p))[:, None]*z)/n
        ev, evec = np.linalg.eigh(metric)
        invsqrt = (evec/np.sqrt(ev))@evec.T
        rr = boxbound(z@invsqrt, low, high)
        d = min(d, rr/np.sqrt(ev[0]))
        ru = np.minimum(ru, rr*np.linalg.norm(z@invsqrt, axis=1))
        rv = np.minimum(rv, rr*np.linalg.norm(w@invsqrt, axis=1))
    ylo = np.where(cache[n:] == 0, 0, (cache[n:]+1)/2)
    yhi = np.where(cache[n:] == 0, 1, (cache[n:]+1)/2)
    glo = expit(aq-bq-rv)-yhi
    ghi = expit(aq+bq+rv)-ylo
    fitbound = min(rr*boxbound(w@invsqrt, glo, ghi),
                   np.mean(np.maximum(np.abs(glo), np.abs(ghi))*rv),
                   d*boxbound(w, glo, ghi))
    separate = fitbound+np.mean(np.maximum(np.abs(remainder(aq-bq)), np.abs(remainder(aq+bq))))
    return dict(z=z,w=w,l=l,u_mat=u_mat,m=m,bb=np.abs(aa)+bb,
                cc=np.abs(aq)+bq,lo=low,hi=high,d=d,ru=ru,rv=rv,
                G=boxbound(w,glo,ghi),separate=separate,
                eta=crossbound(u_mat,z,aenv(np.abs(aa)+bb+ru)),
                wa=aenv(np.abs(aa)+bb+ru), known=known)


def paired_cache(za,zb,wa,wb,lam,cache):
    aa, bb = single_cache(za,wa,lam,cache), single_cache(zb,wb,lam,cache)
    unknown = ~aa['known']
    vp = np.abs(za@aa['m']-zb@bb['m'])+np.abs((za@aa['l']-zb@bb['l'])[:,unknown]).sum(axis=1)
    vq = np.abs(wa@aa['m']-wb@bb['m'])+np.abs((wa@aa['l']-wb@bb['l'])[:,unknown]).sum(axis=1)
    coeff = np.minimum(.25,RHO*np.maximum(aa['bb'],bb['bb']))*vp
    diff_u = aa['u_mat']-bb['u_mat']
    qdiff = min(vecbound(diff_u,np.maximum(np.abs(aa['lo']),np.abs(aa['hi'])))+vecbound(bb['u_mat'],coeff),
                vecbound(diff_u,np.maximum(np.abs(bb['lo']),np.abs(bb['hi'])))+vecbound(aa['u_mat'],coeff))
    delta = aa['d']+bb['d']
    for x,y in [(aa,bb),(bb,aa)]:
        train_diff = np.linalg.norm(x['z']-y['z'],axis=1)
        fixed = (crossbound(x['u_mat']-y['u_mat'],x['z'],x['wa'])
                 +crossbound(y['u_mat'],x['z']-y['z'],x['wa'])
                 +crossbound(y['u_mat'],y['z'],RHO*(vp+train_diff*y['d'])))
        feedback = crossbound(y['u_mat'],y['z'],RHO*np.linalg.norm(x['z'],axis=1))
        den = 1-x['eta']-y['d']*feedback
        if den > 0:
            delta = min(delta,(qdiff+y['d']*fixed)/den)
    if np.array_equal(za,zb) and np.array_equal(wa,wb):
        delta = 0.0
    directions=[]
    for x,y in [(aa,bb),(bb,aa)]:
        tq=x['cc']+np.linalg.norm(x['w'],axis=1)*x['d']
        diff_grad=(vecbound(x['w']-y['w'],expit(tq))
                   +vecbound(y['w'],(vq+np.linalg.norm(x['w'],axis=1)*delta
                              +np.linalg.norm(x['w']-y['w'],axis=1)*y['d'])/4))
        directions.append(diff_grad*x['d']+y['G']*delta)
    cmax=np.maximum(aa['cc'],bb['cc'])
    penv=np.minimum(np.minimum(RHO*cmax*cmax/2,1+cmax/4),cmax**3/48)
    eps=min(min(directions)+np.mean(penv*vq),aa['separate']+bb['separate'])
    return eps,aa,bb,delta


def finite_bounds(z, c, cache):
    q,r,d=z.shape
    known=cache != 0
    m=c.mean(axis=0)-(z*cache.reshape(q,r,1)).sum(axis=(0,1))/(2*q*r)
    unknown_z=z.reshape(-1,d)[~known]
    v=np.abs(unknown_z).sum(axis=0)/(2*q*r)
    gamma=unknown_z.T@unknown_z/(q*r)
    f=len(unknown_z)/(q*r)
    maxvals=[]
    for i in range(q):
        possibilities=np.array(list(itertools.product((-1.,1.),repeat=r)))
        mask=np.all((cache[i*r:(i+1)*r]==0)|(possibilities==cache[i*r:(i+1)*r]),axis=1)
        scores=c[i]-(possibilities[mask]@z[i])/(2*r)
        maxvals.append(np.max(np.sum(scores*scores,axis=1)))
    u2=float(np.mean(maxvals))
    u1=min(u2,np.linalg.norm(np.abs(m)+v)**2,(np.linalg.norm(m)+np.sqrt(f*eigmax(gamma))/2)**2)
    return float(u1),u2,float(np.sqrt(max(maxvals)))


def moments_bounds(u1,u2,b,lam,n):
    u=np.sqrt(u1)
    rr=b+u
    m2=((1-1/n)*u1+u2/n)/lam**2
    m4=(u**4+6*u*u*u2/n+4*u*rr*u2/n**2+3*u2*u2/n**2+rr*rr*u2/n**3)/lam**4
    m4=min(m4,(b/lam)**2*m2)
    return np.array([m2,np.sqrt(m2*m4),m4])


def compatible(cache):
    return np.all((cache==0)|(LABELS==cache),axis=1)


def main():
    rng=np.random.default_rng(20261117)
    z=rng.normal(size=(3,2,2))
    z=z/np.maximum(1,np.linalg.norm(z,axis=2,keepdims=True))
    other=rng.normal(size=(3,2,2))
    other=other/np.maximum(1,np.linalg.norm(other,axis=2,keepdims=True))
    out={'seed':20261117,'author_experiment_modules_imported':False,
         'cache_patterns':len(CACHES),'full_labelings':len(LABELS)}
    a2=[]
    for lam,alpha in [(.05,.35),(.3,1.),(1.,0.)]:
        za=z[:2].reshape(4,2); wa=z[2]
        zb=((1-alpha)*z[:2]+alpha*other[:2]).reshape(4,2)
        wb=(1-alpha)*z[2]+alpha*other[2]
        actual=[]; errors=[]; max_residual=0
        for x in LABELS:
            pair=[]; err=[]
            for zz,ww in [(za,wa),(zb,wb)]:
                theta,res=fit(zz,(x[:4]+1)/2,lam)
                max_residual=max(max_residual,res)
                h=lam*np.eye(2)+zz.T@zz/16
                tt=np.linalg.solve(h,zz.T@x[:4]/8)
                approx=np.mean(np.log(2)-x[4:]*(ww@tt)/2+(ww@tt)**2/8)
                pair.append(loss(ww,(x[4:]+1)/2,theta)-approx)
                err.append(theta-tt)
            actual.append(pair[0]-pair[1]); errors.append(err)
        actual=np.array(actual); errors=np.array(errors)
        max_excess=-np.inf; max_ratio=0.; completion_checks=0
        for cache in CACHES:
            eps,aa,bb,dd=paired_cache(za,zb,wa,wb,lam,cache)
            ix=compatible(cache); completion_checks+=int(ix.sum())
            measured=np.max(np.abs(actual[ix]))
            max_excess=max(max_excess,measured-eps)
            if eps>1e-15: max_ratio=max(max_ratio,measured/eps)
            assert measured <= eps+2e-10
            for j,info in enumerate([aa,bb]):
                assert np.max(np.linalg.norm(errors[ix,j],axis=1)) <= info['d']+2e-10
                assert np.max(np.abs(errors[ix,j]@info['z'].T)-info['ru']) <= 2e-10
                assert np.max(np.abs(errors[ix,j]@info['w'].T)-info['rv']) <= 2e-10
            assert np.max(np.linalg.norm(errors[ix,0]-errors[ix,1],axis=1)) <= dd+2e-10
        a2.append({'lambda':lam,'candidate_mix':alpha,'completion_checks':completion_checks,
                   'maximum_score_error_minus_bound':float(max_excess),
                   'maximum_score_error_to_bound_ratio':float(max_ratio),
                   'max_optimizer_gradient_residual':float(max_residual)})
    out['A2']=a2
    print(json.dumps({'A2':a2},ensure_ascii=False),flush=True)
    c2=[]
    for lam,n,theta0 in [(.1,2,np.zeros(2)),(.4,3,np.array([.4,-.2]))]:
        q,r,d=z.shape
        p=expit(z@theta0)
        c=np.einsum('qrd,qr->qd',z,p-.5)/r+lam*theta0
        cq=c-lam*theta0
        gram=np.einsum('qra,qrb->qab',z,z)/r
        jgram=np.einsum('qr,qra,qrb->qab',np.abs(z@theta0),z,z)/r
        k=max(eigmax(g) for g in gram); j=max(eigmax(g) for g in jgram)
        kq=eigmax(gram.mean(axis=0)); jq=eigmax(jgram.mean(axis=0)); hq=kq/4
        a2c=j/(16*lam); a3=k/(48*lam); ag=RHO*k/(2*lam)
        samples=list(itertools.product(range(q),repeat=n))
        actual_mu=[]; actual_nu=[]; actual_b=[]; actual_moments=[]; actual_risk_error=[]
        max_residual=0
        for x in LABELS:
            score=c-np.einsum('qrd,qr->qd',z,x.reshape(q,r))/(2*r)
            actual_mu.append(np.linalg.norm(score.mean(axis=0))**2)
            actual_nu.append(np.mean(np.sum(score*score,axis=1)))
            actual_b.append(np.max(np.linalg.norm(score,axis=1)))
            moments=[]; risks=[]
            y=(x+1)/2
            gq=np.einsum('qrd,qr->d',z,p-y.reshape(q,r))/(q*r)
            cq_h=np.einsum('qr,qra,qrb->ab',p*(1-p),z,z)/(q*r)
            l0=loss(z.reshape(-1,d),y,theta0)
            for sample in samples:
                inds=np.array(sample)
                zs=z[inds].reshape(-1,d); ys=y.reshape(q,r)[inds].reshape(-1)
                pp=p[inds].reshape(-1)
                hh=lam*np.eye(d)+zs.T@((pp*(1-pp))[:,None]*zs)/(n*r)
                step=-np.linalg.solve(hh,score[inds].mean(axis=0))
                norm=np.linalg.norm(step)
                moments.append([norm**2,norm**3,norm**4])
                theta,res=fit(zs,ys,lam)
                max_residual=max(max_residual,res)
                risks.append(loss(z.reshape(-1,d),y,theta)-(l0+gq@step+step@cq_h@step/2))
            actual_moments.append(np.mean(moments,axis=0))
            actual_risk_error.append(abs(np.mean(risks)))
        actual_mu=np.array(actual_mu); actual_nu=np.array(actual_nu); actual_b=np.array(actual_b)
        actual_moments=np.array(actual_moments); actual_risk_error=np.array(actual_risk_error)
        excesses=np.full(5,-np.inf); completion_checks=0
        for cache in CACHES:
            u1,u2,b=finite_bounds(z,c,cache)
            u1q,_,_=finite_bounds(z,cq,cache)
            mb=moments_bounds(u1,u2,b,lam,n)
            global_eps=(RHO/(2*lam)+RHO*b/(6*lam))*mb[0]
            local_eps=(np.sqrt(u1q)*(a2c*mb[0]+a3*mb[1])
                       +hq*(a2c*mb[1]+a3*mb[2])+.5*hq*ag*ag*mb[2]
                       +jq*mb[1]/48+kq*mb[2]/192)
            ix=compatible(cache); completion_checks+=int(ix.sum())
            values=[max(actual_mu[ix])-u1,max(actual_nu[ix])-u2,max(actual_b[ix])-b,
                    np.max(actual_moments[ix]-mb),max(actual_risk_error[ix])-min(global_eps,local_eps)]
            excesses=np.maximum(excesses,values)
            assert max(values)<2e-10,values
        c2.append({'lambda':lam,'training_size':n,'anchor':theta0.tolist(),
                   'completion_checks':completion_checks,'training_index_tuples_per_labeling':len(samples),
                   'max_excess_u1_u2_B_moments_risk':excesses.tolist(),
                   'max_optimizer_gradient_residual':float(max_residual)})
    out['C2_and_Theorems5_6']=c2
    out['result']='PASS; finite numerical validation only, not a universal proof'
    (ROOT/'independent_cache_math_check.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

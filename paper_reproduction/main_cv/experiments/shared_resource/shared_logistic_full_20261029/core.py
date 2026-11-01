"""Shared logistic fits and a quadratic risk statistic. Labels are in [0,1]."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json, time
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('LOGISTIC_FEATURE_DIR',str(ROOT.parent/'goemotions_comparison_20261028')))

def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    def convert(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,Path):return str(value)
        raise TypeError(f'Unsupported JSON type: {type(value)}')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2,default=convert), encoding='utf-8')
    for attempt in range(12):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.05*2**attempt,1.))

def progress(stage, done, total, note=''):
    d=dict(stage=stage, done=done, total=total, percent=100*done/max(total,1),
           note=note, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'progress.json',d)
    (ROOT/'进程.txt').write_text(f"更新：{d['updated']}\n阶段：{stage}\n进度：{done}/{total} ({d['percent']:.1f}%)\n{note}\n",encoding='utf-8')

def data():
    f=np.load(SOURCE/'prepared/features_public.npz'); y=np.load(SOURCE/'prepared/labels_oracle.npz')
    assert np.array_equal(f['ids'], y['ids'])
    return f['a'], f['other'], (y['y'].astype(float)+1)/2, {k:np.flatnonzero(f['roles']==k) for k in ('training','comparison','audit')}

def fit(z,y,lam,tol=1e-10,maxiter=40,start=None):
    """Damped Newton, all coefficients penalized; a residual certifies accuracy."""
    d=z.shape[-1]; z=z.reshape(-1,d); y=y.reshape(-1)
    t=np.zeros(d) if start is None else start.copy(); calls=0
    for it in range(maxiter):
        v=z@t; p=expit(v)
        g=z.T@(p-y)/len(y)+lam*t
        if np.linalg.norm(g)<=tol: break
        h=(z.T*(p*(1-p)))@z/len(y)+lam*np.eye(d)
        s=np.linalg.solve(h,g); old=np.logaddexp(0,v).mean()-y@v/len(y)+lam*t@t/2
        rate=1.
        for _ in range(30):
            u=t-rate*s; vv=z@u
            obj=np.logaddexp(0,vv).mean()-y@vv/len(y)+lam*u@u/2
            calls+=1
            if obj<=old-1e-4*rate*(g@s)+1e-15: break
            rate*=.5
        t=u
    residual=float(np.linalg.norm(z.T@(expit(z@t)-y)/len(y)+lam*t))
    if residual>tol*10: raise RuntimeError(f'Newton residual {residual}')
    return t,dict(iterations=it,objective_calls=calls,residual=residual)

def fit_batch(z,y,lam,tol=1e-10):
    batch,n,r,d=z.shape; t=np.zeros((batch,d)); iterations=0
    for _ in range(20):
        v=np.einsum('bnrd,bd->bnr',z,t,optimize=True); p=expit(v)
        g=np.einsum('bnrd,bnr->bd',z,p-y,optimize=True)/(n*r)+lam*t
        if np.max(np.linalg.norm(g,axis=1))<=tol:break
        h=np.einsum('bnri,bnr,bnrj->bij',z,p*(1-p),z,optimize=True)/(n*r)+lam*np.eye(d)
        s=np.linalg.solve(h,g[...,None])[...,0]
        # A full Newton step is checked; individual rows backtrack if necessary.
        old=(np.logaddexp(0,v)-y*v).mean((1,2))+lam*np.sum(t*t,1)/2
        rate=np.ones(batch)
        for __ in range(30):
            u=t-rate[:,None]*s; vv=np.einsum('bnrd,bd->bnr',z,u,optimize=True)
            obj=(np.logaddexp(0,vv)-y*vv).mean((1,2))+lam*np.sum(u*u,1)/2
            bad=obj>old-1e-4*rate*np.sum(g*s,1)+1e-14
            if not bad.any():break
            rate[bad]*=.5
        t=u;iterations+=1
    else:raise RuntimeError('batch Newton did not converge')
    return t,iterations

def test_constants(z,t):
    r=z.shape[1]; zz=z.reshape(-1,z.shape[-1]); p=expit(zz@t)
    return dict(q=float(np.logaddexp(0,zz@t).mean()),
                q_center=float((np.logaddexp(0,zz@t)-.5*(zz@t)).mean()),
                b_center=zz.T@(p-.5)/len(zz),
                b=zz.T@p/len(zz), C=(zz.T*(p*(1-p)))@zz/len(zz))

def risk(z,y,t):
    v=np.einsum('ird,d->ir',z,t)
    return float(np.mean(np.logaddexp(0,v)-y*v))

def step(z,y,t,lam):
    d=z.shape[-1]; zz=z.reshape(-1,d); yy=y.reshape(-1); p=expit(zz@t)
    H=(zz.T*(p*(1-p)))@zz/len(yy)+lam*np.eye(d)
    return -np.linalg.solve(H,zz.T@(p-yy)/len(yy)+lam*t)

def quadratic_risk(z,y,t,s,tc):
    meanzy=np.einsum('ird,ir->d',z,y)/y.size
    return float(tc['q']-meanzy@t+(tc['b']-meanzy)@s+.5*s@tc['C']@s)

def coefficients(ztrain,ztest,t,lam,tc):
    """Batch coefficient array [constant, 3r linear, all distinct pairs]."""
    batch,n,r,d=ztrain.shape; p=expit(np.einsum('bnrd,d->bnr',ztrain,t))
    H=np.einsum('bnri,bnr,bnrj->bij',ztrain,p*(1-p),ztrain,optimize=True)/(n*r)+lam*np.eye(d)
    inv=np.linalg.solve(H,np.broadcast_to(np.eye(d),(batch,d,d)))
    g=np.einsum('bnrd,bnr->bd',ztrain,p,optimize=True)/(n*r)+lam*t
    c=-np.einsum('bij,bj->bi',inv,g)
    B=np.einsum('bij,bnrj->bnir',inv,ztrain[:,:2],optimize=True)/r
    B1,B2=B[:,0],B[:,1]; C=tc['C']; b=tc['b']
    cc=C@c[...,None]; cc=cc[...,0]
    constant=tc['q']+c@b+.5*np.einsum('bi,bi->b',c,cc)
    linear=np.zeros((batch,3*r))
    linear[:,:r]=-np.einsum('brd,bd->br',ztest,t+c)/r
    linear[:,r:2*r]=np.einsum('bdr,bd->br',B1,b+cc)
    D=np.einsum('bdi,de,bej->bij',B1,C,B1,optimize=True)/(2*n)
    E=np.einsum('bdi,de,bej->bij',B1,C,B2,optimize=True)*(n-1)/(2*n)
    F=-np.einsum('bid,bdj->bij',ztest,B1,optimize=True)/r
    linear[:,r:2*r]+=np.diagonal(D,axis1=1,axis2=2)
    l,h=np.triu_indices(3*r,1); w=np.zeros((batch,len(l)))
    index={(int(x),int(y)):k for k,(x,y) in enumerate(zip(l,h))}
    for i in range(r):
        for j in range(i+1,r):w[:,index[(r+i,r+j)]]=D[:,i,j]+D[:,j,i]
        for j in range(r):
            w[:,index[(r+i,2*r+j)]]=E[:,i,j]
            w[:,index[(i,r+j)]]=F[:,i,j]
    return constant,linear,w,l,h

def evaluate(co,yy):
    c,v,w,l,h=co
    return c+np.einsum('bi,bi->b',v,yy)+np.einsum('bi,bi->b',w,yy[:,l]*yy[:,h])

def eb(x,absbound,delta):
    x=np.asarray(x); m=len(x)
    if m<2:return np.inf
    return float(np.sqrt(2*np.var(x,ddof=1)*np.log(4/delta)/m)+14*absbound*np.log(4/delta)/(3*(m-1)))

def calibration(z,y,t,lam,public,delta=.01):
    m,r,d=z.shape; p=expit(np.einsum('ird,d->ir',z,t))
    score=np.einsum('ird,ir->id',z,p-y)/r+lam*t
    pp=expit(np.einsum('ird,d->ir',public,t))
    center=np.einsum('ird,ir->id',public,pp-.5)/r
    width=np.abs(public).mean(1)/2
    coordinate=np.maximum(abs(center-width),abs(center+width)).max(0)
    B=min(1.+lam*np.linalg.norm(t),float(np.linalg.norm(coordinate+abs(lam*t))))
    rad=np.array([eb(score[:,k]-lam*t[k],coordinate[k],delta/(d+1)) for k in range(d)])
    mu2=min(B*B,float(np.sum((abs(score.mean(0))+rad)**2)))
    sq=np.sum(score**2,axis=1)
    second=min(B*B,float(sq.mean()+eb(sq,B*B,delta/(d+1))))
    return dict(mu2=mu2,second=second,B=float(B),actual_mean_score_norm=float(np.linalg.norm(score.mean(0))),m=m)

def approx_bound(cal,n,lam):
    rho=1/(6*np.sqrt(3)); v=((1-1/n)*cal['mu2']+cal['second']/n)/lam**2
    return float((rho/(2*lam)+rho*cal['B']/(6*lam))*v)

def deterministic_bound(t,lam):
    """Conservative absolute bound for the raw-label polynomial, one candidate."""
    c=1/lam+np.linalg.norm(t)
    return float(np.logaddexp(0,np.linalg.norm(t))+c+c*c/8 +np.linalg.norm(t)+c
                 +(1/lam)*(1+c/4)+1/(8*lam**2)+1/lam)

def centered(co):
    """The identical polynomial in signed labels x=2y-1."""
    c,v,w,l,h=co; vv=.5*v.copy(); ww=.25*w
    cc=c+.5*v.sum(1)+ww.sum(1)
    for k in range(len(l)):
        vv[:,l[k]]+=ww[:,k]; vv[:,h[k]]+=ww[:,k]
    return cc,vv,ww,l,h

def centered_bound(t,lam):
    """Absolute bound relative to log(2), valid for every feature/label block."""
    rho=1/(6*np.sqrt(3)); norm=float(np.linalg.norm(t))
    v=rho*norm*norm/(2*lam); c=norm+v
    return float(v*v/8+rho*c**3/6 +v/2
                 +(rho*norm*norm/2+v/4)/(2*lam)+1/(32*lam*lam)+1/(4*lam))

def moment_coefficients(ztrain,ztest,t,lam,tc):
    """Center at 1/2 BEFORE replacing sample averages by indices 1 and 2."""
    batch,n,r,d=ztrain.shape; p=expit(np.einsum('bnrd,d->bnr',ztrain,t))
    H=np.einsum('bnri,bnr,bnrj->bij',ztrain,p*(1-p),ztrain,optimize=True)/(n*r)+lam*np.eye(d)
    inv=np.linalg.solve(H,np.broadcast_to(np.eye(d),(batch,d,d)))
    c=np.einsum('bij,bj->bi',inv,np.einsum('bnrd,bnr->bd',ztrain,.5-p,optimize=True)/(n*r)-lam*t)
    B=np.einsum('bij,bnrj->bnir',inv,ztrain[:,:2],optimize=True)/r
    B1,B2=B[:,0],B[:,1]; C=tc['C']; b=tc['b_center']; cc=c@C
    constant=tc['q_center']+c@b+.5*np.einsum('bi,bi->b',c,cc)
    linear=np.zeros((batch,3*r))
    linear[:,:r]=-np.einsum('brd,bd->br',ztest,t+c)/(2*r)
    linear[:,r:2*r]=np.einsum('bdr,bd->br',B1,b+cc)/2
    D=np.einsum('bdi,de,bej->bij',B1,C,B1,optimize=True)/(8*n)
    E=np.einsum('bdi,de,bej->bij',B1,C,B2,optimize=True)*(n-1)/(8*n)
    F=-np.einsum('bid,bdj->bij',ztest,B1,optimize=True)/(4*r)
    constant+=np.trace(D,axis1=1,axis2=2)
    l,h=np.triu_indices(3*r,1); w=np.zeros((batch,len(l)))
    index={(int(x),int(y)):k for k,(x,y) in enumerate(zip(l,h))}
    for i in range(r):
        for j in range(i+1,r): w[:,index[(r+i,r+j)]]=D[:,i,j]+D[:,j,i]
        for j in range(r):
            w[:,index[(r+i,2*r+j)]]=E[:,i,j]
            w[:,index[(i,r+j)]]=F[:,i,j]
    return constant,linear,w,l,h

"""Uniform zero-anchor paired-error cap for iid public feature plans."""
from paired_local import *

def summary(z):
    gram=np.einsum('nri,nrj->nij',z,z)/z.shape[1]
    return dict(k=float(np.linalg.eigvalsh(gram)[:,-1].max()),
                row=float(np.linalg.norm(z,axis=-1).max()))

def mapnorm(k,lam,w):
    t=min(k,lam/w) if w else k
    return np.sqrt(t)/(lam+w*t)

def cap(za,zb,qa,qb,lam):
    p=[summary(z) for z in (za,zb)];q=[summary(z) for z in (qa,qb)]
    dp=summary(za-zb);dq=summary(qa-qb)
    ga=np.einsum('nri,nrj->nij',za,za)/za.shape[1]
    gb=np.einsum('nri,nrj->nij',zb,zb)/zb.shape[1]
    dh=float(np.linalg.norm(ga-gb,ord=2,axis=(1,2)).max())/4
    for v,w in zip(p,q):
        v['u']=mapnorm(v['k'],lam,.25);v['s']=v['u']/2
        v['ell']=v['row']*v['s'];v['ellq']=w['row']*v['s']
        v['rho']=v['ell']/4-.5*np.tanh(v['ell']/2)+2e-15
        v['e']=np.sqrt(v['k'])*v['rho']/lam
        for _ in range(12):
            v['radius']=v['ell']+v['row']*v['e']
            v['wd']=.25-sigmoid_weight(v['radius'])+2e-15
            v['eta']=v['u']*np.sqrt(v['k'])*v['wd']
            if v['eta']<1:v['e']=min(v['e'],v['u']*v['rho']/(1-v['eta']))
        v['radius']=v['ell']+v['row']*v['e'];v['wd']=.25-sigmoid_weight(v['radius'])+2e-15
        v['eta']=v['u']*np.sqrt(v['k'])*v['wd']
        v['grad']=np.sqrt(w['k'])*expit(v['ellq']+w['row']*v['e'])
    ud=min(np.sqrt(dp['k'])/lam+dh*p[0]['u']/lam,
           np.sqrt(dp['k'])/lam+dh*p[1]['u']/lam)
    ds=min(p[0]['row']*ud/2+dp['row']*p[1]['s'],p[1]['row']*ud/2+dp['row']*p[0]['s'])
    dsq=min(q[0]['row']*ud/2+dq['row']*p[1]['s'],q[1]['row']*ud/2+dq['row']*p[0]['s'])
    lu=.25-sigmoid_weight(max(v['ell'] for v in p))+2e-15
    dr=min(ud*p[0]['rho']+p[1]['u']*lu*ds,ud*p[1]['rho']+p[0]['u']*lu*ds)
    lip=float(curvature_lipschitz(max(v['radius'] for v in p)))
    de=sum(v['e'] for v in p)
    for i,j in ((0,1),(1,0)):
        x,y=p[i],p[j]
        geom=ud*np.sqrt(x['k'])*x['wd']+y['u']*np.sqrt(dp['k'])*x['wd']
        weights=y['u']*np.sqrt(y['k'])*lip*(ds+.5*dp['row']*y['e'])
        feedback=.5*y['u']*np.sqrt(y['k'])*lip*x['row']
        den=1-x['eta']-y['e']*feedback
        if den>0:de=min(de,(dr+y['e']*(geom+weights))/den)
    increments=[]
    for i,j in ((0,1),(1,0)):
        x,y=p[i],p[j];xx,yy=q[i],q[j]
        dg=np.sqrt(dq['k'])*expit(x['ellq']+xx['row']*x['e'])
        dg+=np.sqrt(yy['k'])*.25*(dsq+.5*(xx['row']*de+dq['row']*y['e']))
        increments.append(dg*x['e']+y['grad']*de)
    eq=max(v['ellq'] for v in p)
    paired=min(increments)+eq**3*dsq/48
    separate=sum(v['grad']*v['e']+v['ellq']**4/192 for v in p)
    return dict(cap=float(min(paired,separate)+1e-10),paired=float(paired),
                separate=float(separate),p=p,q=q,difference_train=dp,difference_test=dq,de=float(de))

if __name__=='__main__':
    import population_experiment as pop
    out=[]
    for ci in (3,6):
        case=pop.s.CASES[ci];a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        row=cap(a[pools['training']],b[pools['training']],a[pools['comparison']],b[pools['comparison']],case['lam'])
        out.append(dict(case=case,**row));print(case['id'],row['cap'],flush=True)
    pop.save(ROOT/'public_cap.json',out)

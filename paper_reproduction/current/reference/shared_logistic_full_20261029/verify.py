"""Algebra and deterministic-bound checks independent of empirical conclusions."""
from study import *
from itertools import permutations,product


def main():
    rng=np.random.default_rng(20261029111);identity=[];fits=[];local=[]
    for k in range(15):
        n,r,d=3,3,4;lam=[.125,.5,2.][k%3];z=rng.normal(size=(9,r,d));z/=np.maximum(1,np.linalg.norm(z,axis=-1,keepdims=True))
        y=rng.integers(0,2,(9,r));t=rng.normal(size=d)*.15;tc=test_constants(z[5:],t);dc=cache_design(z,t,lam)
        ids=np.array([p for p in permutations(range(n)) for _ in range(4)]);tests=np.tile(np.arange(5,9),6)
        co=fast_coefficients(z,ids,tests,t,lam,tc,dc);nodes=np.stack([tests,ids[:,0],ids[:,1]],1)
        value=evaluate(co,(2*y[nodes]-1).reshape(len(ids),3*r)).mean()
        s=step(z[:n],y[:n],t,lam);target=quadratic_risk(z[5:],y[5:],t,s,tc)
        identity.append(abs(value-target));assert identity[-1]<2e-12
        batch,it=batch_fit(z[ids[:5]],y[ids[:5]],lam,t);single=np.stack([fit(z[v],y[v],lam,start=t)[0] for v in ids[:5]])
        fits.append(float(np.max(abs(batch-single))));assert fits[-1]<1e-8
        exact=single[0];norms=np.linalg.norm(s);e=np.linalg.norm(exact-t-s)
        G=np.einsum('nri,nrj->ij',z[:n],z[:n])/(n*r)
        J=np.einsum('nri,nr,nrj->ij',z[:n],abs(z[:n]@t),z[:n])/(n*r)
        a2=np.linalg.norm(J,2)/(16*lam);a3=np.linalg.norm(G,2)/(48*lam);ag=np.linalg.norm(G,2)/(12*np.sqrt(3)*lam)
        assert e<=a2*norms**2+a3*norms**3+2e-9
        assert e<=ag*norms**2+2e-9
        zz=z[5:];yy=y[5:];Kq=np.linalg.norm(np.einsum('nri,nrj->ij',zz,zz)/yy.size,2)
        Jq=np.linalg.norm(np.einsum('nri,nr,nrj->ij',zz,abs(zz@t),zz)/yy.size,2)
        grad=np.einsum('nrd,nr->d',zz,expit(zz@t)-yy)/yy.size;G0=np.linalg.norm(grad);hq=Kq/4
        bound=G0*(a2*norms**2+a3*norms**3)+hq*(a2*norms**3+a3*norms**4)+.5*hq*ag**2*norms**4+Jq*norms**3/48+Kq*norms**4/192
        error=abs(risk(zz,yy,exact)-target);local.append(dict(error=error,bound=bound));assert error<=bound+2e-9
    # Enumerate a nonconstant-feature, two-parameter, dependent-label experiment.
    n,r,d=2,2,2;lam=.7;zs=[rng.normal(size=(2,r,d)) for _ in range(2)]
    zs=[z/np.maximum(1,np.linalg.norm(z,axis=-1,keepdims=True)) for z in zs]
    py=np.array([[.10,.20,.30,.40],[.32,.14,.18,.36]]);px=np.array([.4,.6]);labels=np.array(list(product([0.,1.],repeat=2)))
    means=py@labels;sig=[np.einsum('ni,nj,n->ij',labels-means[x],labels-means[x],py[x]) for x in range(2)]
    total=[];meanpart=[];covpart=[];joint=[]
    for z in zs:
        t=rng.normal(size=d)*.12;tc=test_constants(np.repeat(z,[4,6],axis=0),t)
        testz=np.repeat(z,4,axis=0);testy=np.tile(labels,(2,1));pw=(px[:,None]*py).ravel()
        my=np.einsum('nrd,nr,n->d',testz,testy,pw)/r
        def qr(s):return tc['q']-my@t+(tc['b']-my)@s+.5*s@tc['C']@s
        exactsum=0.;msum=0.;csum=0.;jsum=0.
        for xx in product(range(2),repeat=n):
            xt=np.array(xx);zf=z[xt];weight=np.prod(px[xt]);mu=means[xt]
            sp=step(zf,mu,t,lam);msum+=weight*qr(sp)
            pp=expit(zf@t);H=np.einsum('nri,nr,nrj->ij',zf,pp*(1-pp),zf)/(n*r)+lam*np.eye(d)
            bs=np.linalg.solve(H,np.transpose(zf,(0,2,1)))/r
            csum+=weight*sum(np.trace(bs[i].T@tc['C']@bs[i]@sig[xt[i]]) for i in range(n))/(2*n*n)
            jsum+=weight*sum(np.trace(bs[i].T@tc['C']@bs[i]@(sig[xt[i]]-np.diag(np.diag(sig[xt[i]])))) for i in range(n))/(2*n*n)
            for yy in product(range(4),repeat=n):
                yidx=np.array(yy);prob=np.prod(py[xt,yidx]);ss=step(zf,labels[yidx],t,lam)
                exactsum+=weight*prob*qr(ss)
        total.append(exactsum);meanpart.append(msum);covpart.append(csum);joint.append(jsum)
    association_error=abs((total[0]-total[1])-(meanpart[0]-meanpart[1]+covpart[0]-covpart[1]))
    assert association_error<2e-12
    cap=(r-1)/(16*n*lam*lam*r);assert abs(joint[0]-joint[1])<=cap
    result=dict(identity_max_error=max(identity),batched_fit_max_parameter_error=max(fits),local_bound_checks=local,
                association_identity_error=association_error,joint_effect=joint[0]-joint[1],association_cap=cap,checks=15)
    save(ROOT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

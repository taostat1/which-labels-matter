"""Exhaustive small-design checks of the new sufficient condition."""
from run_validation import ROOT, base, np, paired_preconditioned
import itertools, time

def geometry(zA,zB,wA,wB,lam):
    N=len(zA);M=len(wA)
    op=lambda a:float(np.linalg.norm(a,2))
    k=max(op(zA),op(zB))/np.sqrt(N);q=max(op(wA),op(wB))/np.sqrt(M)
    u=op(zA-zB)/np.sqrt(N);v=op(wA-wB)/np.sqrt(M)
    T=k/(2*lam);p=T**3/48 if T<1e-4 else T/4-.5*np.tanh(T/2)
    aa=.25*np.tanh(T/2)**2;D=1/(2*lam)+k*k/(4*lam*lam);d=k*p/lam
    gamma=lam-k*k*aa
    if gamma<=0:return dict(valid=False,gamma=gamma)
    E=(k*d/2+p+k*aa*(T+k*D))/gamma
    CZ=d*q*q*(D+E)/4+q*E+p*q*D;CW=d*(1+q*T/4)+p*T
    return dict(valid=True,epsilon=CZ*u+CW*v,gamma=gamma,D=D,E=E,d=d,T=T,u=u,v=v)

def main():
    rng=np.random.default_rng(812047);checks=0;valid_designs=0;skipped=0;worst=0.;start=time.perf_counter()
    for index in range(32):
        lam=[.125,.25,.5,2][index%4]
        aa=rng.normal(size=(5,2));aa/=np.maximum(1,np.linalg.norm(aa,axis=1,keepdims=True))
        bb=aa+(0 if index%8==0 else .08)*rng.normal(size=aa.shape)
        bb/=np.maximum(1,np.linalg.norm(bb,axis=1,keepdims=True))
        za,wa=aa[:3],aa[3:];zb,wb=bb[:3],bb[3:]
        g=geometry(za,zb,wa,wb,lam)
        if not g['valid']:skipped+=1;continue
        valid_designs+=1
        H=[lam*np.eye(2)+z.T@z/12 for z in (za,zb)]
        for bits in itertools.product([0.,1.],repeat=5):
            yy=np.asarray(bits);pred=[];errors=[];ts=[];exact=[]
            for z,w,h in zip((za,zb),(wa,wb),H):
                theta,_=base.fit(z[:,None,:],yy[:3,None],lam)
                t=np.linalg.solve(h,z.T@(yy[:3]-.5)/3)
                assert np.linalg.norm(theta)<=g['T']+1e-8
                assert np.linalg.norm(theta-t)<=g['d']+1e-8
                scores=w@t;approx=np.mean(np.log(2)+(.5-yy[3:])*scores+scores*scores/8)
                actual=base.mean_loss(w[:,None,:],yy[3:,None],theta)
                errors.append(actual-approx);ts.append(t);exact.append(theta)
            diff=abs(errors[0]-errors[1]);worst=max(worst,diff/(g['epsilon']+1e-15))
            assert diff<=g['epsilon']+1e-8,(index,bits,diff,g)
            assert np.linalg.norm(ts[0]-ts[1])<=g['D']*g['u']+1e-8
            assert np.linalg.norm((exact[0]-ts[0])-(exact[1]-ts[1]))<=g['E']*g['u']+1e-8
            checks+=1
    out=dict(passed=True,checked_labelings=checks,valid_designs=valid_designs,
             nonpositive_margin_designs=skipped,max_error_to_bound_ratio=worst,seconds=time.perf_counter()-start)
    base.save(ROOT/'geometry_checks.json',out);print(out)

if __name__=='__main__':main()

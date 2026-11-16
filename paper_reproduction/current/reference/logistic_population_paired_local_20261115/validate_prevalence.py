from pilot_prevalence import *
import itertools

def main():
    rng=np.random.default_rng(202611153101);z=rng.normal(size=(4,2,3));z/=np.maximum(1,np.linalg.norm(z,axis=-1,keepdims=True))
    p=profile(z);lo=np.array([.25,.25]);hi=np.array([.75,.75]);L,U,U2=moment_box(p,lo,hi);num=0
    for xx in itertools.product((0,1),repeat=8):
        y=np.array(xx).reshape(4,2);prev=y.mean(0)
        if np.any(prev<lo) or np.any(prev>hi):continue
        b=np.einsum('ird,ir->id',z,y)/2;mu=b.mean(0)@p['rotation']
        assert np.all(mu>=L-1e-12) and np.all(mu<=U+1e-12)
        assert np.mean(np.sum(b*b,axis=1))<=U2+1e-12
        t=pop.s.fit(z,y,.125)[0]
        c=np.einsum('ird,ir->id',z,expit(z@t))/2+.125*t
        center=c.mean(0)@p['rotation'];upmu=np.sum(np.maximum(abs(center-L),abs(center-U))**2)
        closest=np.where(L>0,L,np.where(U<0,U,0.));varb=max(0,U2-closest@closest)
        varc=np.mean(np.sum((c-c.mean(0))**2,axis=1));upsecond=upmu+(np.sqrt(varc)+np.sqrt(varb))**2
        actual=c-b
        assert np.sum(actual.mean(0)**2)<=upmu+1e-12
        assert np.mean(np.sum(actual*actual,axis=1))<=upsecond+1e-12
        num+=1
    pop.save(ROOT/'prevalence_validation.json',dict(passed=True,completions=num))
    print('PASS',num,'pilot-profile and anchor-uniform moment checks')
if __name__=='__main__':main()

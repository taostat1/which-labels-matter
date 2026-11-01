from pathlib import Path
import os,sys,json,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'_deps'))
sys.path.insert(0,str(ROOT.parent/'shared_logistic_full_20261029'))
import study
import numpy as np
from sklearn.model_selection import ShuffleSplit

def bound(ztr,zte,lam,t=None):
    d=ztr.shape[-1];zt=ztr.reshape(-1,d);zq=zte.reshape(-1,d)
    t=np.zeros(d) if t is None else t
    pt=study.expit(zt@t);pq=study.expit(zq@t)
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

if __name__=='__main__':
    a,o,y,pools=study.load_domain('text');rng=np.random.default_rng(202610301)
    ix=rng.choice(pools['training'],4096,replace=False)
    aa=a[ix];oo=o[ix];yy=y[ix]
    split=list(ShuffleSplit(n_splits=4,train_size=64,test_size=128,random_state=202610302).split(aa))
    data=[]
    for case in study.CASES[:4]:
        bb=(1-case['alpha'])*aa+case['alpha']*oo
        sums=[sum(bound(z[tr],z[te],case['lam'])['epsilon'] for z in (aa,bb)) for tr,te in split]
        row=dict(id=case['id'],g=case['g'],allowances=sums)
        data.append(row)
    print(json.dumps(data,indent=2))
    study.save(ROOT/'design_diagnostics.json',data)

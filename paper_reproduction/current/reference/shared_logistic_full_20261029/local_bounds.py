"""Local logistic curvature and a paired, label-free coefficient range."""
from core import *

def geometry(z,t,lam):
    r=z.shape[1]; logits=np.einsum('ird,d->ir',z,t); p=expit(logits);w=p*(1-p)
    h=np.einsum('nri,nr,nrj->nij',z,w,z,optimize=True)/r
    # Gram eigenvalues can be found in the smaller r by r matrix.
    gram=np.einsum('nri,nsi->nrs',z,z,optimize=True)/r
    kp=float(np.linalg.eigvalsh(gram)[:,-1].max())
    q=np.einsum('nrd,nr->nd',z,w*logits-(p-.5),optimize=True)/r
    beta=float(np.linalg.norm(q,axis=1).max()/lam)
    return dict(h=h,K=kp,tau=float(abs(logits).max()),L=float(np.linalg.norm(z,axis=-1).mean(1).max()),beta=beta)

def paired_geometry(a,b,ta,tb,lam,training,comparison):
    ga=geometry(a[training],ta,lam);gb=geometry(b[training],tb,lam)
    ca=test_constants(a[comparison],ta);cb=test_constants(b[comparison],tb)
    Cnorm=[float(np.linalg.norm(x['C'],2)) for x in (ca,cb)]
    Kg=[];tauq=[];Lq=[]
    for z,t in ((a,ta),(b,tb)):
        zz=z[comparison];Kg.append(float(np.linalg.norm(np.einsum('nri,nrj->ij',zz,zz)/(len(zz)*zz.shape[1]),2)))
        tauq.append(float(abs(np.einsum('ird,d->ir',zz,t)).max()))
        Lq.append(float(np.linalg.norm(zz,axis=-1).mean(1).max()))
    Dh=float(np.linalg.norm(ga['h']-gb['h'],ord=2,axis=(1,2)).max())
    Dz=float(np.linalg.norm(a[training]-b[training],axis=-1).mean(1).max())
    D0=float(np.linalg.norm(a[comparison]-b[comparison],axis=-1).mean(1).max())
    SA,SB=ga['L']/lam,gb['L']/lam
    DB=Dz/lam+Dh*gb['L']/lam**2
    DC=float(np.linalg.norm(ca['C']-cb['C'],2))
    e=[float(np.linalg.norm(c['b_center']-c['C']@t)) for c,t in ((ca,ta),(cb,tb))]
    k=[float(c['q_center']-c['b_center']@t+.5*t@c['C']@t) for c,t in ((ca,ta),(cb,tb))]
    be=[ga['beta'],gb['beta']]
    constant=abs(k[0]-k[1])+sum(ej*bj+.5*cj*bj*bj for ej,bj,cj in zip(e,be,Cnorm))
    linear0=sum(lj*bj/2 for lj,bj in zip(Lq,be))
    linear1=sum(sj*(ej+cj*bj)/2 for sj,ej,cj,bj in zip((SA,SB),e,Cnorm,be))
    quad=(DB*Cnorm[0]*SA+SB*DC*SA+SB*Cnorm[1]*DB)/8
    testtrain=(Lq[0]*DB+D0*SB)/4
    result=dict(range=constant+linear0+linear1+quad+testtrain,
                parts=dict(constant=constant,linear0=linear0,linear1=linear1,quad=quad,testtrain=testtrain),
                Kp=[ga['K'],gb['K']],Kq=Kg,taup=[ga['tau'],gb['tau']],tauq=tauq,
                delta_H=Dh,delta_B=DB,delta_C=DC,beta=be,
                mean_row_train=[ga['L'],gb['L']],mean_row_test=Lq,
                mean_row_difference_train=Dz,mean_row_difference_test=D0)
    return result

def local_allowance(cal,grad,n,lam,Kp,Kq,taup,tauq):
    mu=np.sqrt(cal['mu2']);v=cal['second'];R=cal['B']+mu
    m2=((1-1/n)*cal['mu2']+v/n)/lam**2
    # An upper bound for the centered score variance is its uncentered second moment.
    m4=(mu**4+6*mu*mu*v/n+4*mu*R*v/n**2+3*v*v/n**2+R*R*v/n**3)/lam**4
    m4=min(m4,(cal['B']/lam)**2*m2)
    m3=np.sqrt(m2*m4)
    a2=Kp*taup/(16*lam);a3=Kp/(48*lam)
    ag=Kp/(12*np.sqrt(3)*lam);ht=Kq/4
    G=min(1.,np.sqrt(grad['mu2']))
    val=G*(a2*m2+a3*m3)+ht*(a2*m3+a3*m4)+ht*ag*ag*m4/2+Kq*(tauq*m3/48+m4/192)
    return dict(allowance=float(val),moments=[float(m2),float(m3),float(m4)],gradient_bound=float(G),a2=a2,a3=a3)

def matrix_parameters(z,t):
    """Exact finite-feature-law parameters for matrix Bernstein."""
    r=z.shape[1];logits=abs(np.einsum('ird,d->ir',z,t));out={}
    for name,weight in (('gram',np.ones(logits.shape)),('weighted',logits)):
        mat=np.einsum('nri,nr,nrj->nij',z,weight,z,optimize=True)/r
        avg=mat.mean(0);center=mat-avg
        variance=np.einsum('nij,njk->ik',center,center,optimize=True)/len(mat)
        out[name]=dict(mean=float(np.linalg.norm(avg,2)),
                       variance=float(np.linalg.norm(variance,2)),
                       range=float(np.linalg.norm(center,ord=2,axis=(1,2)).max()),
                       maximum=float(np.linalg.eigvalsh(mat)[:,-1].max()))
    return out

def localized_allowance(cal,grad,n,lam,par,Kq,Jq,d,eta):
    base=local_allowance(cal,grad,n,lam,1,Kq,1,1);m2,m3,m4=base['moments'];G=base['gradient_bound']
    log=np.log(2*d/eta);constants={}
    for name,p in par.items():
        u=p['range']*log/(3*n)
        constants[name]=min(p['maximum'],p['mean']+u+np.sqrt(2*p['variance']*log/n+u*u))
    k=constants['gram'];j=constants['weighted']
    a2=j/(16*lam);a3=k/(48*lam);ag=k/(12*np.sqrt(3)*lam);ht=Kq/4
    good=G*(a2*m2+a3*m3)+ht*(a2*m3+a3*m4)+ht*ag*ag*m4/2+(Jq*m3/48+Kq*m4/192)
    # Uniform old bound handles all feature designs outside the localized event.
    smax=cal['B']/lam;bad=(smax*smax/(2*lam)+smax**3/6)/(6*np.sqrt(3))
    total=good+2*eta*bad
    return dict(allowance=float(total),good=float(good),tail=float(2*eta*bad),
                constants=constants,eta=eta,moments=base['moments'],gradient_bound=G)

def diagnose():
    a,o,y,pools=data();split=json.loads((ROOT/'pilot_split.json').read_text());pilot=np.array(split['pilot'])
    n,lam,alpha=64,.125,.1;b=(1-alpha)*a+alpha*o
    ts=[fit(z[pilot],y[pilot],lam)[0] for z in (a,b)]
    tick=time.perf_counter();geo=paired_geometry(a,b,*ts,lam,pools['training'],pools['comparison'])
    pars=[matrix_parameters(z[pools['training']],t) for z,t in zip((a,b),ts)]
    Jq=[matrix_parameters(z[pools['comparison']],t)['weighted']['mean'] for z,t in zip((a,b),ts)]
    geomtime=time.perf_counter()-tick
    rng=np.random.default_rng(202610290039);traincal=rng.choice(pools['training'],32768);testcal=rng.choice(pools['comparison'],2048)
    # Three possible training-calibration looks; two candidates, each with training/test control.
    alpha_c=.01/12;rows=[]
    for m in (2048,8192,32768):
        vals=[]
        for j,z in enumerate((a,b)):
            cal=calibration(z[traincal[:m]],y[traincal[:m]],ts[j],lam,z[pools['training']],alpha_c)
            gr=calibration(z[testcal],y[testcal],ts[j],0.,z[pools['comparison']],alpha_c)
            val=local_allowance(cal,gr,n,lam,geo['Kp'][j],geo['Kq'][j],geo['taup'][j],geo['tauq'][j])
            val['global_allowance']=approx_bound(cal,n,lam)
            choices=[localized_allowance(cal,gr,n,lam,pars[j],geo['Kq'][j],Jq[j],a.shape[-1],eta) for eta in (1e-4,1e-5,1e-6,1e-7,1e-8)]
            val['localized']=min(choices,key=lambda v:v['allowance']);vals.append(val)
        row=dict(m=m,allowance=sum(v['allowance'] for v in vals),candidates=vals)
        row['localized_allowance']=sum(v['localized']['allowance'] for v in vals)
        rows.append(row);print(m,row['allowance'],row['localized_allowance'],[v['gradient_bound'] for v in vals],flush=True)
    save(ROOT/'local_diagnostic.json',dict(geometry=geo,feature_parameters=pars,Jq=Jq,geometry_seconds=geomtime,looks=rows,
         seed=202610290039,test_calibration=2048,error_budget=.01))
    print('paired range',geo['range'],'geometry time',geomtime)

if __name__=='__main__':diagnose()

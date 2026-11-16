"""History-measurable anchors and fresh conditional CP batches."""
from pathlib import Path
import sys,os,time,gc,json
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logistic_local_anchor_refinement_20261114'))
from local_core import candidate_geometry,certificate,sigmoid_weight
from anchor_core import fit,risk,LabelCache,ArrayLabelProvider
from run import load_domain,SOURCE,summarize_record,exact_fits
from whichlabels._intervals import endpoints,mapping_guard
import numpy as np
from scipy.special import expit
from scipy import sparse
from fast_bound import install
install()

CHECKS=[128,256,512,1024,2048,4096,8192,16384,32768]
def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def anchors_from_cache(a,b,plan,values,lam,mode):
    """No oracle access. Midpoint responses define a reference, not true labels."""
    r=a.shape[1];out=[]
    for tr,te in plan:
        keys=(tr[:,None]*r+np.arange(r)).ravel();known=np.isfinite(values[keys]);row=[]
        for z in (a,b):
            if mode=='zero':t=np.zeros(z.shape[-1])
            elif mode=='midpoint':
                yc=np.where(known,values[keys],.5)
                t=fit(z[tr],yc,lam)[0]
            elif mode=='observed':
                zz=z[tr].reshape(-1,z.shape[-1])
                t=fit(zz[known],values[keys][known],lam)[0] if known.any() else np.zeros(z.shape[-1])
            else:raise ValueError(mode)
            row.append(t)
        out.append(row)
    return out

def conditional_bound(a,b,plan,anchors,lam,values):
    r=a.shape[1];x=np.where(np.isfinite(values),2*values-1,np.nan);eps=[]
    for (tr,te),(ta,tb) in zip(plan,anchors):
        tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
        ga=candidate_geometry(a[tr],a[te],ta,lam,x[tk])
        gb=candidate_geometry(b[tr],b[te],tb,lam,x[tk])
        eps.append(certificate(ga,gb,lam,x[qk])[1]['epsilon'])
    return float(np.mean(eps))

def prepare(a,b,plan,anchors,lam,values):
    r=a.shape[1];total=len(values);x=np.where(np.isfinite(values),2*values-1,np.nan)
    known=np.isfinite(x);linear=np.zeros(total);constant=0.;rr=[];cc=[];vv=[];ee=[];geometries=[]
    for (tr,te),ts in zip(plan,anchors):
        tk=(tr[:,None]*r+np.arange(r)).ravel();qk=(te[:,None]*r+np.arange(r)).ravel()
        upper_i,upper_j=np.triu_indices(len(tk),1)
        quad=np.zeros((len(tk),len(tk)));cross=np.zeros((len(qk),len(tk)));gg=[]
        for sign,z,t in zip([1.,-1.],[a,b],ts):
            g=candidate_geometry(z[tr],z[te],t,lam,x[tk]);gg.append(g)
            L,C,c,q,vq=g['L'],g['C'],g['c'],g['q'],g['vq']
            constant+=sign*(np.mean(np.logaddexp(0,vq)-.5*vq)+g['b']@c+.5*c@C@c)/len(plan)
            np.add.at(linear,tk,sign*(L.T@(g['b']+C@c))/len(plan))
            np.add.at(linear,qk,-sign*(q@(t+c))/(2*len(qk)*len(plan)))
            quad+=sign*.5*(L.T@C@L);cross-=sign*(q@L)/(2*len(qk))
        constant+=float(np.trace(quad))/len(plan)
        left=np.minimum(tk[upper_i],tk[upper_j]);right=np.maximum(tk[upper_i],tk[upper_j])
        coeff=(quad[upper_i,upper_j]+quad[upper_j,upper_i])/len(plan)
        qleft=np.repeat(qk,len(tk));qright=np.tile(tk,len(qk))
        left=np.r_[left,np.minimum(qleft,qright)];right=np.r_[right,np.maximum(qleft,qright)]
        coeff=np.r_[coeff,cross.ravel()/len(plan)]
        # Substitute acquired labels before constructing the large sparse matrix.
        lk=known[left];rk=known[right];both=lk&rk
        constant+=float(np.sum(coeff[both]*x[left[both]]*x[right[both]]))
        mask=lk&~rk;np.add.at(linear,right[mask],coeff[mask]*x[left[mask]])
        mask=rk&~lk;np.add.at(linear,left[mask],coeff[mask]*x[right[mask]])
        remain=~lk&~rk;rr.append(left[remain].astype(np.int32));cc.append(right[remain].astype(np.int32));vv.append(coeff[remain])
        ee.append(certificate(*gg,lam,x[qk])[1]['epsilon'])
        geometries.append([(g['L'],g['c'],g['C'],g['b'],g['vq']) for g in gg])
    matrix=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(total,total)).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros();co=matrix.tocoo()
    constant+=float(linear[known]@x[known]);linear[known]=0;single=np.flatnonzero(linear!=0)
    left=np.r_[co.row,single].astype(np.int32);right=np.r_[co.col,np.full(len(single),-1)].astype(np.int32)
    coef=np.r_[co.data,linear[single]];mass=float(abs(coef).sum())
    cdf=np.cumsum(abs(coef)/mass) if mass else np.array([])
    if mass:cdf[-1]=1.
    assert not known[left].any() and not known[right[right>=0]].any()
    return dict(constant=constant,left=left,right=right,coef=coef,mass=mass,cdf=cdf,
        epsilon=float(np.mean(ee)),geometry=geometries,anchors=anchors)

def decision(lo,hi,g):
    if lo>hi:return 'conflict'
    return 'A' if hi<=g else 'B' if lo>=-g else 'unresolved'

def run_path(a,b,plan,lam,g,cache,seed,mode,initial_epsilon=None):
    rng=np.random.default_rng(seed);history=[];L=-np.inf;U=np.inf;n=0;choice='unresolved'
    audit_snapshots=[];empty_intersection=False;zero_eps=float('inf') if initial_epsilon is None else initial_epsilon
    for phase in range(len(CHECKS)+1):
        anchors=anchors_from_cache(a,b,plan,cache._values,lam,mode)
        poly=prepare(a,b,plan,anchors,lam,cache._values)
        c,mass,eps=poly['constant'],poly['mass'],poly['epsilon']
        if mode=='zero' or phase==0:eps=min(eps,zero_eps)
        if mode=='zero':zero_eps=eps
        L=max(L,c-mass-eps-2e-9);U=min(U,c+mass+eps+2e-9)
        choice=decision(L,U,g)
        h=dict(phase=phase,draws_before=n,labels_before=cache.count,mass=mass,
            epsilon_before=eps,public_interval=[c-mass-eps,c+mass+eps],intersection_before=[L,U],
            mean_anchor_norm=float(np.mean([np.linalg.norm(t) for row in anchors for t in row])))
        audit_snapshots.append(dict(anchors=anchors,epsilon=eps,values=cache._values.copy(),geometry=poly['geometry']))
        if choice!='unresolved' or phase==len(CHECKS):
            h.update(choice=choice,draws=n,labels=cache.count,epsilon=eps,interval=[L,U],stage='public')
            history.append(h);del poly;gc.collect();break
        batch=CHECKS[phase]-n
        assert mass>0
        drift=mapping_guard(poly['coef'],poly['cdf'],mass)
        guard=2e-9+drift
        h['mapping_error_bound']=drift
        idx=np.searchsorted(poly['cdf'],rng.random(batch),side='right')
        xl=2*cache.query_keys(poly['left'][idx],'comparison')-1
        kr=poly['right'][idx];pair=kr>=0;xr=np.ones(batch)
        xr[pair]=2*cache.query_keys(kr[pair],'comparison')-1
        signs=np.sign(poly['coef'][idx])*xl*xr;assert np.isin(signs,[-1.,1.]).all()
        k=int(np.count_nonzero(signs>0));pl,pu=endpoints(k,batch,.05/len(CHECKS));n+=batch
        eps=min(eps,conditional_bound(a,b,plan,anchors,lam,cache._values))
        lo=c+mass*(2*pl-1)-eps-guard;hi=c+mass*(2*pu-1)+eps+guard
        L=max(L,lo);U=min(U,hi);choice=decision(L,U,g)
        audit_snapshots[-1]['epsilon']=eps
        h.update(batch=batch,successes=k,alpha=.05/len(CHECKS),draws=n,labels=cache.count,
            epsilon=eps,phase_interval=[lo,hi],interval=[L,U],choice=choice,stage='query')
        history.append(h)
        print(mode,'phase',phase,'labels',cache.count,'eps',round(eps,6),'mass',round(mass,5),choice,flush=True)
        del poly;gc.collect()
        if choice!='unresolved':break
    if choice=='conflict':empty_intersection=True;choice='unresolved'
    return dict(choice=choice,interval=[L,U],draws=n,history=history,empty_intersection=empty_intersection),audit_snapshots

def audit_snapshots(a,b,y,plan,lam,snapshots,target):
    errors=[]
    for snap in snapshots:
        differences=[]
        for si,(tr,te) in enumerate(plan):
            vals=[]
            for j,z in enumerate((a,b)):
                L,c,C,bb,vq=snap['geometry'][si][j]
                s=c+L@(2*y[tr]-1).ravel();q=z[te].reshape(-1,z.shape[-1]);ye=y[te].ravel()
                vals.append(float(np.mean(np.logaddexp(0,vq)-ye*vq)+(bb-q.T@(ye-.5)/len(ye))@s+.5*s@C@s))
            differences.append(vals[0]-vals[1])
        error=abs(target-float(np.mean(differences)))
        assert error<=snap['epsilon']+2e-8,(error,snap['epsilon'])
        errors.append(error)
    return errors

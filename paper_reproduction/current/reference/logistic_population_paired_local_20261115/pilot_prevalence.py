"""Anchor-uniform gradient moments from pilot prevalences and public profiles."""
from paired_local import *
import population_experiment as pop
from scipy.stats import beta
import json,time

def profile(z):
    # Public rotation is used for a tighter coordinate box.
    d=z.shape[-1];r=z.shape[1]
    gram=np.einsum('ird,ire->de',z,z)/(len(z)*r)
    _,V=np.linalg.eigh(gram);zz=z@V
    vals=np.concatenate([zz,np.sum(z*z,axis=-1)[...,None]],axis=-1)
    ordered=np.sort(vals,axis=0)
    prefix=np.concatenate([np.zeros((1,r,d+1)),np.cumsum(ordered,axis=0)],axis=0)
    return dict(rotation=V,ordered=ordered,prefix=prefix,n=len(z),r=r)

def at_count(p,t):
    n=p['n'];t=np.clip(t,0,n);i=np.floor(t).astype(int);frac=t-i
    ar=np.arange(p['r'])[:,None];col=np.arange(p['prefix'].shape[-1])[None,:]
    return (p['prefix'][i,ar,col]+frac*p['ordered'][np.minimum(i,n-1),ar,col])/n

def moment_box(p,lo,hi):
    n=p['n'];vals=p['ordered']
    # Minimization takes the most negative values, respecting count bounds.
    count=np.sum(vals<0,axis=0)
    t=np.clip(count,n*lo[:,None],n*hi[:,None]);lower=at_count(p,t)
    count=np.sum(vals>0,axis=0)
    t=np.clip(count,n*lo[:,None],n*hi[:,None]);upper=p['prefix'][-1]/n-at_count(p,n-t)
    return lower.mean(0)[:-1],upper.mean(0)[:-1],float(upper.mean(0)[-1])

def main():
    prior=json.loads((ROOT/'pilot_moments.json').read_text())['records'];rows=[]
    for ci in (3,6):
        case=pop.s.CASES[ci];a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        profiles=[profile(z[pools['training']]) for z in (a,b)]
        for group in range(6,10):
            old=json.loads((pop.ROOT/'population_results'/f'case_{ci:02d}_{group:02d}.json').read_text('utf-8'))
            pilot=np.array(old['initialization']['pilot_ids']);cache=pop.p.Cache(y);yp=cache.objects(pilot,'pilot')
            n,r=yp.shape;success=yp.sum(0);alpha=.005/r
            lo=np.where(success>0,beta.ppf(alpha/2,success,n-success+1),0.)
            hi=np.where(success<n,beta.ppf(1-alpha/2,success+1,n-success),1.)
            boxes=[moment_box(p,lo,hi) for p in profiles]
            anchors=[pop.s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
            for name,ts in [('zero',[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]),('pilot32',anchors)]:
                saved=next(v for v in prior if v['case']==case['id'] and v['group']==group and v['anchor']==name)
                par=pop.p.parameters(a,b,pools,ts,case['lam']);ds=[];mom=[]
                for j,z in enumerate((a,b)):
                    tr=z[pools['training']];t=ts[j];V=profiles[j]['rotation'];L,U,U2=boxes[j]
                    c=np.einsum('ird,ir->id',tr,expit(tr@t))/r+case['lam']*t
                    center=c.mean(0)@V;oldmoment=saved['moments'][j]['old']
                    mu2=min(oldmoment['mu2'],float(np.maximum(abs(center-L),abs(center-U))@np.maximum(abs(center-L),abs(center-U))))
                    closest=np.where(L>0,L,np.where(U<0,U,0.));varb=max(0,U2-float(closest@closest))
                    varc=float(np.mean(np.sum((c-c.mean(0))**2,axis=1)))
                    second=min(oldmoment['second'],mu2+(np.sqrt(varb)+np.sqrt(varc))**2)
                    cal=dict(mu2=mu2,second=second,B=oldmoment['B'])
                    # Evaluation-pool labels were not bought by this training pilot.
                    # Its exact deterministic envelope is unchanged.
                    from incremental import EnvelopeTemplate,EnvelopeState
                    te=EnvelopeTemplate(z[pools['comparison']],t,0.)
                    test=EnvelopeState(te).update(np.full((len(pools['comparison']),r),np.nan))
                    ds.append(pop.from_moments(cal,test,z,case,par,j,pop.s));mom.append(dict(old=oldmoment,new=cal))
                row=dict(case=case['id'],group=group,anchor=name,old_epsilon=saved['old_epsilon'],
                         new_epsilon=float(sum(v['used'] for v in ds)),moments=mom,
                         pilot_labels=cache.count,calibration_labels=0,prevalence_lower=lo,prevalence_upper=hi)
                rows.append(row)
            pop.save(ROOT/'pilot_prevalence.json',dict(records=rows,confidence_error=.005))
            print(ci,group,[(v['anchor'],v['old_epsilon'],v['new_epsilon']) for v in rows[-2:]],flush=True)
if __name__=='__main__':main()

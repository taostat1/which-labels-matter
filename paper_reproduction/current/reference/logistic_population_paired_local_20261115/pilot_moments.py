"""Reuse an iid pilot for all anchors via anchor-free label-feature moments.

The fitted anchor may depend on every pilot response. Confidence events concern
only b(X,Y)=(2r)^-1 sum z(X)(2Y-1), and therefore hold for every anchor at once.
No independent calibration labels are requested.
"""
from paired_local import *
import population_experiment as pop
from incremental import EnvelopeTemplate
import json,time

def eb(x,width,alpha):
    n=len(x)
    return np.sqrt(2*np.var(x,axis=0,ddof=1)*np.log(2/alpha)/n)+7*width*np.log(2/alpha)/(3*(n-1))

def summarize(z,pilot_z,pilot_y,alpha):
    n,r,d=pilot_z.shape
    values=np.einsum('ird,ir->id',pilot_z,2*pilot_y-1)/(2*r)
    # Rotate using public features only. Simultaneity then includes fitted anchors.
    gram=np.einsum('ird,ire->de',z,z)/(len(z)*r)
    _,V=np.linalg.eigh(gram);zz=z@V;vv=values@V
    coord=abs(zz).mean(1).max(0)/2
    m=vv.mean(0);sub=alpha/(2*d+2)
    radius=np.minimum(coord*np.sqrt(2*np.log(2/sub)/n),eb(vv,2*coord,sub/2))
    # EB is two-sided here; its sub/2 allocation and the Hoeffding allocation
    # sum to 2*sub per coordinate. Two sub-events remain for norm moments.
    raw=EnvelopeTemplate(z,np.zeros(d),0.)
    B=float(np.sqrt(raw.maximum.max()))
    radial=B*(1+np.sqrt(2*np.log(1/sub)))/np.sqrt(n)
    squares=np.sum(values**2,axis=1)
    second=min(B*B,float(squares.mean()+eb(squares,B*B,sub)))
    lower=np.maximum(abs(m)-radius,0)
    mu_lower=max(float(lower@lower),max(0,float(np.linalg.norm(m))-radial)**2)
    variance=max(0,second-mu_lower)
    return dict(mean=m,radius=radius,rotation=V,B=B,radial=radial,
                second=second,variance=variance,alpha=alpha,pilot_size=n)

def at_anchor(z,theta,lam,stats,deterministic):
    r=z.shape[1]
    c=np.einsum('ird,ir->id',z,expit(z@theta)-.5)/r+lam*theta
    center=c.mean(0);diff=center@stats['rotation']-stats['mean']
    squared_box=float(np.sum((abs(diff)+stats['radius'])**2))
    squared_ball=(float(np.linalg.norm(diff))+stats['radial'])**2
    mu2=min(deterministic['mu2'],squared_box,squared_ball)
    varc=float(np.mean(np.sum((c-center)**2,axis=1)))
    second=min(deterministic['second'],mu2+(np.sqrt(varc)+np.sqrt(stats['variance']))**2)
    return dict(mu2=mu2,second=second,B=deterministic['B'])

def main():
    path=ROOT/'pilot_moments.json'
    rows=json.loads(path.read_text())['records'] if path.exists() else []
    tick=time.perf_counter()
    for ci in (3,6):
        case=pop.s.CASES[ci];a,o,y,pools=pop.s.load_domain(case['domain']);b=(1-case['alpha'])*a+case['alpha']*o
        for group in range(6,10):
            if any(row['case']==case['id'] and row['group']==group and row['anchor']=='pilot32' for row in rows):continue
            old=json.loads((pop.ROOT/'population_results'/f'case_{ci:02d}_{group:02d}.json').read_text('utf-8'))
            pilot=np.array(old['initialization']['pilot_ids']);cache=pop.p.Cache(y);yp=cache.objects(pilot,'pilot')
            ts=[pop.s.fit(z[pilot],yp,case['lam'])[0] for z in (a,b)]
            stats=[summarize(z[pools['training']],z[pilot],yp,.005/3) for z in (a,b,a-b)]
            for name,anchors in [('zero',[np.zeros(a.shape[-1]),np.zeros(b.shape[-1])]),('pilot32',ts)]:
                par=pop.p.parameters(a,b,pools,anchors,case['lam'])
                template=pop.BoundTemplate(a,b,pools,anchors,case);states=template.state()
                # Independent pilot draws may repeat; no extra purchase occurs.
                oldeps,olddetails=template.update(states,cache.values,par,pop.s,pop.from_moments)
                details=[];mom=[]
                for j,z in enumerate((a,b)):
                    train=states[j][0].update(cache.values.reshape(-1,y.shape[1])[pools['training']])
                    test=states[j][1].update(cache.values.reshape(-1,y.shape[1])[pools['comparison']])
                    new=at_anchor(z[pools['training']],anchors[j],case['lam'],stats[j],train)
                    details.append(pop.from_moments(new,test,z,case,par,j,pop.s));mom.append(dict(old=train,new=new))
                rows.append(dict(case=case['id'],group=group,anchor=name,pilot_labels=cache.count,calibration_labels=0,
                                 old_epsilon=oldeps,new_epsilon=float(sum(v['used'] for v in details)),moments=mom,
                                 difference_mean_radius=stats[2]['radial']))
            pop.save(ROOT/'pilot_moments.json',dict(records=rows,seconds=time.perf_counter()-tick))
            print(ci,group,[(q['anchor'],q['old_epsilon'],q['new_epsilon']) for q in rows[-2:]],flush=True)
if __name__=='__main__':main()

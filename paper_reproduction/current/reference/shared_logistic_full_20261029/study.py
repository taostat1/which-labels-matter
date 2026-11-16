"""Fresh logistic comparisons, with paid planning and auditable label reuse."""
from core import *
from local_bounds import paired_geometry, matrix_parameters, localized_allowance
import argparse, hashlib

CASES=[
 dict(id='text_strong_separated',domain='text',n=64,lam=2.,alpha=1.,g=.005),
 dict(id='text_strong_nearby',domain='text',n=64,lam=2.,alpha=.1,g=.005),
 dict(id='text_moderate',domain='text',n=64,lam=.5,alpha=1.,g=.02),
 dict(id='text_weak_nearby',domain='text',n=64,lam=.125,alpha=.1,g=.005),
 dict(id='image_strong',domain='image',n=128,lam=2.,alpha=1.,g=.005),
 dict(id='image_moderate',domain='image',n=128,lam=.5,alpha=1.,g=.005),
 dict(id='image_weak_nearby',domain='image',n=128,lam=.125,alpha=.05,g=.005),
 dict(id='image_backbone',domain='backbone',n=128,lam=.5,alpha=1.,g=.005)]
METHODS=('weighted','complete','refit')
SEED=202610291702

def load_domain(domain):
    data_root=Path(os.environ.get('LOGISTIC_DATA_ROOT',str(ROOT.parent)))
    if domain=='text':source=data_root/'goemotions_comparison_20261028'
    elif domain=='backbone':source=data_root/'cross_backbone_20261028'
    else:source=data_root/'ade20k_decision_20261026'
    f=np.load(source/'prepared/features_public.npz');lab=np.load(source/'prepared/labels_oracle.npz')
    assert np.array_equal(f['ids'],lab['ids'])
    if domain=='image':
        rng=np.random.default_rng(2026102608);proj=rng.normal(size=(64,8))/8;codes=rng.choice([-1.,1.],(6,16));zs=[]
        for chosen in ((0,1),(2,3)):
            glob=np.concatenate([f['global_heads'][:,j]@proj for j in chosen],-1)
            z=np.concatenate([glob[:,None,:]*codes[None],np.broadcast_to(.25*np.eye(6),(len(glob),6,6))],-1).astype(float)
            z/=np.maximum(1,np.linalg.norm(z,axis=-1,keepdims=True));zs.append(z)
        a,o=zs;y=(lab['fractions']>=.01).astype(float)
    else:a,o=f['a'],f['other'];y=(lab['y'].astype(float)+1)/2
    pools={k:np.flatnonzero(f['roles']==k) for k in ('training','comparison','audit','pilot')}
    return a,o,y,pools

def cache_design(z,t,lam):
    r=z.shape[1];v=np.einsum('ird,d->ir',z,t);p=expit(v)
    return dict(h=np.einsum('nri,nr,nrj->nij',z,p*(1-p),z,optimize=True)/r,
                u=np.einsum('nrd,nr->nd',z,.5-p,optimize=True)/r-lam*t)

def fast_coefficients(z,ids,tests,t,lam,tc,dc):
    """Average over training positions 1 and 2 before acquiring labels."""
    bs,n=ids.shape;r,d=z.shape[1:];H=dc['h'][ids].mean(1)+lam*np.eye(d)
    inv=np.linalg.solve(H,np.broadcast_to(np.eye(d),(bs,d,d)))
    c=np.einsum('bij,bj->bi',inv,dc['u'][ids].mean(1));cc=c@tc['C']
    B=np.einsum('bij,bnrj->bnir',inv,z[ids[:,:2]],optimize=True)/r
    constant=tc['q_center']+c@tc['b_center']+.5*np.einsum('bi,bi->b',c,cc)
    linear=np.zeros((bs,3*r));linear[:,:r]=-np.einsum('brd,bd->br',z[tests],t+c)/(2*r)
    mat=np.zeros((bs,3*r,3*r))
    for k in (0,1):
        sl=slice((k+1)*r,(k+2)*r);bk=B[:,k]
        linear[:,sl]=np.einsum('bdr,bd->br',bk,tc['b_center']+cc)/4
        D=np.einsum('bdi,de,bej->bij',bk,tc['C'],bk,optimize=True)/(16*n)
        constant+=np.trace(D,axis1=1,axis2=2)
        mat[:,sl,sl]+=D
        F=-np.einsum('bid,bdj->bij',z[tests],bk,optimize=True)/(8*r)
        mat[:,:r,sl]+=F
    mat[:,r:2*r,2*r:]+=np.einsum('bdi,de,bej->bij',B[:,0],tc['C'],B[:,1],optimize=True)*(n-1)/(8*n)
    l,h=np.triu_indices(3*r,1);w=mat[:,l,h]+mat[:,h,l]
    return constant,linear,w,l,h

def batch_fit(z,y,lam,start,tol=1e-10):
    bs,n,r,d=z.shape;zz=z.reshape(bs,n*r,d);yy=y.reshape(bs,n*r)
    t=np.broadcast_to(start,(bs,d)).copy();iterations=0
    for it in range(25):
        v=(zz@t[...,None])[...,0];p=expit(v)
        g=(zz.transpose(0,2,1)@(p-yy)[...,None])[...,0]/(n*r)+lam*t
        if np.max(np.linalg.norm(g,axis=1))<=tol:break
        H=zz.transpose(0,2,1)@(zz*(p*(1-p))[...,None])/(n*r)+lam*np.eye(d)
        step_=np.linalg.solve(H,g[...,None])[...,0]
        old=(np.logaddexp(0,v)-yy*v).mean(1)+lam*(t*t).sum(1)/2;rate=np.ones(bs)
        for _ in range(25):
            tt=t-rate[:,None]*step_;vv=(zz@tt[...,None])[...,0]
            obj=(np.logaddexp(0,vv)-yy*vv).mean(1)+lam*(tt*tt).sum(1)/2
            bad=obj>old-1e-4*rate*(g*step_).sum(1)+1e-14
            if not bad.any():break
            rate[bad]*=.5
        t=tt;iterations+=1
    else:raise RuntimeError('batched Newton residual')
    return t,iterations

def setup(case,a,o,y,pools,group=0):
    tick=time.perf_counter();ci=next(i for i,c in enumerate(CASES) if c['id']==case['id'])
    rng=np.random.default_rng(SEED+ci*100000+group*1000);b=(1-case['alpha'])*a+case['alpha']*o
    pilot=rng.choice(pools['training'],512);traincal=rng.choice(pools['training'],32768);testcal=rng.choice(pools['comparison'],2048)
    anchor_tick=time.perf_counter();ts=[fit(z[pilot],y[pilot],case['lam'])[0] for z in (a,b)]
    anchor_seconds=time.perf_counter()-anchor_tick
    tcs=[test_constants(z[pools['comparison']],t) for z,t in zip((a,b),ts)]
    geo=paired_geometry(a,b,*ts,case['lam'],pools['training'],pools['comparison'])
    pars=[matrix_parameters(z[pools['training']],t) for z,t in zip((a,b),ts)]
    jq=[matrix_parameters(z[pools['comparison']],t)['weighted']['mean'] for z,t in zip((a,b),ts)]
    looks=[];chosen=None;allcal=[]
    for m in (2048,8192,32768):
        vals=[];cals=[]
        for j,z in enumerate((a,b)):
            cal=calibration(z[traincal[:m]],y[traincal[:m]],ts[j],case['lam'],z[pools['training']],.01/12)
            grad=calibration(z[testcal],y[testcal],ts[j],0.,z[pools['comparison']],.01/12)
            lc=min([localized_allowance(cal,grad,case['n'],case['lam'],pars[j],geo['Kq'][j],jq[j],a.shape[-1],eta)
                    for eta in (1e-4,1e-5,1e-6,1e-7,1e-8)],key=lambda v:v['allowance'])
            lc['global']=approx_bound(cal,case['n'],case['lam']);lc['used']=min(lc['global'],lc['allowance'])
            vals.append(lc);cals.append(cal)
        eps=sum(v['used'] for v in vals);looks.append(dict(m=m,epsilon=eps,details=vals));allcal=cals
        if eps<case['g']:chosen=m;break
    M=min(geo['range'],sum(centered_bound(t,case['lam']) for t in ts))
    TB=geo['mean_row_train'][1]/(2*case['lam'])
    R=min(1/case['lam'],geo['mean_row_test'][0]*geo['mean_row_difference_train']/case['lam']*(1+TB/4)+geo['mean_row_difference_test']*TB)
    dc=[cache_design(z,t,case['lam']) for z,t in zip((a,b),ts)]
    initial=np.unique((np.r_[pilot,traincal[:m],testcal][:,None]*y.shape[1]+np.arange(y.shape[1])).ravel())
    st=dict(group=group,epsilon=eps,M=M,R=R,accepted=chosen is not None,m=m,initial=initial.tolist(),pilot=pilot.tolist(),
            seconds=time.perf_counter()-tick,anchor_seconds=anchor_seconds,looks=looks,geometry=geo,calibrations=allcal)
    return b,ts,tcs,dc,st

class Ledger:
    def __init__(self,y,initial=()):
        self.y=y;self.r=y.shape[1];self.seen=np.zeros(y.size,bool);self.seen[np.asarray(initial,int)]=True;self.requests=0
    def query(self,ids,coords=None):
        if coords is None:keys=(np.asarray(ids)[...,None]*self.r+np.arange(self.r)).reshape(-1)
        else:keys=(np.asarray(ids)*self.r+coords).reshape(-1)
        self.seen[keys]=True;self.requests+=len(keys)
        result=2*self.y.reshape(-1)[keys]-1
        return result.reshape(np.shape(ids)+(self.r,)) if coords is None else result.reshape(np.shape(ids))
    def counts(self):return int(self.seen.sum()),int(self.seen.reshape(-1,self.r).any(1).sum())

def batch_values(ids,tests,rs,a,b,y,ts,tcs,dc,case,st,methods,ledgers):
    bs=len(tests);r=y.shape[1];nodes=np.stack([tests,ids[:,0],ids[:,1]],1);out={};work={};iteration=0
    if any(m in methods for m in ('weighted','complete')):
        tick=time.perf_counter();aa=fast_coefficients(a,ids,tests,ts[0],case['lam'],tcs[0],dc[0]);bb=fast_coefficients(b,ids,tests,ts[1],case['lam'],tcs[1],dc[1])
        c,v,w=[aa[k]-bb[k] for k in range(3)];l,h=aa[3:]
        assert np.max(abs(c)+abs(v).sum(1)+abs(w).sum(1))<=st['M']+1e-9
        common=time.perf_counter()-tick
        for method in ('weighted','complete'):
            if method not in methods:continue
            tick=time.perf_counter();oo=ledgers[method]
            if method=='complete':value=evaluate((c,v,w,l,h),oo.query(nodes).reshape(bs,3*r))
            else:
                coef=np.concatenate([v,w],1);mass=abs(coef).sum(1)
                ix=(np.cumsum(abs(coef),1)<rs[:,None]*mass[:,None]).sum(1).clip(max=coef.shape[1]-1)
                li=np.r_[np.arange(3*r),l][ix];hi=np.r_[np.arange(3*r),h][ix]
                yl=oo.query(nodes[np.arange(bs),li//r],li%r);yr=np.ones(bs);pair=ix>=3*r
                yr[pair]=oo.query(nodes[np.arange(bs)[pair],hi[pair]//r],hi[pair]%r)
                value=c+np.sign(coef[np.arange(bs),ix])*mass*yl*yr
            out[method]=value;work[method]=common+time.perf_counter()-tick
    if 'refit' in methods:
        tick=time.perf_counter();oo=ledgers['refit'];yy=(oo.query(ids)+1)/2;y0=(oo.query(tests)+1)/2
        ta,ia=batch_fit(a[ids],yy,case['lam'],ts[0]);tb,ib=batch_fit(b[ids],yy,case['lam'],ts[1]);iteration=ia+ib
        va=np.einsum('brd,bd->br',a[tests],ta);vb=np.einsum('brd,bd->br',b[tests],tb)
        val=(np.logaddexp(0,va)-np.logaddexp(0,vb)-y0*(va-vb)).mean(1)
        assert max(abs(val))<=st['R']+1e-9
        out['refit']=val;work['refit']=time.perf_counter()-tick
    return out,work,iteration

def interval(su,sq,num,M,eps,delta,checks):
    variance=max(0.,(sq-su*su/num)/(num-1));t=np.log(4*checks/delta)
    h=np.sqrt(2*variance*t/num)+14*M*t/(3*(num-1))+eps
    return su/num-h,su/num+h,h

def decision(lo,hi,g):return 'A' if hi<=g else 'B' if lo>=-g else 'unresolved'

def forecast(case,st,pilot,cache_size):
    K=8;h=case['g']+max(0,abs(pilot['refit']['mean'])-pilot['refit']['se']);out={}
    for m in METHODS:
        eps=0 if m=='refit' else st['epsilon'];M=st['R'] if m=='refit' else st['M'];v=pilot[m]['variance']
        chosen=None
        for q in [256,512,1024,2048,4096,8192,16384,32768]:
            t=np.log(4*K/(.05 if m=='refit' else .04))
            if eps+np.sqrt(2*v*t/q)+14*M*t/(3*(q-1))<=h:chosen=q;break
        if m!='refit' and not st['accepted']:chosen=None
        # The acquisition objective counts distinct newly requested labels.
        labels=float('inf') if chosen is None else chosen*pilot[m]['new_labels']/pilot[m]['blocks']
        out[m]=dict(blocks=chosen,labels=labels)
    selected=min(METHODS,key=lambda m:out[m]['labels'])
    if not np.isfinite(out[selected]['labels']):selected='refit'
    return dict(selected=selected,radius=h,plans=out)

def run_trial(case,cfg,rep,a,b,y,pools,ts,tcs,dc,st):
    ci=next(i for i,c in enumerate(CASES) if c['id']==case['id']);rng=np.random.default_rng(SEED+7000000+ci*100000+rep)
    n=case['n'];r=y.shape[1];K=len(cfg['checkpoints']);initial=st['initial']
    # Planning is independent of later comparison blocks; it is paid in every adaptive rule.
    pl={m:Ledger(y,initial) for m in METHODS};ids=rng.choice(pools['training'],(cfg['planning_blocks'],n));test=rng.choice(pools['comparison'],cfg['planning_blocks'])
    vals,secs,it=batch_values(ids,test,rng.random(len(test)),a,b,y,ts,tcs,dc,case,st,METHODS,pl)
    pstats={};init_count=Ledger(y,initial).counts()
    for m in METHODS:
        lab,doc=pl[m].counts();v=vals[m]
        pstats[m]=dict(mean=float(v.mean()),se=float(v.std(ddof=1)/np.sqrt(len(v))),variance=float(v.var(ddof=1)),seconds=secs[m],blocks=len(v),new_labels=lab-init_count[0],new_docs=doc-init_count[1])
    bought=np.flatnonzero(np.logical_or.reduce([pl[m].seen for m in METHODS]));pplan=forecast(case,st,pstats,len(bought))
    prepseconds=st['seconds']+sum(secs.values());prep=dict(seconds=prepseconds,labels=len(bought),documents=int(np.unique(bought//r).size),plan=pplan,statistics=pstats)
    ledgers={m:Ledger(y,bought) for m in METHODS};sums={m:[0.,0.,0] for m in METHODS};running={m:0. for m in METHODS};rows=[];batches=[]
    active=set(METHODS if st['accepted'] else ['refit']);final_full={};final_half={};cp=cfg['checkpoints']
    for start in range(0,cp[-1],cfg['batch']):
        if not active:break
        size=min(cfg['batch'],cp[-1]-start);ids=rng.choice(pools['training'],(size,n));test=rng.choice(pools['comparison'],size)
        before={m:ledgers[m].seen.copy() for m in active}
        vs,ss,ii=batch_values(ids,test,rng.random(size),a,b,y,ts,tcs,dc,case,st,active,ledgers)
        blocks=start+size
        batch=dict(blocks=blocks,methods={})
        for m in active:
            v=vs[m];sums[m][0]+=float(v.sum());sums[m][1]+=float(v@v);sums[m][2]+=size;running[m]+=ss[m]
            # Compact recorded requests permit the actual shared cache to be reconstructed.
            new=np.flatnonzero(ledgers[m].seen & ~before[m]);lab,doc=ledgers[m].counts()
            batch['methods'][m]=dict(new=new.tolist(),seconds=ss[m],labels=lab,documents=doc)
        batches.append(batch)
        if blocks not in cp:continue
        for m in list(active):
            su,sq,num=sums[m];eps=(2e-10/case['lam']) if m=='refit' else st['epsilon'];M=st['R'] if m=='refit' else st['M']
            valsout={}
            for mode,delta in [('single',.05 if m=='refit' else .04),('protected',.02)]:
                lo,hi,rad=interval(su,sq,num,M,eps,delta,K);dec=decision(lo,hi,case['g']);valsout[mode]=dict(lower=float(lo),upper=float(hi),radius=float(rad),decision=dec,both=bool(lo>=-case['g'] and hi<=case['g']))
            lab,doc=ledgers[m].counts();row=dict(method=m,blocks=blocks,mean=su/num,labels=lab,documents=doc,requests=ledgers[m].requests,seconds=running[m],**valsout);rows.append(row)
            if m not in final_full and valsout['single']['decision']!='unresolved':final_full[m]=row
            if m not in final_half and valsout['protected']['decision']!='unresolved':final_half[m]=row;active.remove(m)
    # One shared, fresh final-training draw; each method is charged for all labels absent at its own stopping prefix.
    fi=np.random.default_rng(SEED+9000000+ci*100000+rep).choice(pools['training'],n)
    tick=time.perf_counter();finalfits={}
    for name,z in [('A',a),('B',b)]:
        tm=time.perf_counter();_,info=fit(z[fi],y[fi],case['lam']);finalfits[name]=dict(seconds=time.perf_counter()-tm,iterations=info['iterations'])
    return dict(case=case,rep=rep,group=st['group'],preparation=prep,initial=bought.tolist(),bounds={k:st[k] for k in ('epsilon','M','R','accepted')},
                checkpoints=rows,batches=batches,final_keys=np.unique((fi[:,None]*r+np.arange(r)).ravel()).tolist(),final_fits=finalfits)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--timing',action='store_true');ap.add_argument('--case',type=int);args=ap.parse_args()
    cfg=dict(seed=SEED,cases=CASES,repetitions=100,preparation_groups=4,planning_blocks=64,batch=128,
             checkpoints=[256,512,1024,2048,4096,8192,16384,32768],primary_resource='distinct attribute labels')
    if args.timing:cfg['repetitions']=1;cfg['preparation_groups']=1;cfg['checkpoints']=[256,512,1024,2048,4096]
    else:save(ROOT/'protocol.json',cfg)
    out=ROOT/('timing' if args.timing else 'results');out.mkdir(exist_ok=True);allstart=time.perf_counter()
    for ci,case in enumerate(CASES):
        if args.case is not None and ci!=args.case:continue
        if all((out/f"{case['id']}_{rep:03d}.json.gz").exists() for rep in range(cfg['repetitions'])):continue
        a,o,y,pools=load_domain(case['domain'])
        for group in range(cfg['preparation_groups']):
            if all((out/f"{case['id']}_{rep:03d}.json.gz").exists() for rep in range(group,cfg['repetitions'],cfg['preparation_groups'])):continue
            b,ts,tcs,dc,st=setup(case,a,o,y,pools,group)
            sp=out/f"setup_{case['id']}_{group}.json"
            if sp.exists():
                prior=json.loads(sp.read_text('utf-8'))
                for key in ('epsilon','M','R'):assert abs(prior[key]-st[key])<=1e-12
                for key in ('m','accepted','pilot','initial'):assert prior[key]==st[key]
                st=prior
            else:save(sp,st)
            print('SETUP',case['id'],group,st['epsilon'],st['M'],st['R'],st['m'],round(st['seconds'],2),flush=True)
            for rep in range(group,cfg['repetitions'],cfg['preparation_groups']):
                dest=out/f"{case['id']}_{rep:03d}.json.gz"
                if dest.exists():continue
                tm=time.perf_counter();res=run_trial(case,cfg,rep,a,b,y,pools,ts,tcs,dc,st)
                import gzip
                tmp=dest.with_suffix('.partial')
                with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(res,f,separators=(',',':'))
                tmp.replace(dest)
                done=len(list(out.glob('*.json.gz')))
                progress('统一 logistic 实验',done,len(CASES)*cfg['repetitions'],f"{case['id']}，准备组{group+1}，重复{rep+1}；已运行{(time.perf_counter()-allstart)/60:.1f}分钟")
                print('TRIAL',case['id'],rep,round(time.perf_counter()-tm,2),{m:[(r['blocks'],r['single']['decision']) for r in res['checkpoints'] if r['method']==m][-1:] for m in METHODS},flush=True)

if __name__=='__main__':main()

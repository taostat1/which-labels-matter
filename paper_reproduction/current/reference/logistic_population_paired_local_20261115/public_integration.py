"""Certified integration over random public designs; no response is queried."""
from paired_local import *
import json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import population_experiment as pop

def init(ci):
    global A,B,TP,QP,CASE
    CASE=pop.s.CASES[ci]
    A,o,unused,pools=pop.s.load_domain(CASE['domain']);del unused
    B=(1-CASE['alpha'])*A+CASE['alpha']*o
    TP=pools['training'];QP=pools['comparison']

def chunk(task):
    ci,start,count=task;r=A.shape[1];ts=[np.zeros(A.shape[-1]),np.zeros(B.shape[-1])]
    rows=[]
    for i in range(start,start+count):
        rng=np.random.default_rng(202611152301+ci*1000000+i)
        tr=rng.choice(TP,CASE['n']);te=rng.choice(QP,128)
        v=certificate(A[tr],B[tr],A[te],B[te],ts,CASE['lam'],np.full(len(tr)*r,np.nan),np.full(len(te)*r,np.nan))
        rows.append([i,v['epsilon'],v['old']])
    return rows

def upper(values,cap,alpha):
    n=len(values);x=np.minimum(np.asarray(values),cap)
    sd=float(np.std(x,ddof=1));mean=float(x.mean());log=np.log(2/alpha)
    width=float(np.sqrt(2*sd**2*log/n)+7*cap*log/(3*(n-1)))
    return dict(n=n,mean=mean,sd=sd,cap=cap,alpha=alpha,public_mc_error=width,epsilon=mean+width)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True)
    ap.add_argument('--plans',type=int,default=16384);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();ci=args.case;tick=time.perf_counter()
    path=ROOT/f'integration_{ci}.json';done={}
    if path.exists():done={int(r[0]):r for r in json.loads(path.read_text())['records']}
    caps=json.loads((ROOT/'public_cap.json').read_text());cap=next(r['cap'] for r in caps if r['case']['id']==pop.s.CASES[ci]['id'])
    tasks=[(ci,i,min(64,args.plans-i)) for i in range(0,args.plans,64) if i not in done]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(ci,)) as ex:
        futures=[ex.submit(chunk,t) for t in tasks]
        for future in as_completed(futures):
            for row in future.result():done[row[0]]=row
            rows=sorted(done.values());certs=[]
            for n in (4096,8192,16384,32768):
                if all(i in done for i in range(n)):
                    certs.append(upper([done[i][1] for i in range(n)],cap,.005/4))
            pop.save(path,dict(case=pop.s.CASES[ci],records=rows,certificates=certs,labels_purchased=0,
                              seed=202611152301,simulated_test_size=128,seconds=time.perf_counter()-tick))
            if len(done)%512==0:print(ci,len(done),'latest',certs[-1] if certs else 'pending',flush=True)
    print('DONE',ci,certs,flush=True)
if __name__=='__main__':main()

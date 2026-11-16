"""Run independent last-case records while the main process finishes case six.

The main process reads completed records when it reaches the last case. This
worker yields with ten case-six records still outstanding, avoiding concurrent
writes to any case-seven record. It uses the frozen protocol without changes.
"""
from study import *
import gzip


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--groups',type=int,nargs='+',default=list(range(4)));args=ap.parse_args()
    cfg=json.loads((ROOT/'protocol.json').read_text('utf-8'));case=CASES[7]
    out=ROOT/'results';a,o,y,pools=load_domain(case['domain'])
    def yield_to_main():
        return len(list(out.glob(CASES[6]['id']+'_[0-9][0-9][0-9].json.gz')))>=90
    for group in args.groups:
        if yield_to_main():break
        if all((out/f"{case['id']}_{r:03d}.json.gz").exists() for r in range(group,100,4)):continue
        b,ts,tcs,dc,st=setup(case,a,o,y,pools,group)
        sp=out/f"setup_{case['id']}_{group}.json"
        if sp.exists():
            prior=json.loads(sp.read_text('utf-8'))
            for key in ('epsilon','M','R'):assert abs(prior[key]-st[key])<=1e-12
            for key in ('m','accepted','pilot','initial'):assert prior[key]==st[key]
            st=prior
        else:save(sp,st)
        print('SETUP',case['id'],group,st['epsilon'],st['m'],flush=True)
        for rep in range(group,100,4):
            if yield_to_main():break
            dest=out/f"{case['id']}_{rep:03d}.json.gz"
            if dest.exists():continue
            tick=time.perf_counter();rec=run_trial(case,cfg,rep,a,b,y,pools,ts,tcs,dc,st)
            tmp=dest.with_suffix('.parallel.partial')
            with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(rec,f,separators=(',',':'))
            tmp.replace(dest)
            done=len(list(out.glob('*.json.gz')))
            save(ROOT/f'parallel_case_progress_{group}.json',dict(completed=done,total=800,case=case['id'],rep=rep,seconds=time.perf_counter()-tick))
            print('TRIAL',rep,round(time.perf_counter()-tick,2),'total',done,flush=True)
    print('Worker finished or yielded to the main process.',flush=True)


if __name__=='__main__':main()

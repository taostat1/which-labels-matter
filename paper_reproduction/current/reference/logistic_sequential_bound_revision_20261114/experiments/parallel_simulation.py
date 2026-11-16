"""Use spare CPU/RAM for a disjoint early block of queued simulation designs."""
from pathlib import Path
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import validate_full as run


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int,default=272)
    parser.add_argument('--stop',type=int,default=472)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    jobs=[j for j in run.configs() if args.start<=j['id']<args.stop
          and not (run.ROOT/'results'/f'job_{j["id"]:04d}.json').exists()]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run.worker,j):j for j in jobs}
        for f in as_completed(futures):
            try:job=f.result();status='done'
            except Exception as exc:job=futures[f]['id'];status=repr(exc)
            run.save(run.ROOT/'progress_parallel.json',dict(last_job=job,status=status,
                seconds=time.perf_counter()-start,range=[args.start,args.stop],workers=args.workers))
            print(job,status,flush=True)

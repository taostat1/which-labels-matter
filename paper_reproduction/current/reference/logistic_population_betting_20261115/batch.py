from pathlib import Path
import argparse,subprocess,sys,time,json
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['matched','independent'],default='matched')
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    jobs=[(ci,g) for ci in range(8) for g in (6,7,8)] if args.kind=='matched' else [(ci,g) for ci in (3,6) for g in (18,19,20,21,22,23)]
    (ROOT/'logs').mkdir(exist_ok=True);running=[];failed=[];done=0
    while jobs or running:
        while jobs and len(running)<args.workers:
            ci,g=jobs.pop(0);dest=ROOT/'results'/f'case_{ci:02d}_{g:02d}.json'
            if dest.exists():done+=1;continue
            log=(ROOT/'logs'/f'case_{ci:02d}_{g:02d}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(ROOT/'experiment.py'),'--case',str(ci),'--group',str(g)],stdout=log,stderr=subprocess.STDOUT)
            running.append((ci,g,proc,log));print('START',ci,g,flush=True)
        keep=[]
        for ci,g,proc,log in running:
            if proc.poll() is None:keep.append((ci,g,proc,log))
            else:
                log.close();done+=1
                if proc.returncode:failed.append((ci,g,proc.returncode))
                print('FINISH',ci,g,proc.returncode,'done',done,flush=True)
        running=keep
        if running:time.sleep(1)
    (ROOT/f'{args.kind}_batch.json').write_text(json.dumps(dict(failed=failed,completed=done),indent=2))
    if failed:raise RuntimeError(failed)
if __name__=='__main__':main()

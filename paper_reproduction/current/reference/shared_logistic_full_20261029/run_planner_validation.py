"""Four preparation workers; numerical libraries use one thread each."""
import sys,subprocess,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
def worker(group):
    with (ROOT/f'planner_worker_{group}.log').open('w',encoding='utf-8') as f:
        code=subprocess.call([sys.executable,'-X','utf8',str(ROOT/'planner_revision.py'),'--group',str(group)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print('WORKER',group,'EXIT',code,flush=True)
    if code:raise RuntimeError(f'worker {group} failed')
if __name__=='__main__':
    protocol=json.loads((ROOT/'planner_revision_protocol.json').read_text('utf-8'))
    assert protocol['code_sha256']==hashlib.sha256((ROOT/'planner_revision.py').read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(worker,range(4)))
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'planner_revision.py')],cwd=ROOT,check=True)
    print('VALIDATION COMPLETE',flush=True)

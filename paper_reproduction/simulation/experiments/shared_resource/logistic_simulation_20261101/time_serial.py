"""Serial replay of five fixed formal pools; used for timing, not extra replicates."""
from pathlib import Path
import importlib.util,os,json,contextlib
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controlled_experiment',ROOT/'run.py')
sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
dest=ROOT/'serial_timing';dest.mkdir(exist_ok=True)
(dest/'protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
(dest/'formal_registration.json').write_text(json.dumps(dict(pools=5)),encoding='utf-8')
sim.ROOT=dest;os.environ['SIM_WORKER']='1'
if not all((dest/'results'/f'pool_{i:03}.json').exists() for i in range(5)):
 with (dest/'run.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f):sim.run(False,0,5)
rows=[]
for i in range(5):
 formal=json.loads((ROOT/'results'/f'pool_{i:03}.json').read_text('utf-8'))['records']
 timed=json.loads((dest/'results'/f'pool_{i:03}.json').read_text('utf-8'))['records']
 for a,b in zip(formal,timed):
  for key in ['name','method','group','choice','labels_total','branch','covered','acceptable','fallback']:assert a[key]==b[key],(key,a,b)
 rows.extend(timed)
summary=[]
for name in sorted({r['name'] for r in rows}):
 for method in sim.METHODS:
  values=[r['seconds'] for r in rows if r['name']==name and r['method']==method]
  summary.append(dict(name=name,method=method,mean_seconds=sum(values)/len(values),replicates=len(values)))
(ROOT/'timing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Serial timing verified: 5 pools, 240 method runs; labels and decisions match formal records.')

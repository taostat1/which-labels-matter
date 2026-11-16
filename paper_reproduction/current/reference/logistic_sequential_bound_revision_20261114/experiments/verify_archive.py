"""Import the companion in isolation and reproduce a synthetic query path."""
from pathlib import Path
import zipfile, tempfile, subprocess, json, sys

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[3]
archive=WORK/'output/LEC_Sequential_Bound_Revision_20261114.zip'
scratch=Path(tempfile.mkdtemp(prefix='lec_archive_',dir='D:/'))
with zipfile.ZipFile(archive) as z:z.extractall(scratch)
script=r'''
import sys,json
from pathlib import Path
base=Path.cwd()
root=base/'experiments/shared_resource/logistic_sequential_bound_revision_20261114/experiments'
sys.path.insert(0,str(root))
import validate_full as run
from whichlabels import compare,LabelCache,ArrayLabelProvider,prepare_comparison,__version__
job=run.configs()[1772]
a,b,y,plan,finals,seed,refs,methods=run.data(job)
p=prepare_comparison(a,b,plan,job['lam'])
new=compare(p,LabelCache(len(a),y.shape[1],ArrayLabelProvider(y)),finals[0],tolerance=job['g'],seed=seed)
saved=json.loads((root/'results/job_1772.json').read_text())
old=next(r for r in saved['records'] if r['method']=='paired_weighted')
assert new['total_labels']==old['total_labels']
assert new['choice']==old['choice'] and new['draws']==old['draws']
run.np.testing.assert_allclose(new['interval'],old['interval'],rtol=0,atol=1e-14)
print(json.dumps(dict(version=__version__,job=1772,labels=new['total_labels'],draws=new['draws'],choice=new['choice'],passed=True)))
'''
finished=subprocess.run([sys.executable,'-c',script],cwd=scratch,text=True,capture_output=True)
evidence=dict(scratch=str(scratch),returncode=finished.returncode,stdout=finished.stdout,stderr=finished.stderr)
(ROOT/'archive_execution_check.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps(evidence,indent=2))
assert finished.returncode==0

"""Record source, frozen inputs, software, and completed output hashes."""
from pathlib import Path
import json, hashlib, shutil, sys, platform, os
import numpy, scipy

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parents[1]
WORK=SHARED.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda:source.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def main():
    sources=list(ROOT.glob('*.py'))+list(ROOT.parent.glob('*.py'))
    for folder in ['logistic_all_scenarios_reanchor_20261114',
                   'logistic_local_anchor_refinement_20261114',
                   'logistic_manuscript_paid_anchor_20261114',
                   'shared_logistic_full_20261029']:
        sources+=list((SHARED/folder).glob('*.py'))
    sources+=list((WORK/'output/github/which-labels-logistic/src/whichlabels').glob('*.py'))
    inputs=[]
    for folder in ['logistic_cp_revision_20261106/real/attribution',
                   'logistic_cp_revision_20261106/real/coverage',
                   'logistic_cp_revision_20261106/cache',
                   'logistic_cp_revision_20261106/fivefold',
                   'logistic_cp_revision_20261106/simulation',
                   'logistic_simulation_refinement_20261106/simulation']:
        p=SHARED/folder
        inputs+=list(p.glob('protocol.json'))
        inputs+=list(p.rglob('records.json'))
        inputs+=list(p.rglob('groups.json'))
        inputs+=list(p.rglob('schedule_*.npz'))
        inputs+=list(p.rglob('pool_*.npz'))
        inputs+=list(p.rglob('pool_*.json'))
    inputs+=list((SHARED/'logistic_cv_earlystop_kfold_20261030/results').glob('schedule_KFold*.npz'))
    for folder in ['cross_backbone_20261028','goemotions_comparison_20261028','ade20k_decision_20261026']:
        inputs+=[SHARED/folder/'prepared/features_public.npz',SHARED/folder/'prepared/labels_oracle.npz']
    out={}
    checks=list(ROOT.parent.glob('*audit*.json'))+[ROOT.parent/'numeric_updates.json',ROOT.parent/'final_theory_consistency_review.json']
    for kind,paths in [('sources',sources),('frozen_inputs',inputs),('outputs',list((ROOT/'results').glob('job_*.*'))),('independent_checks',checks)]:
        out[kind]={str(p.relative_to(WORK)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
                   for p in sorted(set(paths)) if p.is_file()}
    out['software']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,platform=platform.platform())
    out['run_policy']='Fixed zero anchor and coefficient proposal; running-min cache bound; decision-triggered refinement; paired replay seeds unchanged.'
    (ROOT/'manifest.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    snapshot=ROOT/'source_snapshot'
    for source in sources:
        destination=snapshot/source.relative_to(WORK)
        destination.parent.mkdir(parents=True,exist_ok=True)
        # The preserved workspace tree can exceed Win32's legacy 260-char path.
        target=Path('\\\\?\\'+str(destination.resolve())) if os.name=='nt' else destination
        target.write_bytes(source.read_bytes())
    print({k:len(v) for k,v in out.items() if isinstance(v,dict)})


if __name__=='__main__':main()

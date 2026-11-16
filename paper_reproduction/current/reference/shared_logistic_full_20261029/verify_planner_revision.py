"""Audit final validation, preparation-only planning, and label accounting."""
from planner_revision import *
import hashlib


def main():
    protocol=json.loads((ROOT/'planner_revision_protocol.json').read_text('utf-8'))
    assert hashlib.sha256((ROOT/'planner_revision.py').read_bytes()).hexdigest()==protocol['code_sha256']
    files=sorted((ROOT/'planner_final_checked').glob('*.json.gz'));assert len(files)==160
    count=0;groups={};checks=0;audits=json.loads((ROOT/'audit_results.json').read_text('utf-8'))['cases']
    by_case={c['id']:c for c in CASES};pools_by_domain={}
    for case in CASES:
        if case['domain'] not in pools_by_domain:pools_by_domain[case['domain']]=load_domain(case['domain'])[3]
    for path in files:
        with gzip.open(path,'rt',encoding='utf-8') as f:rec=json.load(f)
        case=rec['case'];st=json.loads((ROOT/'results'/f"setup_{case['id']}_{rec['group']}.json").read_text('utf-8'))
        assert 40000<=rec['rep']<40020
        # The forecasting function succeeds without any comparison or audit fields.
        minimal={k:rec[k] for k in ('case','preparation','initial')}
        p=forecast_revision(minimal,st,pools_by_domain[case['domain']])
        original=rec['revision_plan']
        for key in ('selected','trial_label_cap','max_blocks','expected_labels'):
            assert p[key]==original[key],(path,key)
        rebuilt=revised_deferred(rec,original)
        saved=next(v for v in rec['revision_analysis'] if v['method']=='revised')
        for key in ('labels','decision','switched','lower','upper'):
            assert rebuilt[key]==saved[key],(path,key)
        assert len(set(rec['initial']))==rec['preparation']['labels']
        if saved['decision']!='unresolved':
            assert saved['upper']<=case['g'] or saved['lower']>=-case['g']
        for row in rec['checkpoints']:
            if row['method']=='refit': eps=2e-10/case['lam'];M=st['R'];d0=.05
            else:eps=st['epsilon'];M=st['M'];d0=.04
            n=row['blocks'];t0=np.log(32/d0);t1=np.log(32/.02)
            standard=max(0,row['single']['radius']-eps-14*M*t0/(3*(n-1)))
            expected=standard*np.sqrt(t1/t0)+14*M*t1/(3*(n-1))+eps
            assert abs(expected-row['protected']['radius'])<1e-12
            checks+=1
        groups.setdefault(case['id'],{}).setdefault(str(rec['group']),0)
        groups[case['id']][str(rec['group'])]+=1;count+=1
    out=dict(passed=True,paths=count,confidence_budget_checks=checks,
             preparation_only_forecast_checks=count,label_ledger_reconstructions=count,
             group_counts=groups,code_sha256=protocol['code_sha256'])
    save(ROOT/'planner_revision_checks.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()

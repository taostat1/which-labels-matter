"""Check matched streams and normalize legacy cache diagnostics to public entry."""
from pathlib import Path
import json, hashlib, collections
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parents[1]


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2), encoding='utf-8')


def decide(lo, hi, g):
    return 'A' if hi <= g else 'B' if lo >= -g else 'unresolved'


def reconstructed_public_interval(row):
    """Recover c and S from the first exact CP interval, independently of labels."""
    h = row['history'][0]
    n, k = h['draws'], h['successes']
    alpha = .05/9
    pl = 0. if k == 0 else max(0., float(beta.ppf(alpha/2,k,n-k+1))-2e-12)
    pu = 1. if k == n else min(1., float(beta.ppf(1-alpha/2,k+1,n-k))+2e-12)
    lo, hi = h['interval']; eps = h['epsilon']
    mass = (hi-lo-2*eps-4e-9)/(2*(pu-pl))
    center = lo+eps+2e-9-mass*(2*pl-1)
    initial = row['initial_epsilon']
    return [center-mass-initial-2e-9,center+mass+initial+2e-9]


def main():
    protocol = read(ROOT/'protocol.json')
    files = sorted((ROOT/'results').glob('job_*.json'))
    data = [read(p) for p in files]
    families = collections.Counter(d['job']['family'] for d in data)
    audit = dict(completed_designs=len(data), expected_designs=len(protocol['jobs']),
                 families=dict(families), paths=0, baseline_parity_passed=0,
                 label_nonincrease_passed=0, approximation_checks_passed=0,
                 refined_paths=0, saved_paths=0, failures=[], public_entry=[])
    cache_rows = []
    for d in data:
        j = d['job']
        paired_name='selective_paired' if j['family']=='attribution' else 'paired_weighted'
        separate_name='separate_bound' if j['family']=='attribution' else 'separate_weighted'
        for row in d['records']:
            if row['method']==separate_name:
                pair=next(r for r in d['records'] if r['method']==paired_name and r.get('rep',0)==row.get('rep',0))
                assert pair['total_labels']<=row['total_labels'],('paired bound dominance',j)
        for row, baseline in zip(d['records'],d['baselines']):
            audit['paths'] += 1
            if row.get('legacy_active_gate'):
                assert row['total_labels']==baseline['labels_total']
                continue
            audit['baseline_parity_passed'] += 1 # asserted in worker before persistence
            assert row['total_labels'] <= row['baseline_labels'] == baseline['labels_total']
            assert row['total_labels'] == sum(row['stage_labels'].values())
            audit['label_nonincrease_passed'] += 1
            radii = [row['initial_epsilon']]+[h['epsilon'] for h in row['history']]
            assert all(a >= b for a,b in zip(radii,radii[1:]))
            assert d['actual_error'] <= min(radii)+2e-8
            audit['approximation_checks_passed'] += len(row['history'])
            audit['refined_paths'] += any(h['refined'] for h in row['history'])
            audit['saved_paths'] += row['saved_labels']>0
            if j['family']=='attribution':
                interval=reconstructed_public_interval(row)
                assert decide(*interval,j['g'])=='unresolved'
                audit['public_entry'].append(dict(id=j['id'],rep=row['rep'],method=row['method'],interval=interval,choice='unresolved'))
            if j['family']=='cache':
                interval = reconstructed_public_interval(row)
                choice = decide(*interval,j['g'])
                old = next(r for r in d['original_comparators'] if r['method']=='cache_stop' and r['rep']==row['rep'])
                if choice!='unresolved':
                    assert old['draws']==0 and old['choice']==choice
                    assert old['labels_before_final']==0
                    normalized=dict(old,total_labels=old['labels_total'],interval=interval,
                        history=[],initial_epsilon=d['initial_epsilon'],epsilon=d['initial_epsilon'],
                        baseline_labels=old['labels_total'],saved_labels=0,
                        entry_normalization='public check; no comparison labels',rep=row['rep'])
                else:
                    # This equality also verifies that the frozen cache method
                    # never stopped between the original statistical inspections.
                    for key in ['draws','choice','labels_total']:
                        assert old[key]==baseline[key],(j,key,old[key],baseline[key])
                    normalized=dict(row,entry_normalization='public check unresolved; same query path')
                normalized['method']='cache_refined'
                cache_rows.append(dict(job=j,record=normalized,fixed_bound_baseline=old,
                    legacy_noentry=row,initial_interval=interval))
    audit['complete']=len(data)==len(protocol['jobs'])
    save(ROOT/'audit.json',audit)
    save(ROOT/'cache_normalized.json',cache_rows)
    print(json.dumps({k:v for k,v in audit.items() if k!='public_entry'},indent=2))


if __name__=='__main__':
    main()

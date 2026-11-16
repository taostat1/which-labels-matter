"""Audit adaptive error accounting and refresh summaries from observed prefixes."""
from label_study import *
import shutil

def main():
    audits=json.loads((ROOT/'audit_results.json').read_text('utf-8'))['cases']
    inputs=json.loads((ROOT/'input_costs.json').read_text('utf-8'))['cases'];checks=0;changed=0
    for directory,summary,total in [('results','label_replay_summary.json',800),('label_results','label_results_summary.json',160)]:
        rows=[]
        for case in CASES:
            _,_,y,pools=load_domain(case['domain'])
            states={g:json.loads((ROOT/'results'/f"setup_{case['id']}_{g}.json").read_text('utf-8')) for g in range(4)}
            for path in sorted((ROOT/directory).glob(case['id']+'_[0-9][0-9][0-9].json.gz')):
                with gzip.open(path,'rt',encoding='utf-8') as f:rec=json.load(f)
                reviewed,plan=analyze_record(rec,states[rec['group']],pools,inputs[case['id']]['seconds'],audits[case['id']])
                for c in rec['checkpoints']:
                    if c['method']=='refit':
                        assert c['single']['radius']<=c['adaptive']['radius']<=c['protected']['radius']+1e-10
                        checks+=1
                if directory=='label_results':
                    before=rec['label_analysis'];changed+=sum(v['labels']!=old['labels'] or v['decision']!=old['decision'] for v,old in zip(reviewed,before))
                    rec.setdefault('initial_label_analysis',before)
                    rec['label_analysis']=reviewed;rec['analysis_revision']='Adaptive selected-refit sampling error .04; direct refit .05. The raw draws, fitted statistics and trial budgets are unchanged.'
                    tmp=path.with_suffix('.reviewed')
                    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(rec,f,separators=(',',':'))
                    tmp.replace(path)
                rows.extend(reviewed)
            print('AUDITED',directory,case['id'],len(rows)//7,flush=True)
        assert len(rows)==7*total
        prior=ROOT/summary
        backup=ROOT/'revisions/before_label_optimization_20261029'/('before_budget_review_'+summary)
        if prior.exists() and not backup.exists():shutil.copy2(prior,backup)
        save(prior,dict(rows=rows,trajectories=total,methods_per_trajectory=7,
             analysis_note='Calibration .01; pilot-selected refit .04; deferred moment .02 and refit .02; direct refit .05.'))
    protocol=json.loads((ROOT/'label_protocol.json').read_text('utf-8'))
    protocol['error_budgets']=dict(calibration=.01,single_moment=.04,direct_refit=.05,pilot_selected_refit=.04,deferred_moment=.02,deferred_refit=.02)
    protocol['budget_audit']='The adaptive selected-refit branch reserves calibration error. Its interval was reconstructed from the same empirical variance before final reporting; no data, acquisition multipliers or inspection times were changed.'
    save(ROOT/'label_protocol.json',protocol)
    save(ROOT/'label_budget_audit.json',dict(interval_order_checks=checks,changed_method_records=changed,passed=True))
    print('ADAPTIVE ERROR AUDIT',checks,'intervals;',changed,'records changed')

if __name__=='__main__':main()

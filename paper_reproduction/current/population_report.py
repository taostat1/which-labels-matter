from pathlib import Path
import json,statistics,hashlib,csv
ROOT=Path(__file__).resolve().parent;P=ROOT/'paper';SHARED=ROOT.parent
CODES=['T1','T2','T3','T4','V1','V2','V3','V4']
METHODS=['direct_refits','active_testing','ams','lec']
def read(p):return json.loads(p.read_text('utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def table(summary,methods,names,caption,label):
    out=[r'\begin{table}[htbp]\centering\small',r'\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.08}',
      r'\begin{tabular}{@{}lrrrrrrrr@{}}\toprule',r' & \multicolumn{4}{c}{GoEmotions} & \multicolumn{4}{c}{ADE20K}\\',
      r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',r'Method & T1 & T2 & T3 & T4 & V1 & V2 & V3 & V4\\\midrule']
    for m,name in zip(methods,names):
        cells=[]
        for code in CODES:
            v=summary[code][m];mi=min(summary[code][mm]['mean'] for mm in methods)
            mu=f"{v['mean']/1000:.1f}";sd=f"{v['sd']/1000:.1f}"
            if abs(v['mean']-mi)<1e-8:mu=r'\mathbf{'+mu+'}'
            cells.append('$'+mu+r'_{\color{gray}'+sd+'}$')
        out.append(name+' & '+' & '.join(cells)+r'\\')
    out+=[r'\midrule',r'Saving (\%) & '+' & '.join(f"{summary[c]['saving_vs_best_baseline']:.1f}" for c in CODES)+r'\\',
          r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}']
    return '\n'.join(out)+'\n'
def main():
    original=read(SHARED/'logistic_population_comparators_20261114/records.json')
    rows=[r for r in original if r['method']!='lec'];ablation=[];files=sorted((ROOT/'results').glob('case_*.json'));assert len(files)==96
    parity=0
    for f in files:
        d=read(f);rows.append(next(r for r in d['records'] if r['method']=='lec'));ablation.extend(d['records'])
        parity+=d['prior_interval_experiment_parity'] is True
    def summarize(records,methods):
        out={}
        for ci,code in enumerate(CODES):
            stats={}
            for m in methods:
                rr=[r for r in records if r['case_index']==ci and r['method']==m]
                assert len(rr)==12 and {r['group'] for r in rr}==set(range(6,18))
                stats[m]=dict(mean=statistics.mean(r['labels'] for r in rr),sd=statistics.stdev(r['labels'] for r in rr),
                    acceptable=sum(r['acceptable_audit_mean'] for r in rr),covered=sum(r['covered_audit_mean'] for r in rr),
                    issued=sum(r['decision_issued'] for r in rr),errors=sum(not r['acceptable_audit_mean'] for r in rr),
                    fallbacks=sum(r.get('fallback',False) for r in rr))
            best=min(stats[m]['mean'] for m in methods if m!='lec');stats['saving_vs_best_baseline']=100*(1-stats['lec']['mean']/best)
            out[code]=stats
        return out
    summary=summarize(rows,METHODS);ab=summarize(ablation,['fixed_bound','lec'])
    (P/'population_comparison_table.tex').write_text(table(summary,METHODS,['Direct refits','Active testing','AMS',r'\textbf{LEC}'],
      'Population-risk comparison with external methods. Entries are mean distinct labels in thousands, with standard deviations over 12 query paths. Counts include comparison and final training. All methods start without preparation labels and allow early stopping. Savings compare LEC with the lowest baseline mean; bold marks column minima.', 'tab:population_comparison'),encoding='utf-8')
    (P/'population_cache_ablation_table.tex').write_text(table(ab,['fixed_bound','lec'],['Fixed error bound',r'\textbf{LEC}'],
      'Free error-bound refinement with cumulative betting intervals. Entries are mean distinct labels in thousands, with standard deviations over the same 12 paths. Both methods use the paired public certificate, fixed zero anchors, and the same query and final-training draws.', 'tab:population_cache_ablation'),encoding='utf-8')
    # Preserve the existing reference label used in prose.
    old=(SHARED/'logistic_population_comparators_20261114/paper/population_cache_ablation_table.tex').read_text('utf-8')
    import re
    oldlabel=re.search(r'\\label\{([^}]+)\}',old).group(1)
    pp=P/'population_cache_ablation_table.tex';pp.write_text(pp.read_text('utf-8').replace('tab:population_cache_ablation',oldlabel),encoding='utf-8')
    out=dict(cases=summary,ablation=ab,method_paths=len(rows),lec_paths=96,external_paths=288,
       total_issued=sum(r['decision_issued'] for r in rows),total_acceptable=sum(r['acceptable_audit_mean'] for r in rows),
       total_covered=sum(r['covered_audit_mean'] for r in rows),prior_query_path_parity=parity,
       frozen_external_sha256=hashlib.sha256((SHARED/'logistic_population_comparators_20261114/records.json').read_bytes()).hexdigest())
    save(ROOT/'summary.json',out);save(ROOT/'records.json',rows);save(ROOT/'ablation_records.json',ablation)
    save(P/'evidence/population_external_summary.json',out)
    rr=[dict(configuration=CODES[r['case_index']],method=r['method'],group=r['group'],labels=r['labels'],
        acceptable=r['acceptable_audit_mean'],covered=r['covered_audit_mean'],choice=r['choice'],
        fallback=r.get('fallback',False)) for r in rows]
    with (ROOT/'path_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    for c,st in summary.items():print(c,[(m,round(st[m]['mean'],2)) for m in METHODS],round(st['saving_vs_best_baseline'],2),'ablation',round(ab[c]['saving_vs_best_baseline'],3))
    print('issued acceptable covered parity',out['total_issued'],out['total_acceptable'],out['total_covered'],parity)
if __name__=='__main__':main()

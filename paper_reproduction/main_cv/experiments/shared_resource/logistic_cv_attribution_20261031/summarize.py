"""Summarize every prespecified cell and paired component ablation."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;P=ROOT/'paper';R=ROOT/'results'
P.mkdir(parents=True, exist_ok=True)
def read(p):return json.loads(p.read_text('utf-8'))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
cfg=read(ROOT/'protocol.json');rows=read(R/'records.json');groups=read(R/'groups.json')
assert len(rows)==1200 and len(groups)==40
assert all(x['labels_total']==sum(x['stage_labels'].values()) for x in rows)
assert hashlib.sha256((R/'executed_run.py').read_bytes()).hexdigest()==read(R/'source_hash.json')['sha256']
cases=[(d,g) for d in cfg['domains'] for g in d['tolerances']]
summary={};stats=[]
for case,g in cases:
    name=case['name']+'-g'+str(g);rr=[x for x in rows if x['name']==name]
    gg=[x for x in groups if x['domain']==case['name']]
    assert len(rr)==300 and len(gg)==20
    s=dict(domain=case['name'],g=g,lam=case['lam'],n=case['train_size'],methods={},
           target_range=[min(x['target'] for x in gg),max(x['target'] for x in gg)],
           max_actual_error=max(abs(x['actual_error']) for x in gg),
           regime_counts={q:sum((('both' if abs(x['target'])<=g else 'A_only' if x['target']<0 else 'B_only')==q) for x in gg) for q in ['A_only','both','B_only']})
    for method in cfg['methods']:
        mr=[x for x in rr if x['method']==method]
        pm=np.array([np.mean([x['labels_total'] for x in mr if x['group']==i]) for i in range(20)])
        sm=dict(mean=float(pm.mean()),pool_sd=float(pm.std(ddof=1)),pool_means=pm.tolist(),
                min_labels=min(x['labels_total'] for x in mr),max_labels=max(x['labels_total'] for x in mr),
                covered=sum(x['covered'] for x in mr),acceptable=sum(x['acceptable'] for x in mr),runs=len(mr),
                fallback=sum(x['branch']=='exact_fallback' for x in mr),
                seconds=float(np.mean([x['seconds'] for x in mr])),
                mean_draws=float(np.mean([x.get('draws',0) for x in mr])),
                stages={k:float(np.mean([x['stage_labels'].get(k,0) for x in mr])) for k in set().union(*(x['stage_labels'] for x in mr))})
        s['methods'][method]=sm
        stats.append(dict(case=name,method=method,mean=sm['mean'],pool_sd=sm['pool_sd'],
                          covered=sm['covered'],acceptable=sm['acceptable'],fallback=sm['fallback'],seconds=sm['seconds']))
    prop=np.array(s['methods']['selective_paired']['pool_means'])
    s['contrasts']={}
    idx=np.random.default_rng(202610313999).integers(0,20,(20000,20))
    for method in ['active_cv','whole_object_purchases','no_early_stop','separate_bound']:
        baseline=np.array(s['methods'][method]['pool_means'])
        s['contrasts'][method]=dict(saving=float(1-prop.mean()/baseline.mean()),
            saving_ci=np.quantile(1-prop[idx].mean(1)/baseline[idx].mean(1),[.025,.975]).tolist(),
            mean_difference=float((baseline-prop).mean()),better_pools=int(np.sum(prop<baseline)),
            tied_pools=int(np.sum(prop==baseline)))
    s['diagnostic']={}
    for key in ['paired_epsilon','separate_epsilon','geometry_epsilon']:
        vals=[x[key]/g for x in gg if x[key] is not None]
        s['diagnostic'][key]=dict(ratio_range=[min(vals),max(vals)],below_g=sum(x<1 for x in vals))
    s['diagnostic']['positive_splits']=sum(x['gamma_positive'] for x in gg)
    summary[name]=s
save(R/'summary.json',summary)
with (R/'method_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)

# Nested tolerances use exactly the same query sequences and final subsets.
look={(x['domain'],x['group'],x['rep'],x['g'],x['method']):x for x in rows}
for group in range(20):
    for rep in range(3):
        for method in cfg['methods']:
            counts=[look['image',group,rep,g,method]['labels_total'] for g in [.0025,.00375,.005]]
            assert counts==sorted(counts,reverse=True),(group,rep,method,counts)
        for domain,g in [('image',.0025),('image',.00375),('image',.005),('text',.005)]:
            pr=look[domain,group,rep,g,'selective_paired']
            for method in ['whole_object_purchases','no_early_stop','separate_bound']:
                assert pr['labels_total']<=look[domain,group,rep,g,method]['labels_total']

order=list(summary);ss=[summary[x] for x in order]
labels={'active_cv':'Active CV','selective_paired':r'\textbf{Selective moments}',
        'whole_object_purchases':'Whole-object purchases','no_early_stop':'No early stopping','separate_bound':'Separate error bound'}
lines=[r'\begin{table}[htbp]\centering\small',
 r'\caption{Complete labels in the tolerance grid and matched ablations, in thousands. Entries are means; grey subscripts are standard deviations of 20 pool means. Each cell has 60 paths. The last three methods remove one component while keeping the moment estimator and query sequence fixed.}',
 r'\label{tab:attribution}',r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}',
 r'\begin{tabularx}{\linewidth}{@{}l*{4}{>{\centering\arraybackslash}X}@{}}\toprule',
 r' & \multicolumn{3}{c}{ADE20K: DINOv2-S vs DINOv2-B} & GoEmotions\\',
 r'\cmidrule(lr){2-4}\cmidrule(lr){5-5}',r'Tolerance $g$ & .0025 & .00375 & .005 & .005\\',
 r'Penalty $\lambda$ & .5 & .5 & .5 & .125\\\midrule']
for method in cfg['methods']:
    cells=[]
    for s in ss:
        a=s['methods'][method];mean=f'{a["mean"]/1000:.2f}';sd=f'{a["pool_sd"]/1000:.2f}'
        if method=='selective_paired':mean=r'\mathbf{'+mean+'}'
        cells.append('$'+mean+r'_{\color{gray}'+sd+'}$')
    lines.append(labels[method]+' & '+' & '.join(cells)+r'\\')
lines.extend([r'\midrule\rowcolor{black!5}Saving vs active CV (\%) & '+' & '.join(f'{100*s["contrasts"]["active_cv"]["saving"]:.1f}' for s in ss)+r'\\',
    r'95\% pool-bootstrap interval & '+' & '.join('['+', '.join(f'{100*x:.1f}' for x in s['contrasts']['active_cv']['saving_ci'])+']' for s in ss)+r'\\',
    r'Pools: $A$ only / both / $B$ only & '+' & '.join(' / '.join(str(s['regime_counts'][q]) for q in ['A_only','both','B_only']) for s in ss)+r'\\',
    r'\bottomrule\end{tabularx}\end{table}'])
(P/'attribution_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

lines=[r'\begin{table}[htbp]\centering\small',
 r'\caption{Approximation allowances relative to tolerance on the new pools. Ranges are over 20 pool averages. All 640 splitwise contraction margins are positive in each domain. The detailed paired allowance enters the primary decision rule; the simple geometry bound is diagnostic.}',
 r'\label{tab:geometry_diagnostic}',r'\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.12}',
 r'\begin{tabular}{lrrrr}\toprule',r'Comparison & $g$ & Paired $\bar\varepsilon/g$ & Separate $\bar\varepsilon/g$ & Simple $\bar\varepsilon_{\rm geom}/g$\\\midrule']
for s in ss:
    cells=[]
    for key in ['paired_epsilon','separate_epsilon','geometry_epsilon']:
        lo,hi=s['diagnostic'][key]['ratio_range'];fmt='.3f' if hi<1 else '.2f'
        cells.append(format(lo,fmt)+'--'+format(hi,fmt))
    lines.append(('Image' if s['domain']=='image' else 'Text')+f' & {s["g"]:g} & '+' & '.join(cells)+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}')
(P/'diagnostic_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

checks=dict(passed=True,runs=len(rows),pools=len(groups),methods=cfg['methods'],
    covered=sum(x['covered'] for x in rows),acceptable=sum(x['acceptable'] for x in rows),
    primary_runs=sum(x['method']=='selective_paired' for x in rows),
    primary_fallback=sum(x['method']=='selective_paired' and x['branch']=='exact_fallback' for x in rows),
    independent_corpus_draws=40,paired_paths=240,distinct_query_streams=120,tolerance_monotonicity=True,label_ledger_identity=True,
    source_hash=read(R/'source_hash.json')['sha256'])
save(ROOT/'validation_checks.json',checks)
compact={name:dict(labels={m:round(x['mean'],1) for m,x in s['methods'].items()},
                       saving=round(100*s['contrasts']['active_cv']['saving'],2),
                       regimes=s['regime_counts'],fallback={m:x['fallback'] for m,x in s['methods'].items()},
                       diagnostics=s['diagnostic']) for name,s in summary.items()}
print(json.dumps(dict(checks=checks,results=compact),indent=2))

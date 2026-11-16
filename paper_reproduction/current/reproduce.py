"""Regenerate Tables 1--8, Figure 3, and record audits from frozen paths.

This command aggregates saved experiments. It does not refit the real-data
models. Fresh simulation and full replay entry points are documented separately.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,statistics,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=HERE/'data'
def read(p):
    return json.loads(gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text('utf-8'))
def digest(p):
    b=p.read_bytes()
    if p.suffix!='.gz':b=b.replace(b'\r\n',b'\n')
    return hashlib.sha256(b).hexdigest()

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def stats(rows):
    return dict(mean=statistics.mean(r['labels'] for r in rows),sd=statistics.stdev(r['labels'] for r in rows),
      runs=len(rows),acceptable=sum(r['acceptable_audit_mean'] for r in rows),covered=sum(r['covered_audit_mean'] for r in rows),
      fallbacks=sum(r.get('fallback',False) for r in rows),stages={k:statistics.mean(r['stage_labels'].get(k,0) for r in rows) for k in sorted(set().union(*(r['stage_labels'] for r in rows)))})
def matched_table(summary):
    lines=[r'\begin{table}[htbp]\centering\small',r'\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.08}',
      r'\begin{tabular}{@{}lrrrrrrrr@{}}\toprule',r' & \multicolumn{4}{c}{GoEmotions} & \multicolumn{4}{c}{ADE20K}\\',
      r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',r'Initialization / bound & T1 & T2 & T3 & T4 & V1 & V2 & V3 & V4\\\midrule']
    for method,label in [('pilot_fixed','Paid pilot / fixed'),('pilot_cache','Paid pilot / cache'),('zero_fixed','Zero / fixed'),('zero_cache',r'\textbf{Zero / cache (LEC)}')]:
        cells=[]
        for code,v in summary.items():
            s=v[method];n=f"{s['mean']/1000:.1f}";best=min(z['mean'] for z in v.values())
            if abs(s['mean']-best)<1e-8:n=r'\mathbf{'+n+'}'
            cells.append('$'+n+r'_{\color{gray}'+f"{s['sd']/1000:.1f}"+'}$')
        lines.append(label+' & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Matched population-risk decomposition. Mean distinct labels are in thousands, with standard deviations over the same 12 paths. Both initializations use cumulative betting, common query and final-training streams, and the same caps. Paid preparation fits anchors from 512 draws and buys independent calibration labels. Each cache arm retains its fixed counterpart\textquotesingle s initial bound until comparison starts. Bold marks column minima, including ties.}',r'\label{tab:population_cache_ablation}',r'\end{table}']
    return '\n'.join(lines)+'\n'
def figure(cases,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['full_pool_cv','active_cv','whole_object_purchases','selective_paired'];names=['Full annotation','Active CV','Whole objects','LEC'];colors=['#808080','#345e87','#ad6c51','#428779']
    fig,axes=plt.subplots(2,2,figsize=(11.8,6.6));fig.subplots_adjust(left=.15,right=.99,bottom=.07,top=.94,wspace=.52,hspace=.4)
    for ax,c in zip(axes.flat,cases):
        vals=[c['methods'][m]['mean']/1000 for m in methods];sv=[100*(1-v*1000/c['full_labels']) for v in vals]
        ax.barh(range(4),vals,color=colors,height=.52);ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlim(0,41)
        ax.set_title(('ADE20K' if c['domain']=='image' else 'GoEmotions')+f", g={c['g']:g}",loc='left',weight='bold')
        for k,(v,s) in enumerate(zip(vals,sv)):ax.text(v+.45,k,f'{v:.2f}  ({s:.1f}%)',va='center',fontsize=9,weight='bold' if v==min(vals) else 'normal')
        ax.set_xlabel('Distinct labels (thousands); savings vs full in parentheses');ax.spines[['top','right']].set_visible(False)
    fig.savefig(path);plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('build/paper'));ap.add_argument('--skip-record-audit',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    manifest=read(HERE/'manifest.json')
    for name,expected in manifest['files'].items():assert digest(HERE/name)==expected,name
    for name,expected in manifest['reference_sources'].items():
        assert digest(HERE/'reference'/name)==expected,name
    docs=read(DATA/'fixed_records.json.gz');normalized=read(DATA/'cache_normalized.json.gz')
    r=module('fixed_report');r.PAPER=out;r.ROOT=out;r.EXP=out/'record_audit';r.EXP.mkdir(exist_ok=True)
    save(r.EXP/'cache_normalized.json',normalized)
    families={f:r.aggregate(docs,f) for f in ['simulation','boundary','attribution','coverage','cache','fivefold']}
    r.simulation_table(families['simulation']);r.simulation_table(families['boundary'],True)
    r.attribution_table(families['attribution']);r.coverage_figure(families['coverage']);r.fivefold_table(families['fivefold'])
    cache=r.normalized_cache(docs);r.cache_table(cache)
    groups=read(DATA/'geometry_pools.json');geometry=[]
    lines=[r'\begin{table}[htbp]\centering\small',r'\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.12}',r'\begin{tabular}{lrrrr}\toprule',r'Comparison & $g$ & Paired $\bar\varepsilon/g$ & Separate $\bar\varepsilon/g$ & Simple $\bar\varepsilon_{\rm geom}/g$\\\midrule']
    for domain,g in [('image',.0025),('image',.00375),('image',.005),('text',.005)]:
        gg=[x for x in groups if x['domain']==domain];assert len(gg)==20
        row={'domain':domain,'g':g};cells=[]
        for key in ['paired_epsilon','separate_epsilon','geometry_epsilon']:
            v=[x[key]/g for x in gg];lo,hi=min(v),max(v);row[key]=[lo,hi];fmt='.3f' if hi<1 else '.2f';cells.append(format(lo,fmt)+'--'+format(hi,fmt))
        geometry.append(row);lines.append(('Image' if domain=='image' else 'Text')+f' & {g:g} & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Initial public approximation error bounds relative to tolerance. Ranges are over 20 pool averages. All 640 splitwise contraction margins are positive in each domain. LEC refines the detailed paired bound using acquired labels; the simple geometry bound is diagnostic.}',r'\label{tab:geometry_diagnostic}',r'\end{table}']
    (out/'diagnostic_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    zero=read(DATA/'population_zero.json.gz');paid=read(DATA/'population_paid.json.gz');base=read(DATA/'population_baselines.json.gz')
    codes=['T1','T2','T3','T4','V1','V2','V3','V4'];summary={};matched={};allrows=[];matchedrows=[]
    for ci,code in enumerate(codes):
        z=[v for d in zero if d['case_index']==ci for v in d['records']];p=[v for d in paid if d['case_index']==ci for v in d['records']];b=[v for d in base if d['case']['id']==zero[ci*12]['case']['id'] for v in d['records']]
        ss={m:stats([v for v in z+b if v['method']==m]) for m in ['direct_refits','active_testing','ams','lec']}
        ss['saving_vs_best_baseline']=100*(1-ss['lec']['mean']/min(ss[m]['mean'] for m in ['direct_refits','active_testing','ams']));summary[code]=ss
        mm={m:stats([v for v in p if v['method']==m]) for m in ['pilot_fixed','pilot_cache']}
        for name,orig in [('zero_fixed','fixed_bound'),('zero_cache','lec')]:mm[name]=stats([v for v in z if v['method']==orig])
        matched[code]=mm;allrows += b+[v for v in z if v['method']=='lec'];matchedrows+=z+p
    report=module('population_report')
    (out/'population_comparison_table.tex').write_text(report.table(summary,['direct_refits','active_testing','ams','lec'],['Direct refits','Active testing','AMS',r'\textbf{LEC}'],
      'Population-risk comparison with external methods. Entries are mean distinct labels in thousands, with standard deviations over 12 query paths. Counts include comparison and final training. All methods start without preparation labels and allow early stopping. Savings compare LEC with the lowest baseline mean; bold marks column minima.','tab:population_comparison'),encoding='utf-8')
    (out/'population_cache_ablation_table.tex').write_text(matched_table(matched),encoding='utf-8')
    metadata=read(DATA/'matched_protocol.json');audit=read(DATA/'risk_audit.json');case_names=['Pooling','Nearby pooling','Pooling','Nearby pooling','Attention heads','Attention heads','Nearby heads','Encoders']
    configrows=[];lines=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{llrrrrr}\toprule',r'Case & Comparison & $n$ & $\lambda$ & $\alpha$ & $g$ & Audited $\Delta_n$ (s.e.)\\\midrule']
    for code,c,name in zip(codes,metadata['cases'],case_names):
        ar=audit['cases'][c['id']]['difference'];mant,exp=f"{ar['se']:.1e}".split('e')
        lines.append(f"{code} & {name} & {c['n']} & {c['lam']:g} & {c['alpha']:g} & {c['g']:g} & {ar['mean']:.5f} ($"+mant+r'\times10^{'+str(int(exp))+'}$)'+r'\\')
        configrows.append(dict(code=code,**c,audit=ar))
    lines += [r'\bottomrule\end{tabular}',r'\caption{Formal logistic configurations. The independent audit estimates the expected risk difference and its Monte Carlo standard error. Both candidates are acceptable when the difference lies in $[-g,g]$.}',r'\label{tab:configs}',r'\end{table}']
    (out/'config_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    budgets={m:{str(B):sum(v['labels']<=B and v['acceptable_audit_mean'] for v in matchedrows if v['method']==m) for B in metadata['label_budget_grid']} for m in ['lec','fixed_bound','pilot_fixed','pilot_cache']}
    # Shared bookkeeping, monotone refinement, and current population audit.
    for v in allrows+matchedrows:
        assert v['labels']==sum(v['stage_labels'].values());assert 6<=v['group']<=17
    counts={f:{k:sum(c['displayed_'+k] for c in cs) for k in ['runs','covered','acceptable','fallback']} for f,cs in families.items()}
    counts['cache']={k:sum(c['displayed_counts'][k] for c in cache) for k in ['runs','covered','acceptable','fallback']}
    audit_out=dict(fixed=counts,population=dict(runs=len(allrows),acceptable=sum(v['acceptable_audit_mean'] for v in allrows),covered=sum(v['covered_audit_mean'] for v in allrows)),matched=dict(runs=len(matchedrows),acceptable=sum(v['acceptable_audit_mean'] for v in matchedrows),covered=sum(v['covered_audit_mean'] for v in matchedrows)),budgets=budgets)
    if not args.skip_record_audit:
        ar=r.EXP/'results';ar.mkdir(exist_ok=True)
        for i,d in enumerate(docs):save(ar/f'job_{i:04d}.json',d)
        save(r.EXP/'protocol.json',read(DATA/'fixed_protocol.json'))
        fa=module('fixed_audit');fa.ROOT=r.EXP;fa.main()
        audit_out['record_checks']=read(r.EXP/'audit.json')
        audit_out['record_checks'].pop('public_entry',None)
    save(out/'audit.json',audit_out);save(out/'summary.json',dict(families=families,cache=cache,geometry=geometry,population=summary,matched=matched,configurations=configrows,budgets=budgets))
    figure(r.real_order(families['coverage']),out/'Figure_3.pdf')
    tablefiles=['simulation_label_table.tex','attribution_table.tex','diagnostic_results.tex','population_comparison_table.tex','config_table.tex','cache_summary_table.tex','population_cache_ablation_table.tex','simulation_boundary_table.tex']
    # Determine the simulation filename from the retained reporter.
    if not (out/tablefiles[0]).exists():tablefiles[0]='simulation_table.tex'
    for i,f in enumerate(tablefiles,1):
        assert (out/f).exists(),f
        (out/f'Table_{i}.tex').write_bytes((out/f).read_bytes())
    with (out/'population_paths.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['case_index','group','method','labels','acceptable_audit_mean','covered_audit_mean'];writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(allrows+matchedrows)
    save(out/'artifact_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.suffix in ['.tex','.tikz','.pdf','.csv']})
    print(json.dumps({'output':str(out),'tables':8,'figure':3,'population':audit_out['population'],'matched':audit_out['matched'],'budgets':budgets},indent=2))
if __name__=='__main__':main()

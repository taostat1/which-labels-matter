"""Regenerate manuscript displays from the frozen-policy refinement replay.

Run with D:/anaconda3/python.exe report_revision.py.  --partial writes only the
numeric summary; --write-complete-families updates displays whose jobs are all
present, while retaining all other existing manuscript displays.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'
EXP = ROOT / 'experiments'
PRIMARY = {'attribution': 'selective_paired', 'coverage': 'selective_paired',
           'cache': 'current', 'fivefold': 'paired_moments',
           'simulation': 'paired_weighted', 'boundary': 'paired_weighted'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')


def write(name, lines):
    (PAPER / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def count(row):
    return int(row.get('total_labels', row.get('labels_total')))


def val(row, key):
    if key == 'acceptable':
        return bool(row.get(key, row.get('cv_acceptable', False)))
    if key == 'covered':
        return bool(row.get(key, row.get('coverage', False)))
    if key == 'fallback':
        return bool(row.get(key, row.get('branch') in
                            ('exact_fallback', 'active_exact_completion')))
    return row.get(key)


def signature(job):
    return (job['case'], job.get('scenario', ''), job['g'], job['lam'])


def stats(rows):
    """Rows carry _pool from the enclosing job, never inferred from rep."""
    pools = sorted(set(r['_pool'] for r in rows))
    means = np.array([np.mean([count(r) for r in rows if r['_pool'] == p])
                      for p in pools])
    counts = np.array([count(r) for r in rows], dtype=float)
    histories = [h for r in rows for h in r.get('history', [])]
    initial = [float(r['initial_epsilon']) for r in rows
               if r.get('initial_epsilon') is not None]
    end = [float(r['epsilon']) for r in rows if r.get('epsilon') is not None]
    s = dict(mean=float(counts.mean()), pool_sd=float(means.std(ddof=1))
             if len(means) > 1 else 0., path_sd=float(counts.std(ddof=1))
             if len(counts) > 1 else 0., pool_ids=pools,
             pool_means=means.tolist(), runs=len(rows),
             covered=sum(val(r, 'covered') for r in rows),
             acceptable=sum(val(r, 'acceptable') for r in rows),
             errors=sum(not val(r, 'acceptable') for r in rows),
             fallback=sum(val(r, 'fallback') for r in rows),
             refined_paths=sum(any(h.get('refined', False)
                                   for h in r.get('history', [])) for r in rows),
             refinements=sum(h.get('refined', False) for h in histories),
             draws_mean=float(np.mean([r.get('draws', 0) for r in rows])),
             decisions={k: sum(r.get('choice') == k for r in rows)
                        for k in ('A', 'B', 'unresolved')},
             entry_stops=sum(r.get('draws', 0) == 0 and
                             r.get('labels_before_final',-1) == 0 and
                             r.get('choice') in ('A', 'B') for r in rows),
             minimum=int(counts.min()), maximum=int(counts.max()))
    if initial:
        s['initial_epsilon_mean'] = float(np.mean(initial))
        s['initial_epsilon_range'] = [min(initial), max(initial)]
    if end:
        s['stopping_epsilon_mean'] = float(np.mean(end))
        s['stopping_epsilon_range'] = [min(end), max(end)]
        ratios=[float(r['epsilon']/r['initial_epsilon']) for r in rows
                if r.get('epsilon') is not None and r.get('initial_epsilon',0)>0]
        if ratios:s['stopping_to_initial_ratio_mean']=float(np.mean(ratios))
    if all('labels_before_final' in r for r in rows):
        s['before_final_mean'] = float(np.mean([r['labels_before_final'] for r in rows]))
        s['new_final_training_mean'] = float(np.mean([count(r)-r['labels_before_final'] for r in rows]))
    if rows and all('cache_total' in r for r in rows):
        s['touched_objects_mean'] = float(np.mean([
            r['cache_total']['touched_objects'] for r in rows]))
        s['fully_labeled_objects_mean'] = float(np.mean([
            r['cache_total']['fully_labeled_objects'] for r in rows]))
    return s


def contrast(our, ref, seed):
    assert our['pool_ids'] == ref['pool_ids']
    x, y = np.array(our['pool_means']), np.array(ref['pool_means'])
    ii = np.random.default_rng(seed).integers(0, len(x), (20000, len(x)))
    ratio = 1 - x[ii].mean(1) / y[ii].mean(1)
    return dict(saving=float(1-x.mean()/y.mean()),
                saving_ci95=np.quantile(ratio, [.025, .975]).tolist(),
                better_pools=int((x < y).sum()), tied_pools=int((x == y).sum()))


def aggregate(documents, family):
    groups = defaultdict(list)
    for d in documents:
        if d['job']['family'] == family:
            groups[signature(d['job'])].append(d)
    cases = []
    for sig, docs in groups.items():
        docs.sort(key=lambda d: d['job']['pool'])
        j = docs[0]['job']; methodrows = defaultdict(list)
        old_methodrows = defaultdict(list)
        for d in docs:
            for r in d['records'] + d['original_comparators']:
                methodrows[r['method']].append(dict(r, _pool=d['job']['pool']))
            for r in d['baselines']:
                old_methodrows[r['method']].append(dict(r, _pool=d['job']['pool']))
        primary = PRIMARY[family]
        methodrows['fixed_bound'] = old_methodrows[primary]
        ms = {m: stats(rr) for m, rr in methodrows.items()}
        ts = [d['target'] for d in docs]
        c = dict(case=j['case'], domain=j['domain'], g=j['g'], lam=j['lam'],
                 alpha=j.get('alpha'), signal=j.get('signal'),
                 scenario=j.get('scenario'), pools=len(docs), methods=ms,
                 target_range=[min(ts), max(ts)],
                 regimes=dict(A_only=sum(t < -j['g'] for t in ts),
                              both=sum(abs(t) <= j['g'] for t in ts),
                              B_only=sum(t > j['g'] for t in ts)),
                 initial_epsilon_range=[min(d['initial_epsilon'] for d in docs),
                                        max(d['initial_epsilon'] for d in docs)],
                 actual_error_range=[min(d['actual_error'] for d in docs),
                                     max(d['actual_error'] for d in docs)],
                 contrasts={m: contrast(ms[primary], s, 202611145001)
                            for m, s in ms.items()})
        c['all_affected_fixed_bounds'] = {m: stats(rr)
                                         for m, rr in old_methodrows.items()}
        c['historical_policy_new_runs'] = sum(len(d['records']) for d in docs)
        c['displayed_runs'] = sum(m['runs'] for name,m in ms.items() if name!='fixed_bound')
        c['displayed_acceptable'] = sum(m['acceptable'] for name,m in ms.items() if name!='fixed_bound')
        c['displayed_covered'] = sum(m['covered'] for name,m in ms.items() if name!='fixed_bound')
        c['displayed_fallback'] = sum(m['fallback'] for name,m in ms.items() if name!='fixed_bound')
        if family=='coverage':
            figure_methods=['full_pool_cv','active_cv','whole_object_purchases','selective_paired']
            c['figure_counts']={key:sum(ms[m][key] for m in figure_methods)
                                for key in ['runs','covered','acceptable','errors','fallback']}
        if family in ('coverage', 'attribution'):
            c['full_labels'] = 4096 * (6 if j['domain'] == 'image' else 8)
            c['final_labels'] = 768 if j['domain'] == 'image' else 512
            for m, s in ms.items():
                s['saving_vs_full'] = 1-s['mean']/c['full_labels']
                if 'touched_objects_mean' in s:
                    r = 6 if j['domain'] == 'image' else 8
                    s['saving_from_untouched_objects'] = 1-s['touched_objects_mean']/4096
                    s['saving_from_missing_attributes'] = max(0., (r*s['touched_objects_mean']-s['mean'])/c['full_labels'])
        # Full trace: pick a real paired improvement if present, otherwise rep0/pool0.
        rr = methodrows[primary]
        improved = [r for r in rr if r.get('saved_labels', 0) > 0]
        selected = max(improved, key=lambda r: r['saved_labels']) if improved else rr[0]
        c['representative_path'] = {k: v for k, v in selected.items()
                                    if k not in ('final_parameter', 'final_fit')}
        cases.append(c)
    return sorted(cases, key=lambda c: (0 if c['domain'] in ('image','backbone') else 1,
                                       c['scenario'] or '', c['alpha'] or 0,
                                       c['lam'], c['signal'] or 0,
                                       c['case'], c['g']))


def cell(s, scale=1, decimals=1, bold=False, sd='pool_sd'):
    v = f'{s["mean"]/scale:,.{decimals}f}'.replace(',', '{,}')
    if bold:
        v = r'\mathbf{' + v + '}'
    return '$' + v + r'_{\color{gray}' + f'{s[sd]/scale:.{decimals}f}' + '}$'


def real_order(cases):
    return sorted(cases, key=lambda c: (c['domain'] != 'image', c['g']))


def attribution_table(cases):
    cases = real_order(cases)
    methods = [('active_cv', 'Active CV'), ('selective_paired', r'\textbf{LEC}'),
               ('whole_object_purchases', 'Whole-object purchases'),
               ('no_early_stop', 'No early stopping'),
               ('separate_bound', 'Separate error bound')]
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.1}',
             r'\begin{tabularx}{\linewidth}{@{}l*{4}{>{\centering\arraybackslash}X}@{}}\toprule',
             r' & \multicolumn{3}{c}{ADE20K: DINOv2-S vs DINOv2-B} & GoEmotions\\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-5}',
             r'Tolerance $g$ & .0025 & .00375 & .005 & .005\\',
             r'Penalty $\lambda$ & .5 & .5 & .5 & .125\\\midrule']
    for method, label in methods:
        cells = [cell(c['methods'][method], 1000, 2,
                      abs(c['methods'][method]['mean'] -
                          min(c['methods'][m]['mean'] for m, _ in methods)) < 1e-8)
                 for c in cases]
        lines.append(label + ' & ' + ' & '.join(cells) + r'\\')
    lines += [r'\midrule\rowcolor{black!5}Saving vs active CV (\%) & ' +
              ' & '.join(f'{100*c["contrasts"]["active_cv"]["saving"]:.1f}' for c in cases) + r'\\',
              r'95\% pool-bootstrap interval & ' + ' & '.join(
                  '[' + ', '.join(f'{100*v:.1f}' for v in c['contrasts']['active_cv']['saving_ci95']) + ']'
                  for c in cases) + r'\\',
              r'Pools: $A$ only / both / $B$ only & ' + ' & '.join(
                  ' / '.join(str(c['regimes'][k]) for k in ('A_only', 'both', 'B_only')) for c in cases) + r'\\',
              r'\bottomrule\end{tabularx}',
              r'\caption{Complete labels (thousands) in the tolerance grid. Each cell averages 20 pools and three paths per pool; grey subscripts give standard deviations of pool means. Bold marks column minima, including ties. Whole-object and separate-bound variants also refine their approximation bounds. No early stopping completes all queries.}',
              r'\label{tab:attribution}', r'\end{table}']
    write('attribution_table.tex', lines)


def simulation_table(cases, boundary=False):
    cases = sorted(cases, key=lambda c: (c['alpha'], c['lam'], c['signal']))
    methods = ['paired_weighted', 'paired_uniform', 'separate_weighted', 'active_cv']
    keycols = 2 if boundary else 3
    lines = [(r'\begin{table}[H]\centering\small' if boundary else r'\begin{table}[htbp]\centering\small'),
             r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}',
             r'\begin{tabularx}{\linewidth}{@{}' + 'r'*keycols + r'*{4}{>{\centering\arraybackslash}X}r@{}}\toprule',
             (r'$\alpha$ & $s$' if boundary else r'$\alpha$ & $\lambda$ & $s$') +
             r' & \textbf{LEC} & \shortstack{Uniform\\terms} & \shortstack{Separate\\bounds} & Active CV & Saving (\%)\\\midrule']
    for i, c in enumerate(cases):
        if i and c['alpha'] != cases[i-1]['alpha']:
            lines.append(r'\midrule')
        keys = [c['alpha'], c['signal']] if boundary else [c['alpha'], c['lam'], c['signal']]
        best = min(c['methods'][m]['mean'] for m in methods)
        cells = [cell(c['methods'][m], bold=abs(c['methods'][m]['mean']-best)<1e-8) for m in methods]
        lines.append(' & '.join([f'{k:g}' for k in keys] + cells +
                               [f'{100*c["contrasts"]["active_cv"]["saving"]:.1f}']) + r'\\')
    lines += [r'\bottomrule\end{tabularx}']
    if boundary:
        caption = r'Mean complete labels at $\lambda=.125$. Grey subscripts give standard deviations over 100 pools; bold marks row minima. All moment methods refine their approximation bounds from acquired labels.'
        label, name = 'tab:simulation_boundary', 'simulation_boundary_table.tex'
    else:
        caption = r'Mean complete labels over 100 pools, with standard deviations in grey. Bold marks row minima, including ties. Counts include 256 final-training labels; savings are relative to active CV. All moment methods refine their approximation bounds.'
        label, name = 'tab:simulation_labels', 'simulation_label_table.tex'
    lines += [r'\caption{' + caption + '}', r'\label{' + label + '}', r'\end{table}']
    write(name, lines)


def fivefold_table(cases):
    lookup = {(c['case'], c['scenario']): c for c in cases}
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}',
             r'\begin{tabularx}{\linewidth}{@{}l*{4}{>{\centering\arraybackslash}X}@{}}\toprule',
             r' & \multicolumn{2}{c}{Training size $n=64$} & \multicolumn{2}{c}{Training size $n=256$}\\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Case & \textbf{LEC} & Saving (\%) & \textbf{LEC} & Saving (\%)\\\midrule']
    for case in ('T1', 'T2', 'T3', 'T4'):
        cells = [case]
        for scenario in ('KFold_n64', 'KFold_n256'):
            c = lookup[(case, scenario)]; ms = c['methods']; best = min(ms[m]['mean'] for m in ms)
            cells += [cell(ms[m], bold=abs(ms[m]['mean']-best)<1e-8, sd='path_sd')
                      for m in ('paired_moments',)]
            cells += [f'{100*c["contrasts"]["active_cv"]["saving"]:.2f}']
        lines.append(' & '.join(cells)+r'\\')
    lines += [r'\bottomrule\end{tabularx}',
              r'\caption{Complete labels for ordinary fivefold CV. Active CV acquires all 640 labels at $n=64$ and all 2,560 at $n=256$. Grey subscripts give standard deviations across 30 paths in three pools. Savings use unrounded means relative to active CV. This check invokes active CV directly when $\varepsilon_0\ge g$.}',
              r'\label{tab:fivefold_labels}', r'\end{table}']
    write('fivefold_table.tex', lines)


def coverage_figure(cases):
    cases = real_order(cases)
    methods = [('full_pool_cv', 'Full annotation', 'black!50'),
               ('active_cv', 'Active CV', 'paperblue'),
               ('whole_object_purchases', 'Whole objects', 'paperclay'),
               ('selective_paired', r'\textbf{LEC}', 'papersage')]
    lines = [r'\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily\fontsize{8.7}{10.5}\selectfont]']
    for i, c in enumerate(cases):
        x, y = (0 if i%2 == 0 else 9.35), (4.5 if i<2 else 0)
        title = 'ADE20K' if c['domain'] == 'image' else 'GoEmotions'
        lines += [rf'\begin{{scope}}[shift={{({x},{y})}}]',
                  r'\path[draw=black!22,fill=white,line width=.5pt,rounded corners=3pt] (0,0) rectangle (9.05,4.25);',
                  r'\node[anchor=west,font=\sffamily\bfseries\fontsize{10}{12}\selectfont] at (.22,3.93) {' +
                  f'({"abcd"[i]}) {title}: $g={c["g"]:g}$' + '};',
                  r'\node[anchor=west,font=\sffamily\bfseries\fontsize{8.7}{10.5}\selectfont] at (.22,3.46) {Method};',
                  r'\node[align=center] at (4.52,3.46) {Total labels ($\times10^3$) $\downarrow$};',
                  r'\node[align=center] at (7.74,3.46) {Saved vs full (\%) $\uparrow$};']
        best = min(c['methods'][m]['mean'] for m,_,_ in methods)
        for k, (m, label, colour) in enumerate(methods):
            yy = 2.86-.6*k; s=c['methods'][m]; total=s['mean']/1000
            saving=100*(1-s['mean']/c['full_labels']); isbest=abs(s['mean']-best)<1e-8
            lines += [rf'\node[anchor=west] at (.22,{yy:.2f}) {{{label}}};',
                      rf'\fill[black!5] (3.18,{yy-.065:.3f}) rectangle (4.53,{yy+.065:.3f});',
                      rf'\fill[{colour}] (3.18,{yy-.065:.3f}) rectangle ({3.18+1.35*total/35:.6f},{yy+.065:.3f});',
                      rf'\node[anchor=west] at (4.67,{yy:.2f}) '+'{'+cell(s,1000,2,isbest)+'};',
                      rf'\fill[black!5] (6.62,{yy-.065:.3f}) rectangle (7.92,{yy+.065:.3f});']
            if saving>0:
                lines.append(rf'\fill[{colour}] (6.62,{yy-.065:.3f}) rectangle ({6.62+1.3*saving/90:.6f},{yy+.065:.3f});')
            v=f'{saving:.1f}'; v=r'\mathbf{'+v+'}' if isbest else v
            lines.append(rf'\node[anchor=east] at (8.80,{yy:.2f}) '+'{$'+v+'$};')
        lines += [r'\draw[black!25,line width=.35pt] (3.18,.60)--(4.53,.60);',
                  r'\draw[black!25,line width=.35pt] (6.62,.60)--(7.92,.60);',
                  r'\node[anchor=north,text=black!50] at (3.18,.55) {0};',
                  r'\node[anchor=north,text=black!50] at (4.53,.55) {35};',
                  r'\node[anchor=north,text=black!50] at (6.62,.55) {0};',
                  r'\node[anchor=north,text=black!50] at (7.92,.55) {90};',r'\end{scope}']
    lines.append(r'\end{tikzpicture}');write('full_coverage_bars.tikz', lines)
    displayed=[c['methods'][m] for c in cases for m,_,_ in methods]
    runs=sum(m['runs'] for m in displayed); bad=sum(m['errors'] for m in displayed)
    fallback=sum(m['fallback'] for m in displayed)
    outcome = f'{runs-bad} of {runs} choices are acceptable'
    outcome += ', with no fallback.' if fallback == 0 else f'; {fallback} runs use fallback.'
    write('full_coverage_table.tex', [r'\begin{figure}[htbp]\centering',
          r'\resizebox{\linewidth}{!}{\input{full_coverage_bars.tikz}}',
          r'\caption{Complete labels and savings for full validation coverage. Each pool has 4,096 objects. ADE20K has six attributes and final $n=128$; GoEmotions has eight and $n=64$. Means and grey standard deviations summarize 20 pool means, each based on three paths. Savings use unrounded means; bold marks the best values. '+outcome+r' Appendix~\ref{app:full_validation_protocol} reports coverage.}',
          r'\label{fig:full_validation}',r'\end{figure}'])


def tolerance_plot(cases, documents):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = [c for c in real_order(cases) if c['domain']=='image']
    docs = sorted([d for d in documents if d['job']['family']=='attribution'
                   and d['job']['domain']=='image' and d['job']['g']==.0025],
                  key=lambda d:d['job']['pool'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'mathtext.fontset':'cm','pdf.fonttype':42,'ps.fonttype':42})
    fig, axs=plt.subplots(1,2,figsize=(7.2,3),gridspec_kw={'width_ratios':[1.1,1]})
    risk=-np.array([d['target'] for d in docs]); ax=axs[0]
    ax.scatter(np.arange(1,21),risk,s=20,color='#365E83',zorder=3)
    for g,col in [(.0025,'#438477'),(.00375,'#A86B50'),(.005,'#6D6683')]:
        ax.axhline(g,color=col,lw=1,ls='--',alpha=.9)
        ax.text(22.9,g,f'$g={g:g}$',ha='right',va='bottom',fontsize=8,color=col)
    ax.set_title('(a) One or two acceptable candidates',loc='left',fontweight='bold',fontsize=10)
    ax.set_ylabel(r'Risk advantage of A: $-\Delta_{\mathcal{V}}$')
    ax.set_xlabel('Independent pool draw');ax.set_xticks([1,5,10,15,20]);ax.set_xlim(0,23)
    ax.set_ylim(min(.0025,risk.min())*.8,max(.005,risk.max())*1.15)
    ax.set_yticks([.0025,.00375,.005],['.00250','.00375','.00500'])
    ax=axs[1];xs=np.arange(3)
    for m,label,col,shift,style in [('active_cv','Active CV','#A86B50',-.04,'-'),
                                  ('selective_paired','LEC','#438477',.04,'-')]:
        ss=[c['methods'][m] for c in cases]
        ax.errorbar(xs+shift,[s['mean']/1000 for s in ss],
                    yerr=[s['pool_sd']/1000 for s in ss],marker='o',ms=4,capsize=3,
                    lw=1.2,color=col,label=label,linestyle=style)
    ax.set_title('(b) Complete label use',loc='left',fontweight='bold',fontsize=10)
    ax.set_xticks(xs,['.0025','.00375','.005']);ax.set_xlabel('Prespecified tolerance $g$')
    ax.set_ylabel('Distinct labels (thousands)');ax.set_ylim(0,23)
    ax.legend(frameon=False,fontsize=8,loc='upper center')
    for ax in axs:
        ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.16)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(PAPER/'figures/tolerance_grid.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'tolerance_grid.png',dpi=190,bbox_inches='tight');plt.close(fig)


def normalized_cache(documents):
    """Match public entry in both arms, so T2 is never credited to refinement."""
    path=EXP/'cache_normalized.json'
    if not path.exists():return []
    rows=read(path);groups=defaultdict(list)
    for x in rows:groups[signature(x['job'])].append(x)
    out=[]
    for sig,rr in groups.items():
        j=rr[0]['job']; ms={}
        for key,name in [('record','cache_refined'),('fixed_bound_baseline','fixed_bound')]:
            ms[name]=stats([dict(x[key],_pool=x['job']['pool']) for x in rr])
        cache_pools={x['job']['pool'] for x in rr}
        history=[dict(r,_pool=d['job']['pool']) for d in documents
                 if d['job']['family']=='cache' and signature(d['job'])==sig
                 and d['job']['pool'] in cache_pools
                 for r in d['original_comparators'] if r['method']=='residual_sampling']
        if history:ms['historical_residual_sampling']=stats(history)
        assert ms['cache_refined']['pool_ids']==ms['fixed_bound']['pool_ids']
        displayed={key:sum(ms[m][key] for m in ['cache_refined','historical_residual_sampling'])
                   for key in ['runs','covered','acceptable','errors','fallback']} if history else {}
        out.append(dict(case=j['case'],domain=j['domain'],g=j['g'],lam=j['lam'],
                        pools=len(ms['cache_refined']['pool_ids']),methods=ms,
                        contrast=contrast(ms['cache_refined'],ms['fixed_bound'],202611145002),
                        displayed_counts=displayed,
                        public_entry_paths=sum(x['record'].get('draws',0)==0 for x in rr)))
    return sorted(out,key=lambda c:(c['case']=='T4-separated',c['case'],c['g']))


def cache_table(cases):
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\begin{tabular}{@{}lrrr@{}}\toprule',
             r'Setting, $g$ & LEC & Fixed bound & Phased sampling\\\midrule']
    for case in cases:
        label = r'T4$^\ast$' if case['case']=='T4-separated' else case['case']
        row = [f"{label}, {case['g']:g}".replace(', 0.', ', .')]
        methods = [case['methods'][m] for m in
                   ('cache_refined','fixed_bound','historical_residual_sampling')]
        best = min(m['mean'] for m in methods)
        for m in methods:
            mean = f"{m['mean']:.1f}"
            if abs(m['mean']-best)<1e-8:
                mean = r'\textbf{' + mean + '}'
            row.append(f"{mean} ({m['pool_sd']:.1f})")
        lines.append(" & ".join(row) + r"\\")
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Mean distinct labels (pool standard deviation) across eight pools per setting, with three paths per pool. Bold marks all minima. Table~\ref{tab:configs} defines the candidates. T4$^\ast$ uses $\alpha=.2$.}',
              r'\label{tab:cache_ablation}',r'\end{table}']
    write('cache_summary_table.tex',lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true')
    ap.add_argument('--write-complete-families',action='store_true');args=ap.parse_args()
    protocol=read(EXP/'protocol.json'); docs=[read(p) for p in sorted((EXP/'results').glob('job_*.json'))]
    expected=defaultdict(int);done=defaultdict(int)
    for j in protocol['jobs']:expected[j['family']]+=1
    for d in docs:done[d['job']['family']]+=1
    complete={f:done[f]==n for f,n in expected.items()}
    if not args.partial and not args.write_complete_families:
        assert all(complete.values()),dict(done=done,expected=expected)
    summary=dict(expected_jobs=dict(expected),completed_jobs=dict(done),complete=complete,
                 new_runs=sum(len(d['records']) for d in docs),
                 new_covered=sum(val(r,'covered') for d in docs for r in d['records']),
                 new_acceptable=sum(val(r,'acceptable') for d in docs for r in d['records']),
                 new_fallback=sum(val(r,'fallback') for d in docs for r in d['records']),
                 notes=['All new rows preserve the corresponding historical entry and preparation policy.',
                        'Fixed-bound LEC is retained only for local verification, not displayed in the paper.',
                        'Cache-table results require a separate matched-public-entry normalization.'],families={})
    for f in expected:
        cases=aggregate(docs,f);summary['families'][f]=cases
        if args.partial or not complete[f]:continue
        if f=='attribution':attribution_table(cases);tolerance_plot(cases,docs)
        elif f=='coverage':coverage_figure(cases)
        elif f=='simulation':simulation_table(cases)
        elif f=='boundary':simulation_table(cases,True)
        elif f=='fivefold':fivefold_table(cases)
    normalized=normalized_cache(docs)
    summary['cache_normalized']=normalized
    summary['family_counts']={f:{key:sum(c['displayed_'+key] for c in cases)
                                for key in ['runs','covered','acceptable','fallback']}
                              for f,cases in summary['families'].items()}
    if normalized:
        summary['family_counts']['cache']={key:sum(c['displayed_counts'][key] for c in normalized)
                                           for key in ['runs','covered','acceptable','fallback']}
    if (not args.partial and len(normalized)==11 and
            all(c['pools']==8 and c['methods']['cache_refined']['runs']==24 for c in normalized)):
        cache_table(normalized)
    save(ROOT/'numeric_updates.json',summary)
    print(json.dumps(dict(completed_jobs=dict(done),new_runs=summary['new_runs'],
                          covered=summary['new_covered'],acceptable=summary['new_acceptable'],
                          fallback=summary['new_fallback']),indent=2))


if __name__=='__main__':
    main()

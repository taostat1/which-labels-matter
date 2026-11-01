from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent;P=ROOT/'paper'
P.mkdir(parents=True, exist_ok=True)
S=json.loads((ROOT/'summary.json').read_text('utf-8'))
T=json.loads((ROOT/'timing_summary.json').read_text('utf-8'))
lookup={(s['alpha'],s['lam'],s['signal']):s for s in S}
methods=['paired_weighted','paired_uniform','separate_weighted','active_cv']
names={'paired_weighted':'Proposed','paired_uniform':'Uniform terms','separate_weighted':'Separate bounds','active_cv':'Active CV'}
colors={'paired_weighted':'#286988','paired_uniform':'#B18A40','separate_weighted':'#AC5E51','active_cv':'#656B75'}
styles={'paired_weighted':'-','paired_uniform':'--','separate_weighted':'-.','active_cv':':'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11.5,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(11.1,3.8));fig.subplots_adjust(left=.075,right=.99,bottom=.26,top=.85,wspace=.34)
x=np.arange(3);lams=[.125,.5,2.]
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#888888')
 ax.set_xticks(x,['0.125','0.5','2']);ax.set_xlabel(r'Penalty $\lambda$');ax.grid(axis='y',color='#DBE1E5',lw=.6,zorder=0)
a=axes[0];v=[lookup[(.1,lam,4.)] for lam in lams]
a.plot(x,[s['separate_mean']/.01 for s in v],color=colors['separate_weighted'],marker='s',ls='--',lw=1.7,label='Separate bound')
a.plot(x,[s['epsilon_mean']/.01 for s in v],color=colors['paired_weighted'],marker='o',lw=2,label='Paired bound')
a.plot(x,[s['mean_abs_error']/.01 for s in v],color='#363A40',marker='^',ls=':',lw=1.6,label='Actual error, s=4')
a.axhline(1,color='#8C8C8C',lw=.8);a.set_yscale('log');a.set_ylabel('Error / tolerance');a.set_title('(a) Approximation error',loc='left',fontweight='bold')
a.legend(loc='upper right',fontsize=8.1,frameon=False,handlelength=1.8,borderaxespad=.3)
for ax,signal,title in zip(axes[1:],[4.,0.],['(b) Signal present: s = 4','(c) No signal: s = 0']):
 for method in methods:
  vals=[lookup[(.1,lam,signal)]['methods'][method] for lam in lams]
  ax.errorbar(x,[v['mean'] for v in vals],yerr=[1.96*v['se'] for v in vals],color=colors[method],ls=styles[method],marker='o' if method=='paired_weighted' else 's',markersize=4,lw=2 if method=='paired_weighted' else 1.45,capsize=2,zorder=4 if method=='paired_weighted' else 3)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Mean complete labels');ax.set_ylim(0,6100)
handles=[Line2D([0],[0],color=colors[m],ls=styles[m],lw=2,label=names[m]) for m in methods]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.64,.005),ncol=2,frameon=False,fontsize=10,columnspacing=1.8)
(P/'figures').mkdir(exist_ok=True)
fig.savefig(P/'figures/controlled_simulation.pdf',bbox_inches='tight');fig.savefig(ROOT/'controlled_simulation.png',dpi=170,bbox_inches='tight');plt.close(fig)

def number(v):return f'{v:,.1f}'.replace(',',r'{,}')
text=r'''\begin{table}[htbp]\centering\small
\caption{Complete label use in all 12 simulation configurations. Entries are means over 100 independent pools, with standard deviations in grey subscripts. The final fit requires 256 labels. Savings are relative to active CV and use unrounded means; negative values indicate more labels.}
\label{tab:simulation_labels}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.13}
\begin{tabularx}{\linewidth}{@{}rrr*{4}{>{\centering\arraybackslash}X}r@{}}\toprule
$\alpha$ & $\lambda$ & $s$ & \textbf{Proposed} & \shortstack{Uniform\\terms} & \shortstack{Separate\\bounds} & Active CV & Saving (\%)\\\midrule
'''
for i,s in enumerate(S):
 if i==6:text+=r'\midrule'+'\n'
 cells=[]
 for m in methods:
  d=s['methods'][m];mean=number(d['mean']);mean=r'\mathbf{'+mean+'}' if m=='paired_weighted' else mean
  cells.append('$'+mean+r'_{\color{gray}'+f"{d['sd']:.1f}"+'}$')
 text+=f"{s['alpha']:g} & {s['lam']:g} & {s['signal']:g} & "+' & '.join(cells)+f" & {s['saving_active']:.1f}"+r'\\'+'\n'
text+=r'\bottomrule\end{tabularx}\end{table}'+'\n'
(P/'simulation_label_table.tex').write_text(text,encoding='utf-8')

text=r'''\begin{table}[htbp]\centering\small
\caption{Decision regimes, serial computation time, and fallback in the simulation. ``Both'' counts pools in which both candidates are acceptable; only $A$ is acceptable in the remaining pools. Each method cell reports mean seconds / exact-fallback count. Times average serial replays of the first five formal pools; fallback counts use all 100 pools. Serial replays reproduce the same labels and decisions and add no statistical replicates.}
\label{tab:simulation_execution}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}
\begin{tabularx}{\linewidth}{@{}rrrr*{4}{>{\centering\arraybackslash}X}@{}}\toprule
$\alpha$ & $\lambda$ & $s$ & Both & \textbf{Proposed} & \shortstack{Uniform\\terms} & \shortstack{Separate\\bounds} & Active CV\\\midrule
'''
for i,s in enumerate(S):
 if i==6:text+=r'\midrule'+'\n'
 cells=[]
 for m in methods:
  t=next(t for t in T if t['name']==s['name'] and t['method']==m)['mean_seconds'];f=s['methods'][m]['fallback']
  cells.append(f'{t:.2f} / {f}')
 text+=f"{s['alpha']:g} & {s['lam']:g} & {s['signal']:g} & {s['regimes']['both']} & "+' & '.join(cells)+r'\\'+'\n'
text+=r'\bottomrule\end{tabularx}\end{table}'+'\n'
(P/'simulation_execution_table.tex').write_text(text,encoding='utf-8')
print('Simulation figure and two complete tables written.')

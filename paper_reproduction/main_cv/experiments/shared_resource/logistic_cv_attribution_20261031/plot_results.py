from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;P=ROOT/'paper'
P.mkdir(parents=True, exist_ok=True)
summary=json.loads((ROOT/'results/summary.json').read_text('utf-8'))
groups=json.loads((ROOT/'results/groups.json').read_text('utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'mathtext.fontset':'cm','pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={'width_ratios':[1.1,1]})
risk=np.array([-x['target'] for x in groups if x['domain']=='image'])
ax=axs[0];ax.scatter(np.arange(1,21),risk,s=20,color='#315d85',zorder=3)
for g,color in [(.0025,'#348476'),(.00375,'#b18436'),(.005,'#9a6c84')]:
    ax.axhline(g,color=color,lw=1,ls='--',alpha=.9)
    ax.text(22.9,g,f'$g={g:g}$',ha='right',va='bottom',fontsize=8,color=color)
ax.set_title('(a) One or two acceptable candidates',loc='left',fontweight='bold',fontsize=10)
ax.set_ylabel(r'Risk advantage of A: $-\Delta_{\mathcal{V}}$')
ax.set_xlabel('Independent pool draw');ax.set_xticks([1,5,10,15,20]);ax.set_xlim(0,23)
ax.set_ylim(min(.0025,risk.min())*.8,max(.005,risk.max())*1.15)
ax.set_yticks([.0025,.00375,.005]);ax.set_yticklabels(['.00250','.00375','.00500'])
ax=axs[1];xs=np.arange(3)
for method,label,color,shift in [('active_cv','Active CV','#bd673f',-.04),('selective_paired','Selective moments','#315d85',.04)]:
    ms=[summary['image-g'+str(g)]['methods'][method] for g in [.0025,.00375,.005]]
    ax.errorbar(xs+shift,[m['mean']/1000 for m in ms],yerr=[m['pool_sd']/1000 for m in ms],
                marker='o',ms=4,capsize=3,lw=1.4,color=color,label=label)
ax.set_title('(b) Complete label use',loc='left',fontweight='bold',fontsize=10)
ax.set_xticks(xs,['.0025','.00375','.005']);ax.set_xlabel('Prespecified tolerance $g$')
ax.set_ylabel('Distinct labels (thousands)');ax.set_ylim(0,23)
ax.legend(frameon=False,fontsize=8,loc='upper center',ncol=1)
for ax in axs:
    ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.16)
fig.tight_layout(w_pad=1.8)
(P/'figures').mkdir(exist_ok=True)
fig.savefig(P/'figures/tolerance_grid.pdf',bbox_inches='tight')
fig.savefig(ROOT/'tolerance_grid.png',dpi=190,bbox_inches='tight')
plt.close(fig)

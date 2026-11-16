"""Summarize the registered paid-anchor comparison without rerunning policies."""
from pathlib import Path
import collections,csv,html,json,statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
rows=[];complete=[];diagnostics=[]
for path in sorted((ROOT/'results').glob('*/*/pool_*/pilot_*/records.json')):
    if not (path.parent/'complete.json').exists():continue
    part=json.loads(path.read_text('utf-8'))
    assert len(part)==json.loads((path.parent/'complete.json').read_text('utf-8'))['paths']
    masks=np.load(path.parent/'masks.npz')
    public=json.loads((path.parent/'public.json').read_text('utf-8'))
    for r in part:
        assert sum(r['stage_labels'].values())==r['total_labels']
        assert int(np.unpackbits(masks[f"{r['g']}_{r['rep']}"]).sum())==r['total_labels']
        assert r['pilot_labels']==r['pilot_objects']*({'image':6,'text':8}[r['domain']])
        if r['pilot_objects']==0:assert all(r['archive_parity'].values())
        assert r['actual_error']<=r['epsilon']+2e-8
        r['source']=str(path.relative_to(ROOT));rows.append(r)
    complete.append(str(path.parent.relative_to(ROOT)))
    diagnostics.append(dict(study=part[0]['study'],domain=part[0]['domain'],pool=part[0]['pool'],
        pilot_objects=part[0]['pilot_objects'],epsilon=public['epsilon'],native_epsilon=public['native_epsilon'],
        mass=public['mass'],actual_error=part[0]['actual_error'],sampling_variance=part[0]['sampling_variance']))
assert rows,'No completed results'
zero={(r['study'],r['domain'],r['pool'],r['rep'],r['g']):r for r in rows if r['pilot_objects']==0}
groups=collections.defaultdict(list)
paired=[]
for r in rows:
    key=(r['study'],r['domain'],r['pool'],r['rep'],r['g'])
    z=zero.get(key)
    if z:
        assert abs(r['target']-z['target'])<1e-10
        paired.append(dict(study=r['study'],domain=r['domain'],g=r['g'],pool=r['pool'],rep=r['rep'],
            pilot_objects=r['pilot_objects'],zero_labels=z['total_labels'],paid_labels=r['total_labels'],
            difference=r['total_labels']-z['total_labels'],
            variance_ratio=r['sampling_variance']/z['sampling_variance'] if z['sampling_variance'] else None))
    groups[(r['study'],r['domain'],r['g'],r['pilot_objects'])].append(r)
mean=lambda a:statistics.fmean(a)
summary=[]
for key,rs in sorted(groups.items()):
    study,domain,g,m=key
    pools=sorted({r['pool'] for r in rs})
    pool_means=[mean(r['total_labels'] for r in rs if r['pool']==p) for p in pools]
    bases=[zero[(study,domain,r['pool'],r['rep'],g)]['total_labels'] for r in rs]
    avg=mean(r['total_labels'] for r in rs)
    summary.append(dict(study=study,domain=domain,g=g,pilot_objects=m,paths=len(rs),pools=len(pools),
        labels_mean=avg,pool_sd=statistics.stdev(pool_means) if len(pool_means)>1 else 0.,
        saving_vs_zero_pct=100*(1-avg/mean(bases)),
        pilot_labels_mean=mean(r['pilot_labels'] for r in rs),
        query_labels_mean=mean(r['stage_labels'].get('comparison',0) for r in rs),
        fallback_labels_mean=mean(r['stage_labels'].get('exact_cv_fallback',0) for r in rs),
        final_new_labels_mean=mean(r['new_final_training_labels'] for r in rs),
        draws_mean=mean(r['draws'] for r in rs),fallback=sum(r['fallback'] for r in rs),
        acceptable=sum(r['acceptable'] for r in rs),covered=sum(r['coverage'] for r in rs),
        epsilon_mean=mean(r['epsilon'] for r in rs),epsilon_max=max(r['epsilon'] for r in rs),
        epsilon_over_g=mean(r['epsilon']/g for r in rs),
        actual_error_mean=mean(r['actual_error'] for r in rs),
        variance_mean=mean(r['sampling_variance'] for r in rs)))
def write_csv(name,records):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
write_csv('summary.csv',summary)
write_csv('paired_paths.csv',paired)
write_csv('pool_diagnostics.csv',diagnostics)
flat=[{k:r[k] for k in ['study','domain','pool','rep','g','pilot_objects','pilot_labels','total_labels',
    'new_final_training_labels','draws','fallback','choice','acceptable','coverage','target','epsilon',
    'mass','actual_error','sampling_variance','source']} for r in rows]
write_csv('paths.csv',flat)
checks=dict(completed_polynomials=len(complete),expected_polynomials=60,method_paths=len(rows),expected_paths=360,
    zero_paths=sum(r['pilot_objects']==0 for r in rows),archive_parity_all=True,label_union_masks_checked=True,
    targets_unchanged=True,actual_errors_within_bounds=True,acceptable=sum(r['acceptable'] for r in rows),
    covered=sum(r['coverage'] for r in rows),fallback=sum(r['fallback'] for r in rows),
    all_registered_results_complete=len(rows)==360 and len(complete)==60,new_experiment=True)
if checks['all_registered_results_complete']:
    expected={(s,d,p,m) for s in ['attribution','coverage'] for d in ['image','text'] for p in [0,1,2] for m in [0,16,32,64,128]}
    observed={(r['study'],r['domain'],r['pool'],r['pilot_objects']) for r in rows}
    assert expected==observed
    assert all(r['paths']==9 for r in summary)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
intro='原论文模型上的付费非零 Newton 锚点试验'
parts=[f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{intro}</title>',
    '<style>body{max-width:1250px;margin:36px auto;font:16px/1.65 system-ui;color:#23384b;padding:0 20px}table{border-collapse:collapse;width:100%;margin:18px 0 36px;font-size:14px}th,td{border-bottom:1px solid #ccd4da;padding:8px;text-align:right}th{background:#edf2f6}td:first-child,th:first-child{text-align:left}.best{font-weight:700;color:#286a61}h2{margin-top:34px}code{background:#f2f4f6}a{color:#32648b}</style>',
    f'<h1>{intro}</h1><p>2026-11-14。固定原论文候选、正则化、训练规模、划分、检查点、查询种子和最终子集。',
    'ADE20K：6属性、22维共享设计、n=128、λ=0.5；GoEmotions：8属性、20维共享设计、n=64、λ=0.125。',
    '每项研究使用原正式实验前3个池，每池3条配对路径；先导规模为0、16、32、64、128个对象。</p>',
    '<p>零先导直接调用当前公开算法，并核对历史逐路径结果。付费组在已购标签上拟合各候选锚点，代入已知标签，保留一次及二次标签项。',
    '使用条件近似误差界及CP区间；完整标签为先导、查询、精确CV回退和最终训练标签的去重并集。',
    '所有配置均报告，审计真值仅在决策后使用。</p>',
    '<p>正的节省率表示付费锚点优于零锚点。粗体标出每一配置内最低平均完整标签数。标准差按三个池的均值计算。</p>']
for study in ['attribution','coverage']:
    parts.append('<h2>'+{'attribution':'重复随机划分','coverage':'全验证覆盖'}[study]+'</h2>')
    for domain,g in sorted({(r['domain'],r['g']) for r in summary if r['study']==study}):
        ss=[r for r in summary if r['study']==study and r['domain']==domain and r['g']==g]
        best=min(r['labels_mean'] for r in ss)
        parts.append(f'<h3>{"ADE20K" if domain=="image" else "GoEmotions"}，g={g}</h3>')
        parts.append('<table><tr><th>先导对象</th><th>先导标签</th><th>完整标签均值 ± 池间SD</th><th>比零锚点节省</th><th>ε均值</th><th>实际误差均值</th><th>合格/路径</th><th>回退</th></tr>')
        for r in ss:
            style=' class="best"' if r['labels_mean']==best else ''
            parts.append(f'<tr{style}><td>{r["pilot_objects"]}</td><td>{r["pilot_labels_mean"]:.0f}</td><td>{r["labels_mean"]:.1f} ± {r["pool_sd"]:.1f}</td><td>{r["saving_vs_zero_pct"]:+.1f}%</td><td>{r["epsilon_mean"]:.6g}</td><td>{r["actual_error_mean"]:.3g}</td><td>{r["acceptable"]}/{r["paths"]}</td><td>{r["fallback"]}</td></tr>')
        parts.append('</table>')
parts.append('<h2>核查与文件</h2><pre>'+html.escape(json.dumps(checks,indent=2))+'</pre>')
parts.append('<p><a href="summary.csv">完整汇总</a> · <a href="paired_paths.csv">配对标签差</a> · <a href="pool_diagnostics.csv">逐池近似与方差</a> · <a href="paths.csv">逐路径记录</a> · <a href="../protocol.json">固定方案</a></p></html>')
(OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(checks))
for r in summary:print(r['study'],r['domain'],r['g'],r['pilot_objects'],round(r['labels_mean'],1),round(r['saving_vs_zero_pct'],1),r['fallback'])

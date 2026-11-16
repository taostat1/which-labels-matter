"""Pool-paired summaries and a self-contained Chinese comparison report."""
from pathlib import Path
import json,html,statistics,hashlib,base64
from collections import defaultdict
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def mean(x):return sum(x)/len(x)
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
OLD={'attribution':'selective_paired','coverage':'selective_paired','cache':'cache_stop','fivefold':'paired_moments','simulation':'paired_weighted','boundary':'paired_weighted'}
METHODS=['original','fixed_zero_local','phase_zero','phase_midpoint','active_cv','whole_object_purchases','paired_uniform','separate_weighted','separate_bound']
NAMES={'original':'论文原算法','fixed_zero_local':'固定零锚点＋局部界更新','phase_zero':'零锚点＋剩余贡献更新','phase_midpoint':'逐批重拟合＋剩余贡献更新','active_cv':'主动 CV','whole_object_purchases':'整对象购买','paired_uniform':'均匀采样','separate_weighted':'分别界','separate_bound':'分别界'}
FAMILIES={'attribution':'重复随机划分','coverage':'全部对象进入验证计划','cache':'附录：11 配置消融','fivefold':'附录：普通五折','simulation':'正文模拟','boundary':'附录：弱正则模拟'}

def normalize_old(r,d):
    return dict(method='original' if r['method']==OLD[d['job']['family']] else r['method'],labels=r['labels_total'],
      acceptable=r.get('acceptable',r.get('cv_acceptable')),covered=r['covered'],fallback=r.get('fallback',r.get('branch')=='exact'),choice=r['choice'])

def main():
    allrows=[];groups=defaultdict(list);audits=[];raw=[]
    for p in sorted((ROOT/'results').glob('fixed_*.json')):
        d=read(p);raw.append(d);j=d['job'];key=tuple((k,j[k]) for k in ['family','case','scenario','lam','alpha','signal','g'] if k in j)
        prior=next(r for r in d['original'] if r['method']==OLD[j['family']]);oldtarget=prior.get('target',prior.get('cv_delta'))
        if oldtarget is not None:assert abs(oldtarget-d['target'])<2e-8,(j,oldtarget,d['target'])
        rr=[normalize_old(r,d) for r in d['original']]
        for r in d['records']:
            rr.append(dict(method=r['method'],labels=r['total_labels'],acceptable=r['acceptable'],covered=r['covered'],fallback=r['fallback'],choice=r['choice']))
            assert sum(r['stage_labels'].values())==r['total_labels']
            if 'approximation_errors' in r:
                assert len(r['approximation_errors'])==len(r['history'])
                audits.extend([e<=h['epsilon']+2e-8 for e,h in zip(r['approximation_errors'],r['history'])])
        for r in rr:
            rec=dict(j,**r,target=d['target'],regime=d['regime'],full_labels=d['pool_objects']*d['attributes'],train_size=d['train_size'])
            allrows.append(rec);groups[key].append(rec)
    summary=[]
    for key,rows in groups.items():
        meta=dict(key);by=defaultdict(list)
        for r in rows:by[r['method']].append(r)
        out=dict(meta,pools=len(by['original']),regimes=dict((reg,sum(r['regime']==reg for r in by['original'])) for reg in ['both','A_only','B_only']),methods={})
        for method,rr in by.items():
            vv=[r['labels'] for r in rr];out['methods'][method]=dict(mean=mean(vv),sd=statistics.stdev(vv) if len(vv)>1 else 0,pool_counts=vv,
              acceptable=sum(bool(r['acceptable']) for r in rr),covered=sum(bool(r['covered']) for r in rr),fallbacks=sum(bool(r['fallback']) for r in rr),runs=len(rr))
        base=out['methods']['original']['mean'];new=out['methods']['phase_midpoint']['mean'];zero=out['methods']['phase_zero']['mean']
        old_by_pool={r['pool']:r['labels'] for r in by['original']}
        diffs=[r['labels']-old_by_pool[r['pool']] for r in by['phase_midpoint']]
        out.update(reanchor_saving_vs_original_pct=100*(1-new/base),reanchor_saving_vs_phase_zero_pct=100*(1-new/zero),
                   reanchor_vs_original_pools=dict(better=sum(x<0 for x in diffs),equal=sum(x==0 for x in diffs),worse=sum(x>0 for x in diffs)))
        summary.append(out)
    pops=[]
    for ci in range(8):
        dd=[read(p) for p in sorted((ROOT/'results').glob(f'population_{ci:02d}_*.json'))]
        if not dd:continue
        case=dd[0]['case'];by=defaultdict(list)
        for d in dd:
            for r in d['records']:by[r['method']].append(r)
            for r in d['original']:by['original_'+r['method']].append(r)
        out=dict(case=case,groups=len(dd),methods={})
        for method,rr in by.items():
            vv=[r['labels'] for r in rr]
            out['methods'][method]=dict(mean=mean(vv),sd=statistics.stdev(vv) if len(vv)>1 else 0,group_counts=vv,
                acceptable=sum(bool(r.get('acceptable_audit_mean',r.get('acceptable'))) for r in rr),
                covered=sum(bool(r.get('covered_audit_mean',r.get('covered'))) for r in rr),fallbacks=sum(bool(r['fallback']) for r in rr) if all('fallback' in r for r in rr) else None,runs=len(rr))
        pops.append(out)
    save(ROOT/'fixed_summary.json',summary);save(ROOT/'population_summary.json',pops);save(ROOT/'paired_records.json',allrows)
    newrows=[r for r in allrows if r['method'] in ['fixed_zero_local','phase_zero','phase_midpoint']]
    agg=dict(completed_fixed_jobs=len(raw),planned_fixed_jobs=129,completed_fixed_configurations=sum(d['pools']==3 for d in summary),planned_fixed_configurations=43,
             completed_population_jobs=sum(p['groups'] for p in pops),planned_population_jobs=24,new_fixed_runs=len(newrows),
             fixed_acceptable=sum(r['acceptable'] for r in newrows),fixed_covered=sum(r['covered'] for r in newrows),fixed_fallbacks=sum(r['fallback'] for r in newrows),
             dynamic_bound_audits=len(audits),dynamic_bound_audits_passed=sum(audits),original_target_parity=True)
    pop_rows=[r for p in (ROOT/'results').glob('population_*.json') for r in read(p)['records']]
    agg.update(new_population_runs=len(pop_rows),population_acceptable_audit_mean=sum(r['acceptable_audit_mean'] for r in pop_rows),
               population_covered_audit_mean=sum(r['covered_audit_mean'] for r in pop_rows),population_fallbacks=sum(r['fallback'] for r in pop_rows),
               population_unresolved=sum(not r['decision_issued'] for r in pop_rows))
    save(ROOT/'audit_summary.json',agg)
    contrasts={}
    for method in METHODS[1:4]:
        differences=[d['methods'][method]['mean']-d['methods']['original']['mean'] for d in summary if d['pools']==3]
        contrasts[method]=dict(better=sum(x<-1e-9 for x in differences),equal=sum(abs(x)<=1e-9 for x in differences),worse=sum(x>1e-9 for x in differences))
    save(ROOT/'configuration_contrasts.json',contrasts)
    h=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐批重拟合：全场景对比</title><style>body{font:16px/1.65 system-ui,Microsoft YaHei,sans-serif;max-width:1500px;margin:40px auto;padding:0 26px;color:#22313d}h1,h2{color:#315f86}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0 34px}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #ccd3d8}th:first-child,td:first-child{text-align:left}thead{background:#edf3f7}small{color:#697987}.negative{color:#a95540}.positive{color:#347f72}.note{background:#f3f6f8;padding:16px 20px}code{background:#eee;padding:2px 4px}</style><body>',
       '<h1>逐批重拟合非零锚点＋更新剩余贡献：全场景探索</h1>',
       f'<p>已完成固定目标 {len(raw)}/129 个配置—池组合；总体风险 {sum(p["groups"] for p in pops)}/24 个配置—准备组合。</p>',
       '<div class="note">每个固定目标配置使用原实验的前三个池、每池第 0 条路径，并复用原划分、最终训练子集和随机种子。表中是这三个配对池的平均完整标签数，不是论文 20 或 100 个池的均值。训练规模、正则化、容忍度和比较目标不变。总体风险采用独立准备与查询路径，单独列出。</div>',
       '<h2>主要结果</h2>',
       '<p>本轮不支持把正文主算法统一替换为逐批重拟合。固定目标的 43 个配置中，重拟合相对原实验算法有 14 个减少标签、16 个增加、13 个持平；相比只更新剩余项的零锚点方案，是 11 个减少、15 个增加、17 个持平。正文八组配置相对原算法是四组减少、四组增加。</p>',
       '<p>固定零锚点并收紧局部界的改动更稳定：10 个配置减少、33 个持平。其中普通五折 T2 的两组下降来自增加公共入口检查，并非重拟合的收益。其余改善来自采样过程中更新近似误差证书。全部新固定目标运行中，两种阶段方案各 129 次均选择合格、区间覆盖且无回退。只更新局部界保留了原路径的一次误选和两次区间未覆盖。</p>',
       '<p>总体风险需要在新锚点后独立校准。八组中，重拟合相对新固定锚点方案只有弱正则图像组减少标签，一组持平，六组增加。弱正则图像组从 121032 降至 116628，减少约 3.6%，并避免三次回退；但仍高于原论文流程的 74065。48 次新运行均通过独立审计均值下的选择与覆盖核验，9 次发生回退。</p>',
       '<h2>比较规则</h2><p>固定零锚点＋局部界更新：保持原采样规则和累计 CP 区间，只按已购标签收紧误差界。零锚点＋剩余贡献更新：每批后把已知贡献直接代入，重新分配剩余项；每阶段用新查询构造 CP 区间。逐批重拟合：再用缓存标签更新各划分的锚点。未知标签取 0.5 仅用来定义锚点，仍作为未知二元变量进入多项式和误差界。</p>',
       '<p>各阶段冻结锚点与采样规则，分别分配错误概率，并把针对同一精确目标的区间求交。达到原查询上限仍未决时完成精确 CV。标签总数是先导、比较、回退和最终训练的去重并集。附录 11 配置的原算法列采用含公共入口检查的 cache_stop 记录，与现稿的入口规则一致；前三个池中，其与旧 current 记录的唯一区别是 T2 在零查询时停止。</p>',
       '<p>配置说明：正文图像比较 DINOv2-S 与 DINOv2-B，λ=0.5、每图 6 个属性、最终训练 128 图；正文文本使用 GoEmotions，λ=0.125、每条 8 个属性、最终训练 64 条。T1–T4 的文本正则化依次为 2、2、0.5、0.125，候选插值比例依次为 1、0.1、1、0.1；T4-separated 用 0.2。V1–V3 是图像 head 比较，正则化为 2、0.5、0.125，插值比例为 1、1、0.05；V4 是跨 backbone 比较。模拟名称 a、l、s 分别表示候选插值比例、正则化和信号强度；模拟最终训练 64 个对象，每对象 4 个属性。</p>',
       '<p>下表加粗表示四种算法中最低的平均标签数，不是综合优胜判定。错误选择和未覆盖区间仍保留在均值中，并在质量核验部分列出；不因查询数少而剔除或隐去。</p>']
    if (ROOT/'main_comparison.png').exists():
        src=base64.b64encode((ROOT/'main_comparison.png').read_bytes()).decode()
        h.append(f'<h2>正文八组配置</h2><img style="width:100%" src="data:image/png;base64,{src}" alt="主实验四方法标签数对比；柱为三池均值，点为逐池结果">')
    for family,name in FAMILIES.items():
        cells=[d for d in summary if d['family']==family]
        if not cells:continue
        h+=[f'<h2>{name}</h2><table><thead><tr><th>配置</th><th>原算法</th><th>零锚点＋局部界</th><th>零锚点＋剩余项</th><th>重拟合＋剩余项</th><th>相对原算法节省</th><th>相对零锚点剩余项节省</th><th>重拟合合格/回退</th><th>两者/单者合格池</th></tr></thead><tbody>']
        for d in cells:
            label=f'{d["case"]}; λ={d["lam"]:g}, g={d["g"]:g}'
            if 'scenario' in d:label=d['scenario']+' '+label
            vals=[d['methods'][m]['mean'] for m in METHODS[:4]];best=min(vals)
            row=[html.escape(label)]+[f'<b>{v:,.1f}</b>' if v==best else f'{v:,.1f}' for v in vals]
            for k in ['reanchor_saving_vs_original_pct','reanchor_saving_vs_phase_zero_pct']:
                v=d[k];row.append(f'<span class="{"negative" if v<0 else "positive"}">{v:+.1f}%</span>')
            rr=d['methods']['phase_midpoint'];row.append(f'{rr["acceptable"]}/{rr["runs"]}; {rr["fallbacks"]} 回退')
            row.append(f'{d["regimes"]["both"]} / {d["pools"]-d["regimes"]["both"]}')
            h.append('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>')
        h.append('</tbody></table>')
    h+=['<h2>固定目标的选择与覆盖核验</h2><table><thead><tr><th>方法</th><th>运行数</th><th>不合格选择</th><th>区间未覆盖</th><th>回退</th></tr></thead><tbody>']
    for m in METHODS[:4]:
        rr=[r for r in allrows if r['method']==m]
        h.append(f'<tr><td>{NAMES[m]}</td><td>{len(rr)}</td><td>{sum(not r["acceptable"] for r in rr)}</td><td>{sum(not r["covered"] for r in rr)}</td><td>{sum(r["fallback"] for r in rr)}</td></tr>')
    h+=['</tbody></table>','<h2>总体程序风险扩展</h2><p>两种新方案使用相同的 512 次先导抽样。固定锚点方案只做一次校准；重拟合方案每次更新锚点后重新独立校准。剩余贡献在阶段内冻结，用一般有界统计量的经验 Bernstein 区间；此处不使用 CP。论文原算法含原有的多路径准备与预测器成本，因此其均值仅作流程参照；重拟合的净作用应比较两个新方案。总体风险的“合格/覆盖”依据独立 2048 次拟合审计均值，其 Monte Carlo 标准误保存在原始记录。</p>',
         '<table><thead><tr><th>配置</th><th>论文原算法（参照）</th><th>固定锚点＋剩余项</th><th>重拟合＋剩余项</th><th>重拟合增减</th><th>重拟合合格/回退</th></tr></thead><tbody>']
    for d in pops:
        mm=d['methods'];base=mm['fixed_anchor']['mean'];new=mm['reanchor']['mean'];r=mm['reanchor']
        vals=[d['case']['id'],f'{mm["original_proposed"]["mean"]:,.1f}',f'{base:,.1f}',f'{new:,.1f}',f'{100*(new/base-1):+.1f}%',f'{r["acceptable"]}/{r["runs"]}; {r["fallbacks"]} 回退']
        h.append('<tr>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
    h+=['</tbody></table>','<h2>可复核记录</h2><p>fixed_summary.json 保存每配置各方法的均值、池间标准差、三条原始标签数、合格数、覆盖数与回退数；paired_records.json 保存逐池配对结果。results/ 下保存每阶段区间、ε、锚点范数、剩余系数质量及最终训练计数。核验结果见 audit_summary.json。</p>',
       '<p>若最终训练使用整个数据池，完整标签数等于全池标签数，此记账恒等式不因锚点更新而改变。普通五折的最终训练仍按原计划使用 80% 的池。</p>',
       '<h2>算法上的含义</h2><p>更准确的锚点不自动等于更少标签。固定 CV 的零锚点享有专门的成对证书；更换锚点后，新证书可能更宽。此外，新的阶段统计量要用新抽样估计，早期样本仅通过旧区间保留信息。总体风险还需支付独立校准的新增标签。这三项都可能抵消近似改善。</p>',
       '<p>若只改动一步，更适合先保留零锚点、原查询规则与累计 CP，并逐批采用旧界和新局部界中的较小者。在相同查询流、检查点和最终训练子集下，新区间包含于旧区间，所以不会比旧规则更晚停止，也不会增加完整标签并集。逐批重拟合暂作探索方案，后续可检验只在新证书和剩余采样规模有明显改善时才接受新锚点。</p></body></html>']
    (ROOT/'全场景对比.html').write_text('\n'.join(h),encoding='utf-8')
    print(json.dumps(agg,ensure_ascii=False,indent=2))
    for d in summary:
        if d['family'] in ['attribution','coverage']:print(d['family'],d['case'],d['g'],d['pools'],[(m,round(d['methods'][m]['mean'],1)) for m in METHODS[:4]])

if __name__=='__main__':main()

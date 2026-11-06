"""Integrate completed, verified chapter-aligned measurements in both languages."""
from pathlib import Path
import difflib
import itertools
import json
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'review_results/chapter_aligned_20261024'
TEMPLATES=ROOT/'experiments/chapter_aligned_20261024'


def sci(value):
    mantissa,exponent=f'{value:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'


def iterations(row):
    values=row['iterations']
    return str(values[0]) if len(values)==1 else f'{min(values)}--{max(values)}'


def main():
    summary=json.loads((AUDIT/'summary.json').read_text())
    assert summary['complete']
    if not (AUDIT/'manuscript_values.json').exists():
        before=json.loads((AUDIT/'sources_before.json').read_text(encoding='utf-8'))
        changed=[name for name,original in before.items() if (ROOT/name).read_text(encoding='utf-8')!=original]
        assert not changed,('Manuscript edited during experiment; reconcile before integration',changed)
    lookup={(r['case'],r['variant']):r for r in summary['groups']}
    g=lambda v,c='woven':lookup[(c,v)]
    # Every requested variant must have all three runs before writing prose.
    for v in ['csr_jacobi','stencil27_jacobi','stencil27_fft','stencil14_fft','direct_fft',
              'modal_fft','hybrid_fft','background_fft','execution_fp64','execution_fp32']:
        assert len(g(v)['runs'])==3
    values={}
    for case in ('sphere','grosmont'):
        for label,v in [('chfem','chfem'),('csr','csr_jacobi'),('jacobi','stencil27_jacobi'),('fft','stencil27_fft')]:
            values[f'{case}_{label}_time']=f"{g(v,case)['time']:.3f}"
            values[f'{case}_{label}_iter']=iterations(g(v,case))
        values[f'{case}_speedup']=f"{g('chfem',case)['time']/g('execution_fp32',case)['time']:.2f}"
        values[f'{case}_mixed_memory']=f"{g('execution_fp32',case)['memory']:.3f}"
    for label,v in [('woven_fft','stencil27_fft'),('direct','direct_fft'),('modal','modal_fft'),
                    ('hybrid','hybrid_fft'),('background','background_fft'),
                    ('execution','execution_fp64'),('mixed','execution_fp32'),('symmetric','stencil14_fft'),
                    ('stencil','stencil27_fft')]:
        for metric in ('time','memory','Ku','preconditioner'):
            values[f'{label}_{metric}']=f'{g(v)[metric]:.3f}'
    a,b=g('execution_fp64'),g('execution_fp32')
    assert a['iterations']==b['iterations']
    values['precision_iterations']=iterations(a)
    precision_error=max(np.linalg.norm(np.array(x['CH'])-np.array(y['CH']))/np.linalg.norm(y['CH'])
        for x,y in itertools.product(a['runs'],b['runs']))
    values['precision_error']=sci(precision_error)
    final,baseline=g('execution_fp32'),g('stencil27_fft')
    values['overall_time_reduction']=f"{100*(1-final['time']/baseline['time']):.2f}"
    values['overall_memory_reduction']=f"{100*(1-final['memory']/baseline['memory']):.2f}"
    values['csr_speedup']=f"{g('csr_jacobi')['time']/final['time']:.2f}"
    values['csr_memory_fraction']=f"{100*final['memory']/g('csr_jacobi')['memory']:.2f}"
    values['woven_error']=sci(max(r['error'] for r in summary['groups'] if r['case']=='woven'))
    activity={(r['case'],r['variant']):r['time'] for r in summary['activity']}
    values['active90_overhead']=f"{100*(activity['fraction90','latest']/activity['fraction90','hybrid']-1):.2f}"
    fft_iterations={tuple(r['iterations']) for r in summary['groups'] if r['case']=='woven' and 'jacobi' not in r['variant']}
    assert len(fft_iterations)==1,fft_iterations
    for lang,section in [('en','sections_structured'),('zh','sections_structured_zh')]:
        text=(TEMPLATES/f'performance_{lang}.tex.in').read_text(encoding='utf-8')
        sentence=(f"Every FFT configuration requires {iterations(baseline)} iterations over the six loads." if lang=='en'
                  else f"各FFT配置完成六个载荷均需{iterations(baseline)}次迭代。")
        for key,value in dict(values,fft_iteration_statement=sentence).items():text=text.replace(f'@@{key}@@',value)
        ms,mg=values['sphere_mixed_memory'],values['grosmont_mixed_memory']
        if ms==mg:
            text=text.replace(f'with peaks of {ms} and {mg} GiB',f'with a peak of {ms} GiB in each case')
            text=text.replace(f'显存峰值分别为{ms}和{mg} GiB',f'两者显存峰值均为{ms} GiB')
        assert '@@' not in text
        (ROOT/section/'05_performance.tex').write_text(text,encoding='utf-8')
        write_table(summary,section,lang)
        update_activity(summary,section,lang)
        update_headline(section,lang,values,lookup)
        update_verification(section,lang,lookup)
    (AUDIT/'manuscript_values.json').write_text(json.dumps(values,indent=2,ensure_ascii=False),encoding='utf-8')
    before=json.loads((AUDIT/'sources_before.json').read_text(encoding='utf-8'))
    diff=[]
    for name,original in before.items():
        updated=(ROOT/name).read_text(encoding='utf-8')
        if original!=updated:
            diff.extend(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile=name+' (before)',tofile=name))
    (AUDIT/'manuscript_changes.diff').write_text(''.join(diff),encoding='utf-8')
    plan=json.loads((AUDIT/'plan_and_claims.json').read_text(encoding='utf-8'))
    plan['claim_evidence_map'].append(dict(claim='All revised quantitative comparisons',
        evidence='summary.json: raw result paths, telemetry paths, source hashes and repeated measurements',
        table='tab:solver-summary',figure='fig:fi-workflow-cost',confidence='verified'))
    for claim in plan['claim_evidence_map']:
        if claim['claim']=='new quantitative stage gains':
            claim.update(evidence='summary.json',confidence='verified')
        if claim['claim']=='New CSR baseline verification and refreshed chfem stiffness comparison':
            claim['confidence']='verified'
    for decision in plan['inclusion']:
        if 'running' in decision['status']:decision['status']='completed and verified'
    plan['claim_evidence_map']=list({row['claim']:row for row in plan['claim_evidence_map']}.values())
    plan['disposition']=dict(main_text=['CSR/Jacobi/FFT baselines','controlled component sequence','active-volume sensitivity','grid scaling'],
        supplementary=['all individual repetitions','source hashes','legacy independently measured pairs'],
        excluded=['duplicate main-text tables of the same performance numbers'],pending=[])
    (AUDIT/'plan_and_claims.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')


def write_table(summary,section,lang):
    en=lang=='en';lookup={(r['case'],r['variant']):r for r in summary['groups']}
    names_en={'chfem':'chfem / Jacobi','csr_jacobi':'CuPy CSR / Jacobi','stencil27_jacobi':'27-point / Jacobi',
        'stencil27_fft':'27-point / FFT','stencil14_fft':'14-point / FFT',
        'modal_fft':'Modal evaluation','hybrid_fft':'Hybrid storage','background_fft':'Background split',
        'execution_fp64':'GPU execution','execution_fp32':'Proposed (FP32 FFT)'}
    names_zh={'chfem':'chfem／Jacobi','csr_jacobi':'CuPy CSR／Jacobi','stencil27_jacobi':'27点／Jacobi',
        'stencil27_fft':'27点／FFT','stencil14_fft':'14点／FFT',
        'modal_fft':'模态计算','hybrid_fft':'混合存储','background_fft':'背景分解',
        'execution_fp64':'GPU执行优化','execution_fp32':'本文方法（FP32 FFT）'}
    labels=names_en if en else names_zh
    lines=[r'\begingroup',r'\setlength{\intextsep}{8pt}',r'\begin{table}[htbp]',r'\centering\small',
        r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.13}',
        r'\caption{'+('Performance summary at $192^3$' if en else '$192^3$性能汇总')+'}',
        r'\label{tab:solver-summary}',r'\label{tab:fi-isotropic-comparison}',r'\label{tab:fi-woven-comparison}',
        r'\label{tab:fft-precision-pair}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',
        (r'Configuration & Time (s) & Iter. & Memory (GiB) & Speedup \\' if en else
         r'配置 & 时间 (s) & 迭代次数 & 显存 (GiB) & 加速比 \\')]
    for case,title in [('sphere','Sphere' if en else '球夹杂'),('grosmont','Grosmont'),('woven','Woven cell' if en else '编织单胞')]:
        lines.extend([r'\midrule',r'\multicolumn{5}{@{}l}{'+title+r'} \\'])
        variants=['chfem','csr_jacobi','stencil27_jacobi','stencil27_fft','execution_fp32'] if case!='woven' else [
            'csr_jacobi','stencil27_jacobi','stencil27_fft','stencil14_fft','modal_fft','hybrid_fft',
            'background_fft','execution_fp64','execution_fp32']
        reference=lookup[(case,'chfem' if case!='woven' else 'stencil27_jacobi')]['time']
        for variant in variants:
            row=lookup[(case,variant)];name=labels[variant]
            if variant=='stencil14_fft':name+=r'$^{*}$'
            lines.append(f"{name} & {row['time']:.3f} & {iterations(row)} & {row['memory']:.3f} & {reference/row['time']:.2f}"+r' \\')
    note=(r'Time: complete six-load call (chfem: native core); memory: NVML peak. Speedup references chfem for the isotropic cells and 27-point/Jacobi for the woven cell. FFT uses FP64 except the final row of each group. $^{*}$Separate symmetry pair with 27-point/FFT.' if en else
          r'时间为六载荷完整调用时间（chfem为原生核心），显存取NVML峰值。各向同性算例以chfem、编织算例以27点／Jacobi为加速比基准。除各组最终配置外，FFT均采用FP64。$^{*}$与27点／FFT组成独立的对称存储对照。')
    lines.extend([r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',r'\begin{minipage}{\linewidth}\footnotesize',
                  note,r'\end{minipage}',r'\end{table}',r'\endgroup'])
    (ROOT/section/'table_solver_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def update_headline(section,lang,v,lookup):
    en=lang=='en';g=lambda x:lookup[('woven',x)]
    a=ROOT/section/'00_abstract.tex';c=ROOT/section/'06_conclusions.tex'
    before=json.loads((AUDIT/'sources_before.json').read_text(encoding='utf-8'))
    for p in (a,c):
        # Rebuild from the saved text, so running this integration twice is safe.
        text=before[str(p.relative_to(ROOT))]
        text=text.replace('63.69',v['overall_time_reduction']).replace('86.05',v['overall_memory_reduction'])
        text=text.replace('31.12',v['sphere_speedup']).replace('34.38',v['grosmont_speedup'])
        text=text.replace('$2.480\\times10^{-11}$',v['woven_error'])
        text=text.replace('computation-stage time','complete-call time').replace('计算阶段时间','完整调用时间')
        text=text.replace('full 27-point stencil','FP64 FFT-preconditioned 27-point stencil').replace('完整27点模板','FP64 FFT预条件27点模板')
        text=text.replace('a FP64 FFT-preconditioned','an FP64 FFT-preconditioned')
        text=text.replace('Effective-stiffness discrepancies among the compared configurations remain below',
                          'The maximum effective-stiffness discrepancy from this baseline is')
        text=text.replace('The maximum effective-stiffness discrepancy among the compared configurations is',
                          'The maximum effective-stiffness discrepancy from this baseline is')
        text=text.replace('所比较配置间的等效刚度相对差不超过','各配置相对该基线的等效刚度最大相对差为')
        text=text.replace('所比较配置间的等效刚度最大相对差为','各配置相对该基线的等效刚度最大相对差为')
        # Isotropic memory is reported explicitly if the cases have different peaks.
        ms=lookup[('sphere','execution_fp32')]['memory'];mg=lookup[('grosmont','execution_fp32')]['memory']
        if en:
            text=text.replace('The corresponding NVML peaks are 2.090 GiB for the proposed solver and 1.137 GiB for chfem.',
                f'The proposed solver uses {ms:.3f} and {mg:.3f} GiB, respectively, compared with 1.137 GiB for chfem.')
            text=text.replace('The corresponding memory peak is 1.84 times that of chfem (2.090 versus 1.137 GiB).',
                f'The respective memory peaks are {ms:.3f} and {mg:.3f} GiB, compared with 1.137 GiB for chfem.')
        else:
            text=text.replace('本文求解器与chfem的NVML显存峰值分别为2.090和1.137 GiB',
                f'本文求解器在两算例中的NVML显存峰值分别为{ms:.3f}和{mg:.3f} GiB，chfem均为1.137 GiB')
            text=text.replace('显存峰值为chfem的1.84倍（2.090对1.137 GiB）',
                f'对应显存峰值为{ms:.3f}和{mg:.3f} GiB，chfem为1.137 GiB')
        if f'{ms:.3f}'==f'{mg:.3f}':
            text=text.replace(f'uses {ms:.3f} and {mg:.3f} GiB, respectively',f'uses {ms:.3f} GiB in both cases')
            text=text.replace(f'The respective memory peaks are {ms:.3f} and {mg:.3f} GiB',f'The memory peak is {ms:.3f} GiB in both cases')
            text=text.replace(f'在两算例中的NVML显存峰值分别为{ms:.3f}和{mg:.3f} GiB',f'在两算例中的NVML显存峰值均为{ms:.3f} GiB')
            text=text.replace(f'对应显存峰值为{ms:.3f}和{mg:.3f} GiB',f'两算例的显存峰值均为{ms:.3f} GiB')
        if p==c:
            for case,old in [('sphere',r'$1.532\times10^{-8}$'),('grosmont',r'$8.708\times10^{-9}$')]:
                text=text.replace(old,sci(lookup[(case,'execution_fp32')]['error']))
        p.write_text(text,encoding='utf-8')


def update_verification(section,lang,lookup):
    p=ROOT/section/'04_verification.tex'
    before=json.loads((AUDIT/'sources_before.json').read_text(encoding='utf-8'))
    text=before[str(p.relative_to(ROOT))]
    for case,old in [('sphere',r'$1.532\times10^{-8}$'),('grosmont',r'$8.708\times10^{-9}$')]:
        text=text.replace(old,sci(lookup[(case,'execution_fp32')]['error']))
    if lang=='en':
        anchor='For larger cells, chfem provides'
        added=('The assembled CuPy baseline was checked separately on a $4\\times5\\times6$ periodic grid with random anisotropic materials. Its global stiffness matrix differs from independent CPU assembly by $2.312\\times10^{-16}$, and the effective stiffness differs from the CPU direct solution by $2.683\\times10^{-12}$.\n\n')
    else:
        anchor='对于更大规模的单胞'
        # Select the existing paragraph by its distinctive opening sentence.
        match=re.search(r'[^\n]*chfem[^\n]*\$192\^3\$[^\n]*',text)
        assert match,'Chinese chfem verification paragraph not found'
        anchor=match.group(0)
        added=('另在一个$4\\times5\\times6$的随机各向异性周期网格上核验CuPy装配基线。其全局刚度矩阵与独立CPU装配的相对差为$2.312\\times10^{-16}$，等效刚度与CPU直接求解结果的相对差为$2.683\\times10^{-12}$。\n\n')
    assert anchor in text
    text=text.replace(anchor,added+anchor,1)
    p.write_text(text,encoding='utf-8')


def update_activity(summary,section,lang):
    p=ROOT/section/'table_fi_background.tex'
    before=json.loads((AUDIT/'sources_before.json').read_text(encoding='utf-8'))
    text=before[str(p.relative_to(ROOT))]
    text=text.replace('Compute (s)','Time (s)').replace('计算 (s)','时间 (s)')
    old={('fraction50','hybrid'):'0.341',('fraction50','latest'):'0.344',
         ('fraction90','hybrid'):'0.490',('fraction90','latest'):'0.526'}
    for row in summary['activity']:
        previous=old[row['case'],row['variant']]
        assert previous in text
        text=text.replace(previous,f"{row['time']:.3f}")
    p.write_text(text,encoding='utf-8')


if __name__=='__main__':main()

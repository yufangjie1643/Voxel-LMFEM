"""Controlled FFT-stage costs at native manuscript width; measured data only."""
from pathlib import Path
import json
import csv
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fi_figure_style import apply_style,export_figure,PHASE,PRESENT,SECOND_PHASE,INK

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'review_results/chapter_aligned_20261024'


def main(summary=None,chain_only=False):
    if summary is None:summary=json.loads((AUDIT/'summary.json').read_text())
    keys=['stencil27_fft','modal_fft','hybrid_fft','background_fft','execution_fp64','execution_fp32']
    rows=[next(r for r in summary['groups'] if r['case']=='woven' and r['variant']==v) for v in keys]
    apply_style()
    matplotlib.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
    reports=[]
    for lang in ('en','zh'):
        en=lang=='en'
        fig=plt.figure(figsize=(140/25.4,83/25.4))
        a=fig.add_axes([.255,.19,.31,.61]);b=fig.add_axes([.715,.19,.265,.61])
        labels=['27-point / FFT','Modal','Hybrid storage','Background split','GPU execution','Mixed precision'] if en else [
            '27点／FFT','模态计算','混合存储','背景分解','GPU执行优化','混合精度']
        colors=['#969696',PHASE,SECOND_PHASE,'#d7d7d7']
        categories=['Preparation','PCG','Load / stress','Other'] if en else ['准备','PCG','载荷／应力','其他']
        for i,r in enumerate(rows):
            y=len(rows)-1-i
            run=sorted(r['runs'],key=lambda x:x['time'])[1]
            offset=0
            for value,color,label in zip(run['stage'],colors,categories):
                a.barh(y,value,left=offset,height=.54,color=color,edgecolor='white',linewidth=.3,
                       label=label if i==0 else None)
                offset+=value
            a.errorbar(r['time'],y,xerr=[[r['time']-r['time_min']],[r['time_max']-r['time']]],
                       fmt='none',color=INK,lw=.7,capsize=2)
            a.text(r['time']+.08,y,f"{r['time']:.2f}",fontsize=7,va='center')
            b.barh(y,r['memory'],height=.54,color=PRESENT if i==len(rows)-1 else PHASE)
            b.text(r['memory']+.25,y,f"{r['memory']:.2f}",fontsize=7,va='center')
        for ax in (a,b):
            ax.set_ylim(-.6,len(rows)-.4)
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        a.set_yticks(np.arange(len(rows))[::-1],labels);b.set_yticks([])
        a.set_xlim(0,1.19*max(r['time_max'] for r in rows))
        b.set_xlim(0,1.19*max(r['memory'] for r in rows))
        a.set_xlabel('Complete-call time (s)' if en else '完整调用时间 (s)')
        b.set_xlabel('Peak GPU memory (GiB)' if en else '显存峰值 (GiB)')
        fig.text(.255,.845,'(a) Time' if en else '(a) 时间',fontsize=9)
        fig.text(.715,.845,'(b) Memory' if en else '(b) 显存',fontsize=9)
        fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.60,.99),ncol=4,
                   columnspacing=1,handlelength=1,handletextpad=.45)
        report=export_figure(fig,'fi_optimization_chain',lang,dict(
            conclusion='Controlled stages separate material-memory savings from operator and iteration-time savings.',
            source=str((AUDIT/'summary.json').relative_to(ROOT)),variants=keys,
            timing='complete-call median; stacked segments from median-time repeat; min-max whiskers over three runs',
            memory='maximum NVML process peak over three runs',
            controls='same 192^3 woven cell and normalized reference; FP64 FFT until final FP32 stage',
            axes='linear from zero'))
        reports.append(report)
        if not chain_only:reports.append(verification(summary,lang))
    (AUDIT/'figure_qa.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


def verification(summary,lang):
    en=lang=='en'
    lookup={(r['case'],r['variant']):r for r in summary['groups']}
    fig=plt.figure(figsize=(140/25.4,73/25.4))
    a=fig.add_axes([.11,.23,.33,.61]);b=fig.add_axes([.64,.23,.33,.61])
    fig.text(.045,.91,'(a) Stiffness agreement' if en else '(a) 刚度一致性',fontsize=9)
    fig.text(.565,.91,'(b) Load-wise discrepancy' if en else '(b) 各载荷的相对差',fontsize=9)
    largest=0;error_rows=[];components=[];sources=[]
    for i,(case,name,color,marker) in enumerate([
        ('sphere','Sphere' if en else '球夹杂',PHASE,'o'),('grosmont','Grosmont',PRESENT,'D')]):
        current=lookup[(case,'execution_fp32')]['runs'];ref=lookup[(case,'chfem')]['runs']
        current=sorted(current,key=lambda x:x['repeat']);ref=sorted(ref,key=lambda x:x['repeat'])
        x,y=np.array(ref[0]['CH'])/1000,np.array(current[0]['CH'])/1000
        a.scatter(x.ravel(),y.ravel(),s=18,marker=marker,facecolors='white',edgecolors=color,
                  linewidths=.75,label=name,zorder=3)
        largest=max(largest,x.max(),y.max())
        errors=np.max([np.linalg.norm(np.array(c['CH'])-r['CH'],axis=0)/np.linalg.norm(r['CH'],axis=0)
                       for c,r in itertools.product(current,ref)],axis=0)
        b.scatter(np.arange(6)+(-.09 if i==0 else .09),errors*1e8,s=18,marker=marker,color=color,zorder=3)
        for j,k in itertools.product(range(6),repeat=2):
            components.append(dict(case=case,row=j+1,column=k+1,reference_GPa=x[j,k],present_GPa=y[j,k],repeat=0))
        error_rows.extend(dict(case=case,load=load,max_column_relative_error=error,repeat_pairs=9)
                          for load,error in zip(('11','22','33','23','13','12'),errors))
        sources.extend(r['source'] for r in current+ref)
    upper=largest*1.07
    a.plot([0,upper],[0,upper],'--',lw=.65,color=INK,zorder=1)
    a.set_xlim(-upper*.025,upper);a.set_ylim(-upper*.025,upper)
    a.set_aspect('equal',adjustable='box')
    a.set_xlabel('chfem (GPa)');a.set_ylabel('Present (GPa)' if en else '本文结果 (GPa)')
    b.set_xlim(-.45,5.45);b.set_ylim(0,np.ceil(max(r['max_column_relative_error'] for r in error_rows)*1e8*2)/2)
    b.set_yticks(np.linspace(0,b.get_ylim()[1],5))
    b.set_xticks(range(6),['11','22','33','23','13','12'])
    b.set_xlabel('Loading component' if en else '载荷分量')
    b.set_ylabel('Column discrepancy (×10⁻⁸)' if en else '刚度列相对差 (×10⁻⁸)',labelpad=4)
    fig.legend(*a.get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.53,.015))
    for name,rows in [('stiffness_components.csv',components),('stiffness_column_errors.csv',error_rows)]:
        with (AUDIT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    return export_figure(fig,'fi_stiffness_verification',lang,dict(sources=sources,
        component_selection='all 36 signed components from repeat 0; MPa converted to GPa',
        error_definition='maximum column-wise Euclidean relative error over all nine repeat pairs',
        axes='linear; error starts at zero; units of 1e-8',
        source_summary=str((AUDIT/'summary.json').relative_to(ROOT))))


if __name__=='__main__':main()

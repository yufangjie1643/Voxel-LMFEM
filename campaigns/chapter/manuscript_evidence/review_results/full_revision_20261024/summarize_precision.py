"""Verify the paired records and regenerate the bilingual precision table."""
from pathlib import Path
from statistics import median
import csv
import hashlib
import itertools
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'fft_precision_192_20261024'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def relative(a, b):
    return float(np.linalg.norm(np.array(a)-b)/np.linalg.norm(b))

def sci(x):
    a,b=f'{x:.3e}'.split('e')
    return '$'+a+r'\times10^{'+str(int(b))+'}$'

def main():
    assert read(RUN/'complete.json')['complete']
    rows=[];records={};hashes={}
    for precision in ['fp32','fp64']:
        runs=[]
        for i in range(3):
            p=RUN/'results'/f'192_{precision}_r{i}.json'
            t=RUN/'logs'/f'192_{precision}_r{i}_telemetry.json'
            r,tele=read(p),read(t)
            assert r['complete'] and r['max_true_residual']<=1e-8
            assert len(r['measurement']['calls'])==6
            assert tele['returncode']==0 and tele['failure'] is None
            assert not tele['initial']['processes'] and tele['measurement_sample_count']>0
            assert [m['kind'] for m in tele['markers']]==['begin','end']
            assert r['precision_metadata']['cache_dtype']=='torch.float'+precision[2:]
            for source,digest in r['sources'].items():
                if source in hashes: assert hashes[source]==digest,source
                hashes[source]=digest
            records[precision,i]=r
            runs.append(dict(time_s=r['workflow_seconds'],compute_s=r['measurement']['seconds'],
                nvml_GiB=tele['peak_process_bytes_measured']/2**30,
                iterations=sum(c['iterations'] for c in r['measurement']['calls']),
                restarts=sum(c['restarts'] for c in r['measurement']['calls']),
                residual=r['max_true_residual'],preconditioner_ms=r['microbenchmark']['preconditioner']['median_ms'],
                record=p.relative_to(ROOT).as_posix(),telemetry=t.relative_to(ROOT).as_posix()))
        rows.append(dict(precision=precision,repeats=3,time_s=median(r['time_s'] for r in runs),
            time_range=[min(r['time_s'] for r in runs),max(r['time_s'] for r in runs)],
            compute_s=median(r['compute_s'] for r in runs),nvml_GiB=max(r['nvml_GiB'] for r in runs),
            iterations=sorted(set(r['iterations'] for r in runs)),restarts=sorted(set(r['restarts'] for r in runs)),
            residual=max(r['residual'] for r in runs),preconditioner_ms=median(r['preconditioner_ms'] for r in runs),
            runs=runs))
    assert len({r['input_hash'] for r in records.values()})==1
    assert len({json.dumps(r['precision_metadata']['reference']) for r in records.values()})==1
    error=max(relative(records['fp32',i]['measurement']['CH'],records['fp64',j]['measurement']['CH']) for i,j in itertools.product(range(3),repeat=2))
    assert error<1e-8
    small={}
    for precision in ['original','fp32','fp64']:
        r=read(HERE/'fft_precision_validation_20261024_c/results'/f'8_{precision}_r0.json')
        small[precision]=r
    original_difference=relative(small['fp32']['measurement']['CH'],small['original']['measurement']['CH'])
    assert original_difference<1e-12
    assert small['fp32']['validation']==small['original']['validation']
    assert small['fp64']['validation']['preconditioner_relative_to_dense_symbol']<2e-12
    summary=dict(complete=True,rows=rows,CH_difference_max_9_pairs=error,
        time_reduction_percent=100*(1-rows[0]['time_s']/rows[1]['time_s']),
        memory_reduction_percent=100*(1-rows[0]['nvml_GiB']/rows[1]['nvml_GiB']),
        original_FP32_small_grid_CH_difference=original_difference,
        dense_symbol_validation={p:r['validation'] for p,r in small.items()},
        input_hash=records['fp32',0]['input_hash'],reference=records['fp32',0]['precision_metadata']['reference'],
        source_sha256=hashes,protocol=records['fp32',0]['protocol'])
    (HERE/'precision_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (HERE/'precision_summary.csv').open('w',newline='',encoding='utf-8') as f:
        keys=['precision','time_s','nvml_GiB','iterations','restarts','residual','preconditioner_ms']
        w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    for lang in ['en','zh']:
        folder=ROOT/('sections_structured' if lang=='en' else 'sections_structured_zh')
        caption=r'FFT precision on the $192^3$ woven cell' if lang=='en' else r'$192^3$ 编织算例的FFT精度对照'
        heading=r'FFT & Complete (s) & NVML (GiB) & Iter. & Restarts & True residual' if lang=='en' else r'FFT & 完整调用 (s) & NVML (GiB) & 迭代 & 重启 & 真实残差'
        table=[r'\begingroup',r'\setlength{\intextsep}{8pt}',r'\begin{table}[H]',r'\centering\small',
            r'\setlength{\tabcolsep}{3pt}',r'\setlength{\abovecaptionskip}{6pt}',r'\setlength{\belowcaptionskip}{5pt}',
            r'\renewcommand{\arraystretch}{1.15}',r'\caption{'+caption+'}',r'\label{tab:fft-precision-pair}',
            r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',heading+r' \\',r'\midrule']
        for row in rows:
            assert len(row['iterations'])==len(row['restarts'])==1
            table.append(' & '.join([row['precision'].upper(),f"{row['time_s']:.3f}",f"{row['nvml_GiB']:.3f}",str(row['iterations'][0]),str(row['restarts'][0]),sci(row['residual'])])+r' \\')
        table.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}',r'\endgroup',''])
        (folder/'table_fft_precision.tex').write_text('\n'.join(table),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','source_sha256','reference']},indent=2))

if __name__=='__main__':main()

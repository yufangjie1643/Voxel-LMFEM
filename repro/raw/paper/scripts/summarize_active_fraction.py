"""Audit the 54 measured active-fraction runs and export figure source data."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'review_results/active_fraction_20261024'
sys.path.insert(0, str(AUDIT))
from analytical_laminate import from_record


def summarize():
    status = json.loads((AUDIT/'raw/status.json').read_text(encoding='utf-8'))
    assert status['complete'] and len(status['completed']) == 54 and status['failed'] is None
    records = []
    for path in sorted((AUDIT/'raw/results').glob('*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        assert d['complete'] and d['grid'] == 128 and d['max_true_residual'] <= 1e-8
        exact = from_record(d)
        error = np.linalg.norm(np.array(d['measurement']['CH'])-exact)/np.linalg.norm(exact)
        assert error < 1e-8
        tpath = AUDIT/'raw/logs'/f'{path.stem}_telemetry.json'
        telemetry = json.loads(tpath.read_text(encoding='utf-8'))
        assert telemetry['returncode'] == 0 and telemetry['failure'] is None
        assert telemetry['measurement_sample_count'] > 0
        assert [m['kind'] for m in telemetry['markers']] == ['begin', 'end']
        records.append(dict(case=d['case'], variant=d['variant'], repeat=d['repeat'],
            fraction=d['input']['active_fraction'], layers=d['input']['layers'],
            time_s=d['workflow_seconds'], ku_ms=d['microbenchmark']['Ku']['median_ms'],
            iterations=sum(c['iterations'] for c in d['measurement']['calls']),
            error=float(error), residual=d['max_true_residual'],
            memory_GiB=telemetry['peak_process_bytes_measured']/2**30,
            source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            telemetry=str(tpath.relative_to(ROOT)), input_hash=d['input_hash'],
            CH=d['measurement']['CH'], reference=d['reference_design'], sources=d['sources']))
    assert len(records) == 54
    source_hashes = {}
    for r in records:
        for name, value in r['sources'].items():
            assert name not in source_hashes or source_hashes[name] == value, name
            source_hashes[name] = value
    snapshot = json.loads((AUDIT/'raw/source_manifest.json').read_text(encoding='utf-8'))
    for name, value in source_hashes.items():
        assert snapshot[name.removeprefix('/home/yufan/code/voxel-mechanics/')] == value
    groups = []
    pairs = []
    for requested in range(10, 100, 10):
        same_case = [r for r in records if r['case'] == f'fraction{requested}']
        assert len(same_case) == 6
        assert len({r['input_hash'] for r in same_case}) == 1
        assert len({json.dumps(r['reference'],sort_keys=True) for r in same_case}) == 1
        assert len({r['iterations'] for r in same_case}) == 1
        assert same_case[0]['fraction'] == round(requested/100*128)/128
        pair_error = max(np.linalg.norm(np.array(a['CH'])-b['CH'])/np.linalg.norm(b['CH'])
            for a in same_case if a['variant']=='latest' for b in same_case if b['variant']=='hybrid')
        assert pair_error < 1e-8
        pair = dict(requested=requested, fraction=same_case[0]['fraction'],
            layers=same_case[0]['layers'], iterations=same_case[0]['iterations'], pair_error=float(pair_error))
        for variant in ('hybrid','latest'):
            runs = [r for r in same_case if r['variant'] == variant]
            assert sorted(r['repeat'] for r in runs) == [0,1,2]
            g = dict(requested=requested, fraction=pair['fraction'], layers=pair['layers'], variant=variant,
                iterations=pair['iterations'], error_max=max(r['error'] for r in runs),
                residual_max=max(r['residual'] for r in runs), source_files=[r['source'] for r in runs])
            for metric in ('time_s','ku_ms'):
                values = [r[metric] for r in runs]
                g[metric] = float(np.median(values))
                g[metric+'_min'] = min(values)
                g[metric+'_max'] = max(values)
            groups.append(g)
            pair[variant] = g
        pair['ku_reduction_percent'] = 100*(1-pair['latest']['ku_ms']/pair['hybrid']['ku_ms'])
        pair['time_reduction_percent'] = 100*(1-pair['latest']['time_s']/pair['hybrid']['time_s'])
        pairs.append(pair)
    report = dict(complete=True, groups=groups, pairs=pairs, records=records, source_hashes=source_hashes,
        max_analytic_error=max(r['error'] for r in records),
        max_pair_error=max(p['pair_error'] for p in pairs),
        max_true_residual=max(r['residual'] for r in records),
        controls='Material layout, hybrid storage, reference and GPU configuration identical within each pair; all six-load iteration totals match.',
        statistic='Median of three independent process repeats; min-max whiskers. Each Ku process value is the median of five 20-call CUDA-event batches.',
        archived_table='Original two-fraction table remains in sections_structured*/table_fi_background.tex; replaced in the manuscript by a nine-fraction figure.')
    (AUDIT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fields = [k for k in groups[0] if k != 'source_files']
    with (AUDIT/'figure_source.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fields); w.writeheader()
        w.writerows({k:g[k] for k in fields} for g in groups)
    print(json.dumps(dict(max_error=report['max_analytic_error'],max_pair=report['max_pair_error'],
        points=[dict(fraction=p['fraction'],ku_reduction=p['ku_reduction_percent'],
                     time_reduction=p['time_reduction_percent']) for p in pairs]),indent=2))
    return report


if __name__ == '__main__':
    summarize()

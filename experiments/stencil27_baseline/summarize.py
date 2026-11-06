"""Check matched stiffness results and summarize freshly measured baselines."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    directory = parser.parse_args().directory
    plan = json.loads((directory/('full_plan.json' if (directory/'full_plan.json').exists() else 'plan.json')).read_text())
    grouped = defaultdict(list)
    sources = {}
    for job in plan:
        name = job['name']
        path = directory/'results'/f'{name}.json'
        row = json.loads(path.read_text())
        telemetry = json.loads((directory/'logs'/f'{name}_telemetry.json').read_text())
        assert row['complete'] and row['max_true_residual'] <= 1e-8
        assert telemetry['failure'] is None and telemetry['measurement_sample_count'] > 0
        assert len(row['measurement']['calls']) == 6
        assert np.isfinite(row['measurement']['CH']).all()
        row['nvml_GiB'] = telemetry['peak_process_bytes_measured']/2**30
        row['record'] = str(path.relative_to(directory))
        grouped[(row['case'], row['grid'], row['variant'])].append(row)
        sources[row['record']] = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = []
    pairs = []
    for (case, grid, variant), runs in grouped.items():
        baseline = grouped[(case, grid, 'stencil27')]
        assert {r['input_hash'] for r in runs+baseline} == {baseline[0]['input_hash']}
        errors = [float(np.linalg.norm(np.asarray(a['measurement']['CH'])-np.asarray(b['measurement']['CH'])) /
                        np.linalg.norm(b['measurement']['CH'])) for a, b in itertools.product(runs, baseline)]
        assert max(errors) < 1e-8, (case, grid, variant, errors)
        def times(values):
            return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)
        group = dict(case=case, grid=grid, variant=variant, repeats=len(runs),
            time=times([r['workflow_seconds'] for r in runs]),
            compute=times([r['measurement']['seconds'] for r in runs]),
            nvml=max(r['nvml_GiB'] for r in runs),
            iterations=sorted(set(sum(c['iterations'] for c in r['measurement']['calls']) for r in runs)),
            allocated=max(r['measurement']['peak_allocated_bytes']/2**30 for r in runs),
            reserved=max(r['measurement']['peak_reserved_bytes']/2**30 for r in runs),
            max_true_residual=max(r['max_true_residual'] for r in runs),
            ku_ms=times([r['microbenchmark']['Ku']['median_ms'] for r in runs]),
            precond_ms=times([r['microbenchmark']['preconditioner']['median_ms'] for r in runs]),
            CH_error_vs27=max(errors), records=[r['record'] for r in runs])
        for prefix in ['rhs','solve','post']:
            group[prefix+'_seconds'] = times([sum(s['seconds'] for s in r['measurement']['stages'] if s['name'].startswith(prefix)) for r in runs])
        group['setup_seconds'] = times([sum(s['seconds'] for s in r['measurement']['stages'] if not s['name'].startswith(('rhs','solve','post'))) for r in runs])
        groups.append(group)
    for group in groups:
        base = next(r for r in groups if (r['case'], r['grid'], r['variant']) == (group['case'], group['grid'], 'stencil27'))
        pairs.append(dict(case=group['case'], grid=group['grid'], variant=group['variant'],
                          compute_reduction_percent=100*(1-group['compute']['median']/base['compute']['median']),
                          nvml_reduction_percent=100*(1-group['nvml']/base['nvml']),
                          CH_error=group['CH_error_vs27']))
    report = dict(passed=True, baseline='full 27-point FI stencil', groups=groups, comparisons=pairs,
                  source_sha256=sources, protocol='Common fresh-process six-load warmup, FP64 physics/FP32 FFT, true residual <=1e-8, NVML sampling every 20 ms')
    (directory/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()

"""Run 54 serial, GPU-exclusive, fresh-process comparisons through ./run."""
import importlib.util
import json
import os
from pathlib import Path
from project_paths import ROOT


def main():
    out = ROOT/'runs/active_fraction_128_20261024'
    out.mkdir(parents=True, exist_ok=True)
    os.environ['VOXEL_RUN_DIR'] = str(out)
    spec = importlib.util.spec_from_file_location('active_runner', ROOT/'repro/bench/solver_comparison/run_jobs.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.RUN_DIR = out
    plan = []
    for repeat in range(3):
        fractions = list(range(10, 100, 10))
        if repeat % 2:
            fractions.reverse()
        for fraction in fractions:
            variants = ('hybrid', 'latest') if (repeat+fraction//10)%2 else ('latest', 'hybrid')
            for variant in variants:
                name = f'fraction{fraction}_128_{variant}_r{repeat}'
                plan.append(dict(name=name, arguments=['study.py'], command=[str(ROOT/'repro/run'),
                    'active-study', '--fraction', str(fraction), '--variant', variant, '--repeat', str(repeat)]))
    status = dict(complete=False, completed=[], failed=None)
    with (out/'plan.json').open('x') as f:
        json.dump(plan, f, indent=2)
    nvml = runner.NVML()
    for job in plan:
        try:
            status['completed'].append(runner.execute(job, nvml, .02))
            runner.save(out/'status.json', status)
            print('DONE', len(status['completed']), '/', len(plan), job['name'], flush=True)
        except BaseException as error:
            status['failed'] = dict(name=job['name'], error=repr(error))
            runner.save(out/'status.json', status)
            raise
    status['complete'] = True
    runner.save(out/'status.json', status)


if __name__ == '__main__':
    main()

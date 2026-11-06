"""Run the matched stencil comparison with common NVML sampling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    from project_paths import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--phase', choices=['smoke', 'formal'], default='smoke')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--grids', nargs='+', type=int,
                        help='Override the case list with these woven grids')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('Use a single run directory name')
    if args.grids and (min(args.grids) < 2 or len(set(args.grids)) != len(args.grids)):
        parser.error('Grids must be distinct integers greater than one')
    if args.grids and args.resume_from:
        parser.error('A resumed campaign uses its archived grid plan')
    output = ROOT/'runs'/args.name
    output.mkdir(exist_ok=False)
    if args.phase == 'smoke':
        cases = [('woven', 8)]
        variants = ['stencil27', 'paper', 'modal', 'latest']
        repeats = 1
    else:
        cases = [('woven', 64), ('woven', 128), ('woven', 192), ('sphere', 192), ('grosmont', 192)]
        variants = ['stencil27', 'paper', 'chain', 'latest']
        repeats = 3
    if args.grids:
        cases = [('woven', grid) for grid in args.grids]
    jobs = []
    for repeat in range(repeats):
        for case, grid in cases:
            selected = variants + (['modal'] if args.phase == 'formal' and case == 'woven' and grid == 192 else [])
            for variant in selected:
                jobs.append(dict(name=f'{case}_{grid}_{variant}_r{repeat}', arguments=[str(ROOT/'run.py')], command=[
                    str(ROOT/'run'), 'stencil-study', '--variant', variant,
                    '--case', case, '--grid', str(grid), '--repeat', str(repeat)]))
    plan = output/'plan.json'
    if args.resume_from:
        previous = args.resume_from.resolve()
        if not previous.is_relative_to(ROOT/'runs'):
            parser.error('Resume source must be under project runs/')
        status = json.loads((previous/'plan_status.json').read_text())
        completed = {r['name'] for r in status['completed']}
        full_plan = json.loads((previous/('full_plan.json' if (previous/'full_plan.json').exists() else 'plan.json')).read_text())
        if (previous/'continuation.json').exists():
            completed.update(json.loads((previous/'continuation.json').read_text())['inherited_completed'])
        for folder in ['results', 'logs']:
            (output/folder).mkdir(exist_ok=True)
        for name in completed:
            for folder, suffix in [('results','.json'),('logs','.log'),('logs','_telemetry.json')]:
                shutil.copy2(previous/folder/(name+suffix), output/folder/(name+suffix))
        (output/'full_plan.json').write_text(json.dumps(full_plan,indent=2)+'\n')
        (output/'continuation.json').write_text(json.dumps(dict(source=str(previous), inherited_completed=sorted(completed)),indent=2)+'\n')
        jobs = [job for job in full_plan if job['name'] not in completed]
    plan.write_text(json.dumps(jobs, indent=2)+'\n')
    env = dict(os.environ, VOXEL_RUN_DIR=str(output))
    subprocess.run([sys.executable, str(ROOT/'experiments/solver_comparison/run_jobs.py'),
                    '--plan', str(plan)], env=env, check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('summarize.py')),
                    '--directory', str(output)], check=True)


if __name__ == '__main__':
    main()

"""Run an immutable comparison plan into a fresh directory under runs/."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from project_paths import ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--plan', choices=['smoke','formal','certification','validation'], default='smoke')
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ('.','..'):
        p.error('Use a single experiment name')
    output = ROOT/'runs'/a.name
    output.mkdir(exist_ok=False)
    directory = ROOT/'experiments/solver_comparison'
    plans = {'smoke':'smoke_plan.json','formal':'plan.json',
             'certification':'certification_plan.json','validation':'validation_plan.json'}
    plan = directory/plans[a.plan]
    env = dict(os.environ, VOXEL_RUN_DIR=str(output))
    subprocess.run([sys.executable,str(directory/'cases.py'),'--grids','8','--cases','sphere'],
                   cwd=directory,env=env,check=True)
    subprocess.run([sys.executable,str(directory/'run_jobs.py'),'--plan',str(plan)],
                   cwd=directory,env=env,check=True)
    print(json.dumps(dict(passed=True,run_directory=str(output),plan=str(plan))))

if __name__ == '__main__':
    main()

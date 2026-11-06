"""Five interleaved fresh-process comparisons on the current strong workload."""
import argparse, subprocess, sys, json, random
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rng=random.Random(610061);order=[]
for repeat in range(5):
    block=[('fp64',repeat),('mixed',repeat)];rng.shuffle(block);order.extend(block)
plan=a.out/'order.json'
if plan.exists():assert json.loads(plan.read_text())==[list(x) for x in order]
else:plan.write_text(json.dumps(order,indent=2))
for precision,repeat in order:
    out=a.out/f'{precision}_{repeat}.json'
    if out.exists():
        assert json.loads(out.read_text())['status']=='passed';continue
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if occupied:raise SystemExit('GPU OCCUPIED: stopped: '+occupied)
    print('RUN',precision,repeat,flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_precision.py')),
        '--config','A11','--precision',precision,'--repeat',str(repeat),
        '--data',str(a.data),'--out',str(out)],check=True)
print('ALL 10 COMPLETE',flush=True)

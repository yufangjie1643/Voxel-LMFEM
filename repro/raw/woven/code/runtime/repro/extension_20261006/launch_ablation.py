"""Pre-specified randomized 75-run design; refuses any occupied GPU."""
import argparse,subprocess,sys,json,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
configs=['A00','A01','A10','A11','F0','F1','U','O27','O14','B1','B2','B8','B16']
rng=random.Random(61006);order=[]
for repeat in range(5):
    block=[(c,'strong',repeat) for c in configs]+[(c,'near',repeat) for c in ['U','A11']]
    rng.shuffle(block);order.extend(block)
plan=a.out/'order.json'
if plan.exists():assert json.loads(plan.read_text())==[list(x) for x in order]
else:plan.write_text(json.dumps(order,indent=2))
for k,(config,material,repeat) in enumerate(order):
    out=a.out/f'{material}_{config}_{repeat}.json'
    if out.exists():
        assert json.loads(out.read_text())['status']=='passed';continue
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if occupied:raise SystemExit('GPU OCCUPIED: stopped: '+occupied)
    print('RUN',k+1,'/',len(order),config,material,repeat,flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_ablation.py')),'--config',config,'--material',material,
       '--repeat',str(repeat),'--data',str(a.data),'--out',str(out)],check=True)
print('ALL 75 COMPLETE',flush=True)

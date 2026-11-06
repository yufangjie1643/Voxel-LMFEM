import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for repeat in range(3):
    for config in (['A00','A01','A10','A11','F0','F1','U'] if repeat!=1 else ['U','F1','F0','A11','A10','A01','A00']):
        out=a.out/f'{config}_{repeat}.json'
        if out.exists():continue
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        if apps:raise SystemExit('GPU OCCUPIED: '+apps)
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_ablation.py')),'--config',config,'--repeat',str(repeat),'--profile','--data',str(a.data),'--out',str(out)],check=True)

import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for N in [64,128,192,256,320]:
    for config in ['U','A11']:
        earlier=list(a.out.glob(config+'_*.json'))
        if any(json.loads(x.read_text()).get('status')=='OOM' for x in earlier):continue
        for repeat in range(2):
            out=a.out/f'{config}_{N}_{repeat}.json'
            if out.exists():continue
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            if apps:raise SystemExit('GPU OCCUPIED: '+apps)
            subprocess.run([sys.executable,str(Path(__file__).with_name('run_resolution.py')),'--grid',str(N),'--config',config,'--repeat',str(repeat),'--out',str(out)],check=True)
            if json.loads(out.read_text())['status']=='OOM':break

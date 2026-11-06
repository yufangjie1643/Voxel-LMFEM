"""Independent HG scale check and public-entry regression after formal jobs."""
import json
from pathlib import Path
import numpy as np

import run_rve
import pipeline_adapter as pipeline
import pipeline_study as study

HERE=Path(__file__).resolve().parent
if json.loads((HERE/"status.json").read_text())["status"]!="complete":
    raise SystemExit("Formal jobs must finish first")
if study.previous.foreign():
    raise SystemExit("GPU busy")
b=study.previous.load_runtime("hg")
torch=b.torch
from femlib.c3d8r import hourglass_scale
torch.manual_seed(20261020)
a=torch.randn((1,210,6,6),device="cuda",dtype=torch.float64)
full=a.transpose(-1,-2)@a+2*torch.eye(6,device="cuda",dtype=torch.float64)
c=torch.stack([full[...,i,j] for i,j in b.PAIRS],dim=-1).contiguous()
h=(.3,.5,.7)
actual=pipeline.extension().scales(c,.1*np.prod(h)/(sum(h)/3)**2).cpu().numpy()[0]
expected=np.array([hourglass_scale(ce,h,.1) for ce in full[0].cpu().numpy()])
error=study.previous.relative(actual,expected)
if error>1e-12:raise RuntimeError("Independent hourglass scale mismatch")
del a,full,c
b.cleanup()
rows=[]
for case in ("strong","near"):
    ref=run_rve.solve_rve("hg",16,"modal",case,warmup=False)
    actual=run_rve.solve_rve("hg",16,"auto",case,warmup=False)
    diff=study.previous.relative(np.asarray(ref["measurement"]["CH"]),np.asarray(actual["measurement"]["CH"]))
    rows.append(dict(kind="hg",case=case,grid=16,variant=actual["variant"],CH_relative=diff,
                     calls=actual["measurement"]["calls"]))
    if diff>1e-8:raise RuntimeError("Public entry regression")
# FI uses a different frozen femlib tree and is checked in a fresh process by CLI.
result=dict(passed=True,independent_hourglass_scale_relative=error,public_entry=rows,
            scope="regression only; cold/JIT timings are not performance evidence")
(HERE/"results/finish_checks.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

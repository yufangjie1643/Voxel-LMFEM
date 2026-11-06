"""Preserve the first 192 probe and add two fresh-process repetitions."""
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
original=here/"results/probe_hg_192_interleaved.json"
first=original.with_name("probe_hg_192_interleaved_r0.json")
if first.exists():raise SystemExit("Repetitions already started")
if not original.exists():raise SystemExit("First successful 192 probe required")
original.rename(first)
for repeat in (1,2):
    target=original.with_name(f"probe_hg_192_interleaved_r{repeat}.json")
    if target.exists():raise FileExistsError(target)
    with (here/"logs"/f"probe_hg_192_interleaved_r{repeat}.log").open("x") as log:
        subprocess.run([sys.executable,"-u",str(here/"gpu_probe.py"),"--grid","192"],
                       cwd=here,stdout=log,stderr=subprocess.STDOUT,check=True)
    original.rename(target)
    print("DONE",repeat,flush=True)

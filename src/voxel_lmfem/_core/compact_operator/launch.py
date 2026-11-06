"""Detach the guarded sequential experiment without overwriting prior state."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
if (here/"status.json").exists() or (here/"launch.json").exists():raise SystemExit("Experiment already exists")
with (here/"orchestrator.log").open("x") as log:
    process=subprocess.Popen([sys.executable,"-u",str(here/"study.py"),"--phase","orchestrate"],cwd=here,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(here/"launch.json").write_text(json.dumps(dict(pid=process.pid,python=sys.executable))+"\n")
print(process.pid)

"""Detach the FI supplement after its small-model GPU checks."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
if (here/"fi_status.json").exists() or (here/"fi_launch.json").exists():raise SystemExit("FI state already exists")
with (here/"fi_orchestrator.log").open("x") as log:
    child=subprocess.Popen([sys.executable,"-u",str(here/"fi_supplement.py"),"--phase","orchestrate"],cwd=here,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(here/"fi_launch.json").write_text(json.dumps(dict(pid=child.pid,python=sys.executable))+"\n")
print(child.pid)

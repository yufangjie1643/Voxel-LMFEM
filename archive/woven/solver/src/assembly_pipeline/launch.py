"""Launch the serial experiment in its own process and retain its PID."""
import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
if (here / "status.json").exists() or (here / "launch.json").exists():
    raise SystemExit("Existing experiment state; choose a new directory")
with (here / "orchestrator.log").open("w") as log:
    child = subprocess.Popen([sys.executable, "-u", str(here / "pipeline_study.py"),
                              "--phase", "orchestrate", "--grid", "192"],
                             cwd=here, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
(here / "launch.json").write_text(json.dumps(dict(pid=child.pid,python=sys.executable)) + "\n")
print(child.pid)

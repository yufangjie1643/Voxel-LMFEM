"""Package completed original GPU evidence with per-file SHA256 hashes."""
import hashlib
import json
from pathlib import Path
import tarfile

here=Path(__file__).resolve().parent
if json.loads((here/"status.json").read_text())["status"]!="complete":raise SystemExit("Incomplete experiment")
if (here/"fi_status.json").exists() and json.loads((here/"fi_status.json").read_text())["status"]!="complete":raise SystemExit("Incomplete FI supplement")
files=[p for p in sorted(here.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
       and p.suffix in (".json",".log",".py",".cu",".cuh",".md") and p.name!="archive_manifest.json"]
manifest={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
dependency=here.parent/"constitutive_basis/structural_modal.cuh"
manifest["dependency_snapshot/structural_modal.cuh"]=hashlib.sha256(dependency.read_bytes()).hexdigest()
(here/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
archive=here/"evidence.tar.gz"
with tarfile.open(archive,"w:gz") as f:
    for p in files+[here/"archive_manifest.json"]:f.add(p,arcname=str(p.relative_to(here)),recursive=False)
    f.add(dependency,arcname="dependency_snapshot/structural_modal.cuh",recursive=False)
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(files))))

"""Archive completed raw evidence with per-file SHA256 hashes."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE=Path(__file__).resolve().parent
state=json.loads((HERE/"status.json").read_text())
if state["status"]!="complete":
    raise SystemExit("Experiment incomplete")
files=[p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
       and p.suffix in (".json",".log",".py",".cu",".cuh",".md")
       and p.name!="archive_manifest.json"]
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(HERE/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
archive=HERE/"evidence_192.tar.gz"
with tarfile.open(archive,"w:gz") as out:
    for p in files+[HERE/"archive_manifest.json"]:
        out.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
print(json.dumps(dict(file=archive.name,bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))

"""Save completed formula evidence with per-file hashes."""
import hashlib,json,tarfile
from pathlib import Path
here=Path(__file__).resolve().parent
if json.loads((here/"status.json").read_text())["status"]!="complete":raise SystemExit("Incomplete experiment")
files=[p for p in here.rglob("*") if p.is_file() and "__pycache__" not in p.parts
       and p.suffix in (".json",".log",".py",".cu",".cuh",".md") and p.name!="archive_manifest.json"]
manifest={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(here/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
archive=here/"evidence_192.tar.gz"
with tarfile.open(archive,"w:gz") as out:
    for p in files+[here/"archive_manifest.json"]:out.add(p,arcname=str(p.relative_to(here)),recursive=False)
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))

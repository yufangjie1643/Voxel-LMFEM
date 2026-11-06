"""Archive completed remote evidence with per-file and archive SHA256 checksums."""
from pathlib import Path
import hashlib
import json
import tarfile


def main():
    root=Path(__file__).resolve().parent
    for status in (root/"status.json",root/"native_controls/status.json"):
        assert json.loads(status.read_text())["status"]=="complete"
    archive=root/"evidence_192.tar.gz"
    if archive.exists():raise FileExistsError(archive)
    files=[]
    for directory in ("results","logs","native_controls"):
        files.extend(p for p in (root/directory).rglob("*") if p.is_file())
    files.extend(p for p in root.iterdir() if p.is_file() and p.suffix in (".py",".cu",".md"))
    files.append(root/"status.json")
    files=sorted(set(files))
    checksums={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    with tarfile.open(archive,"w:gz") as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    manifest=dict(archive=archive.name,archive_bytes=archive.stat().st_size,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=checksums)
    (root/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(archive=str(archive),bytes=manifest["archive_bytes"],files=len(files),sha256=manifest["archive_sha256"])))


if __name__=="__main__":main()

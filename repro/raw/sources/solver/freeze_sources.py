"""Archive the exact runtime sources and input hashes after the GPU campaign.

Usage on the measurement host: python freeze_sources.py
"""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
BASE=Path('/home/yufan/code')


def main():
    expected={}
    for path in (HERE/'results').glob('*.json'):
        r=json.loads(path.read_text())
        for source,digest in r.get('sources',{}).items():
            if source in expected:assert expected[source]==digest,source
            expected[source]=digest
    snapshot=HERE/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    mapping=[]
    for source,digest in sorted(expected.items()):
        p=Path(source);assert p.is_relative_to(BASE),source
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,source
        relative=p.relative_to(BASE);target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        mapping.append(dict(original=source,snapshot=str(target.relative_to(HERE)),sha256=digest))
    inputs={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'inputs').iterdir() if p.is_file()}
    (snapshot/'manifest.json').write_text(json.dumps(dict(sources=mapping,inputs=inputs),indent=2)+'\n')
    print('FROZEN',len(mapping),'runtime sources and',len(inputs),'input files')


if __name__=='__main__':main()

"""Archive measured source identities and native CUDA sources in the run folder."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from project_paths import ROOT
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, required=True)
directory = parser.parse_args().directory.resolve()
assert directory.is_relative_to(ROOT/'runs')
expected = {}
for result in (directory/'results').glob('*.json'):
    row = json.loads(result.read_text())
    for name, digest in row['sources'].items():
        if name in expected:
            assert expected[name] == digest, ('source changed during measurements', name)
        expected[name] = digest
paths = {Path(p) for p in expected}
for path in list(paths):
    native = path.parent/'csrc'
    if native.is_dir():
        paths.update(p for p in native.rglob('*') if p.is_file() and p.suffix in ('.cu','.cpp','.cuh','.h'))
paths.update((ROOT/'experiments/stencil27_baseline').glob('*.py'))
paths.update([ROOT/'run', ROOT/'run.py', ROOT/'project_paths.py'])
manifest = []
with tarfile.open(directory/'source_snapshot.tar.gz','w:gz') as archive:
    for path in sorted(paths):
        assert path.is_relative_to(ROOT)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in expected:
            assert digest == expected[str(path)], ('source changed after measurement', path)
        relative = str(path.relative_to(ROOT))
        archive.add(path, arcname=relative)
        manifest.append(dict(path=relative, sha256=digest, recorded_during_measurement=str(path) in expected,
                             modified_time=path.stat().st_mtime))
(directory/'source_manifest.json').write_text(json.dumps(dict(passed=True, files=manifest),indent=2)+'\n')
print('Archived',len(manifest),'source files, including native CUDA sources.')

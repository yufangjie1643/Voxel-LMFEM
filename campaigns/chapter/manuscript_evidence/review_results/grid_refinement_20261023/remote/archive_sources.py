"""Freeze the exact sources reported by a completed grid-refinement campaign.

Usage: python archive_sources.py --directory /absolute/project/runs/campaign
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    directory=parser.parse_args().directory.resolve()
    root=directory.parent.parent
    assert directory.parent.name=='runs'
    status=json.loads((directory/'plan_status.json').read_text())
    assert status['complete'] and status['failed'] is None
    expected={}
    for completed in status['completed']:
        row=json.loads((directory/'results'/(completed['name']+'.json')).read_text())
        for source,sha in row['sources'].items():
            if source in expected:assert expected[source]==sha
            expected[source]=sha
    for name in ['run','run.py','project_paths.py','experiments/stencil27_baseline/launch.py',
                 'experiments/stencil27_baseline/summarize.py','experiments/solver_comparison/run_jobs.py']:
        path=root/name
        expected[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    entries=[]
    with tarfile.open(directory/'source_snapshot.tar.gz','w:gz') as archive:
        for name,sha in sorted(expected.items()):
            path=Path(name).resolve()
            assert path.is_relative_to(root),name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
            relative=path.relative_to(root).as_posix()
            archive.add(path,arcname=relative,recursive=False)
            entries.append(dict(path=relative,sha256=sha))
    (directory/'source_manifest.json').write_text(json.dumps(dict(files=entries),indent=2)+'\n')
    print('Verified and archived',len(entries),'source files')


if __name__=='__main__':main()

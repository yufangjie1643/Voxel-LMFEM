"""Add isolated comparison commands to the canonical run entry point."""
from pathlib import Path
root = Path('/home/yufan/code/voxel-mechanics')
path = root/'run.py'
text = path.read_text()
entries = {
    'stencil-study': 'study.py',
    'compare-stencils': 'launch.py',
    'verify-stencils-cpu': 'validate_cpu.py',
}
anchor = 'COMMANDS = {\n'
assert text.count(anchor) == 1
for command, file in entries.items():
    line = f'    "{command}": ROOT / "experiments/stencil27_baseline/{file}",\n'
    if f'"{command}":' not in text:
        text = text.replace(anchor, anchor+line)
path.write_text(text)

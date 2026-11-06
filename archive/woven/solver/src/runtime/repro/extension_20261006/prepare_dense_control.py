"""Generate inspectable matched 36-coefficient consumer from frozen source.

No production source is edited. Only storage indexing and extension name change;
PCG, force assembly, thread128 layout and spectral kernels stay the same.
The generated Triton backend is deliberately disabled: tests use native CUDA.
"""
from pathlib import Path
import shutil,re,difflib
P=Path(__file__).resolve().parent
src=P.parents[1]/'repro/source/femlib';dest=P/'dense_control/femlib'
dest.mkdir(parents=True,exist_ok=True)
changes=[]
for f in src.rglob('*'):
    if f.suffix not in ['.py','.cu','.cpp','.h']:continue
    rel=f.relative_to(src);s=f.read_text(encoding='utf8');t=s
    if str(rel).replace('\\','/')=='batched/c3d8r_hourglass_operator.py':
        t=t.replace('for col in range(row, 6)','for col in range(6)')
        t=t.replace('if row == col:\n            continue','if row >= col:\n            continue')
        t=t.replace('21 + 1 + 3','36 + 1 + 3').replace('int(n_elem) * 21','int(n_elem) * 36')
        # This generated control supports CUDA only; no accidentally wrong Triton layout.
        t=t.replace('def _apply_triton(', 'def _unsupported_apply_triton(')
    elif str(rel).replace('\\','/')=='batched/csrc/c3d8r_hourglass_cuda.cu':
        t=re.sub(r'\b21\b','36',t)
        begin=t.index('  // Upper-triangular packed order:')
        end=t.index('\n}',begin)
        body='  // Matched dense36 consumer: same scalar operation order per row.\n'
        for i in range(6):
            body+=f'  result[{i}] = '+' + '.join(f'packed[{6*i+j}] * vector[{j}]' for j in range(6))+';\n'
        t=t[:begin]+body.rstrip()+t[end:]
    elif str(rel).replace('\\','/')=='batched/c3d8r_hourglass_cuda.py':
        t=t.replace('name="voxel_acdm_c3d8r_hourglass_cuda"','name="fead_dense36_matched_cuda"')
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(t,encoding='utf8')
    if s!=t:changes.extend(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(rel),tofile='dense_control/'+str(rel)))
(P/'dense_control.patch').write_text(''.join(changes),encoding='utf8')
print('Created dense36 control; frozen source unchanged')

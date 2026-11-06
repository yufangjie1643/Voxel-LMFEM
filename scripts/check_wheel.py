"""Install the built wheel outside the checkout and exercise its bundled core.

Usage: python scripts/check_wheel.py [wheel-path]
Host NumPy/SciPy are reused; the project itself must load from the new environment.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
wheel = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else next((ROOT/"dist").glob("voxel_lmfem-*.whl"))
environment = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "VOXEL_LMFEM_ROOT", "VOXEL_PROJECT_ROOT") and not k.startswith("FEAD_")}
with tempfile.TemporaryDirectory(prefix="vlmfem-wheel-") as temporary:
    root = Path(temporary).resolve()
    if root.parent != Path(tempfile.gettempdir()).resolve():
        raise RuntimeError("Unexpected temporary directory location")
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root/"env")
    python = root/"env"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    work = root/"outside-checkout"
    work.mkdir()
    subprocess.run([str(python),"-m","pip","install","--no-deps",str(wheel)],
                   cwd=work,env=environment,check=True)
    code = """
from pathlib import Path
import sys
import numpy as np
import voxel_lmfem
from voxel_lmfem.paths import source_path
assert Path(voxel_lmfem.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert source_path('hybrid_operator/operator_kernels.cu').is_file()
result = voxel_lmfem.homogenize(np.zeros((2,3,4),dtype=np.uint8),
    {0:dict(kind='isotropic',E=3500.,nu=.35)},[1.,2.,3.],backend='cpu')
assert abs(result['CH'][0][0]-3500*.65/(1.35*.30)) < 1e-9
print('PASS isolated wheel:',voxel_lmfem.__version__,voxel_lmfem.__file__)
"""
    subprocess.run([str(python),"-c",code],cwd=work,env=environment,check=True)
    subprocess.run([str(python),"-m","voxel_lmfem","example","--out","input"],
                   cwd=work,env=environment,check=True)
    subprocess.run([str(python),"-m","voxel_lmfem","solve","--input","input/homogeneous.npz",
                    "--materials","input/materials.json","--backend","cpu","--output","result.json"],
                   cwd=work,env=environment,check=True)

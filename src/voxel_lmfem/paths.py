"""Locate installed numerical sources independently of the caller's directory."""
import os
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parent / "_core"
_checkout = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("VOXEL_LMFEM_ROOT",
    str(_checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()))).resolve()


def source_path(relative=""):
    return CORE / relative


def configure():
    """Configure the existing numerical modules without a dated source tree."""
    components = dict(FEAD_ABLATION="compact_operator", FEAD_STRUCTURAL="constitutive_basis",
        FEAD_RUNTIME="runtime", FEAD_MODAL="hex8_modal", FEAD_PIPELINE="assembly_pipeline",
        FEAD_FORMULA="reference_medium", FEAD_FFT="spectral", FEAD_CHAIN="solver_chain",
        FEAD_HYBRID="materials", FEAD_OPERATOR="hybrid_operator")
    for key, directory in components.items():
        path = source_path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"Numerical module missing from installation: {path}")
        os.environ[key] = str(path)
    os.environ["FEAD_PACKAGE_ROOT"] = str(ROOT)
    os.environ["VOXEL_PROJECT_ROOT"] = str(ROOT)
    os.environ["FEAD_TOL"] = "1e-8"
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache/voxel-lmfem/torch_extensions"))
    for key, value in (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("MAX_JOBS", "2")):
        os.environ.setdefault(key, value)
    # Legacy numerical modules import this small compatibility interface.
    sys.modules["project_paths"] = sys.modules[__name__]
    for directory in ("runtime/repro/source", "materials", "solver_chain", "hybrid_operator"):
        path = str(source_path(directory))
        if path not in sys.path:
            sys.path.insert(0, path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([str(ROOT), str(Path(__file__).resolve().parents[1]), *filter(None, inherited.split(os.pathsep))]))


def require_checkout():
    if not (ROOT / "repro/bench").is_dir() or not (ROOT / "repro/table_manifest.json").is_file():
        raise ValueError("This command needs a repository checkout. Run it from Voxel-LMFEM, "
                         "or set VOXEL_LMFEM_ROOT to that checkout.")
    return ROOT

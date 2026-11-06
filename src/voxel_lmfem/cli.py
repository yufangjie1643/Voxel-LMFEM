"""Command-line entry points for examples, solves, checks and paper reproduction."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import runpy
import sys

from . import __version__
from .paths import ROOT, configure, require_checkout, source_path

DRIVERS = {
    "chapter-study": "experiments/chapter_aligned/study.py",
    "woven-fraction": "experiments/woven_fraction/study.py",
    "active-study": "experiments/active_fraction/study.py",
    "active-sweep": "experiments/active_fraction/launch.py",
    "stencil-study": "experiments/stencil27_baseline/study.py",
    "compare-stencils": "experiments/stencil27_baseline/launch.py",
    "fft-precision": "experiments/solver_comparison/fft_precision.py",
    "verify-stencils-cpu": "experiments/stencil27_baseline/validate_cpu.py",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _driver(command, arguments):
    require_checkout()
    configure()
    target = ROOT / DRIVERS[command]
    original_directory = Path.cwd()
    original_arguments = sys.argv
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(target.parent))
        for i, value in enumerate(arguments):
            if value in ("--out", "--output", "--fields") and i+1 < len(arguments):
                if not Path(arguments[i+1]).resolve().is_relative_to(ROOT):
                    raise ValueError("Experiment output must stay inside the repository")
        sys.argv = [str(target), *arguments]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        os.chdir(original_directory)
        sys.argv = original_arguments
        if str(target.parent) in sys.path:
            sys.path.remove(str(target.parent))
    return 0


def _prepare(cases, grids, directory):
    require_checkout()
    configure()
    import shutil
    module = _load("voxel_case_inputs", ROOT / "experiments/solver_comparison/cases.py")
    frozen = ROOT / "campaigns/chapter/manuscript_evidence/review_results/solver_study_20261022/remote/inputs"
    archive = ROOT / "campaigns/chapter/solver/data/grosmont"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        for grid in grids:
            name = f"{case}_{grid}.npy"
            destination = directory / name
            if (frozen / name).is_file() and not destination.exists():
                if destination.with_suffix(".json").exists():
                    raise ValueError(f"Metadata exists without input: {destination.with_suffix('.json')}")
                shutil.copyfile(frozen/name, destination)
                shutil.copyfile((frozen/name).with_suffix(".json"), destination.with_suffix(".json"))
            else:
                if case == "grosmont" and not (archive/name).exists():
                    raise ValueError("The included Grosmont archive supports grids 192 and 400")
                module.prepare(case, grid, directory, archive)
            _, metadata = module.load_case(destination)
            results.append(dict(case=case, grid=grid, path=str(destination),
                                array_sha256=metadata["array_sha256"]))
    print(json.dumps(results, indent=2))


def _example(out):
    import numpy as np
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Example directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # The matrix phase of manuscript Table 1; example lengths are illustrative.
    np.savez(out / "homogeneous.npz", phase_id=np.zeros((4,4,4), dtype=np.uint8),
             domain_size=np.array([1.,1.,1.]))
    _save(out / "materials.json", {"0": {"kind": "isotropic", "E": 3500., "nu": .35}})
    print(f"Created {out/'homogeneous.npz'} and {out/'materials.json'}")


def _check(kind, output):
    configure()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if kind == "fi":
        module = _load("voxel_fi_validation", source_path("hybrid_operator/validate.py"))
        _save(output, module.cpu())
    else:
        require_checkout()
        target = ROOT / "validation/material_state/verify.py"
        sys.path.insert(0, str(target.parent))
        module = _load("voxel_state_validation", target)
        # The original verifier writes validation.json; use a dedicated temporary
        # directory and copy the result to the requested new output file.
        import tempfile
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="voxel-state-", dir=output.parent) as temporary:
            temp_path = Path(temporary).resolve()
            if not temp_path.is_relative_to(output.parent.resolve()):
                raise ValueError("Validation temporary directory escaped its output directory")
            module.OUT = temp_path
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
            _save(output, json.loads((temp_path/"validation.json").read_text(encoding="utf-8")))
    print(json.dumps(dict(status="passed", check=kind, output=str(output))))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if sys.flags.optimize:
        raise SystemExit("Run without -O: numerical assertions are required.")
    if argv and argv[0] in DRIVERS:
        return _driver(argv[0], argv[1:])
    parser = argparse.ArgumentParser(prog="voxel-lmfem",
        description="Periodic fully integrated HEX8 elastic homogenization.",
        epilog="Repository experiment commands: " + ", ".join(DRIVERS))
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Report the Python and CUDA environment")
    doctor.add_argument("--require-cuda", action="store_true")
    solve = sub.add_parser("solve", help="Homogenize an NPZ phase field and JSON material table")
    solve.add_argument("--input", type=Path, required=True)
    solve.add_argument("--materials", type=Path, required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    solve.add_argument("--warmup", action="store_true")
    example = sub.add_parser("example", help="Write a small homogeneous input example")
    example.add_argument("--out", type=Path, default=Path("runs/example"))
    check = sub.add_parser("check", help="Run independent numerical verification")
    check.add_argument("kind", choices=("fi", "material-state"))
    check.add_argument("--output", type=Path, required=True)
    reproduce = sub.add_parser("reproduce", help="Verify archived evidence and regenerate paper tables")
    reproduce.add_argument("--out", type=Path)
    prepare = sub.add_parser("prepare-inputs", help="Prepare hash-verified benchmark inputs")
    prepare.add_argument("--cases", nargs="+", choices=("sphere", "grosmont"), default=["sphere","grosmont"])
    prepare.add_argument("--grids", nargs="+", type=int, default=[192])
    prepare.add_argument("--directory", type=Path, default=ROOT/"data/cases")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            import numpy
            import scipy
            info = dict(version=__version__, python=platform.python_version(), platform=platform.system(),
                        numpy=numpy.__version__, scipy=scipy.__version__, cuda_available=False)
            try:
                import torch
                info.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                            cuda_available=torch.cuda.is_available())
                if info["cuda_available"]:
                    info["gpu"] = torch.cuda.get_device_name()
            except ImportError:
                info["torch"] = "not installed (optional for CPU reference)"
            print(json.dumps(info, indent=2))
            return 1 if args.require_cuda and not info["cuda_available"] else 0
        if args.command == "example":
            _example(args.out)
        elif args.command == "solve":
            if args.output.exists():
                raise FileExistsError(args.output)
            from .inputs import load_input
            from .solver import homogenize
            phase, materials, domain, orientation = load_input(args.input, args.materials)
            result = homogenize(phase, materials, domain, orientation=orientation,
                                backend=args.backend, warmup=args.warmup)
            _save(args.output, result)
            print(json.dumps(dict(output=str(args.output), backend=result["backend"],
                                  max_true_residual=result["max_true_residual"])))
        elif args.command == "check":
            _check(args.kind, args.output)
        elif args.command == "prepare-inputs":
            if min(args.grids) < 2:
                raise ValueError("Grid sizes must be >= 2")
            _prepare(args.cases, args.grids, args.directory)
        elif args.command == "reproduce":
            require_checkout()
            module = _load("voxel_paper_reproduction", ROOT/"repro/scripts/reproduce_paper_tables.py")
            module.regenerate((args.out or ROOT/"repro/generated").resolve())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0

"""Run the measured default pipeline for one six-load woven RVE."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_MODAL = HERE.parent / "modal_hex8_gpu_20261020"
if LOCAL_MODAL.is_dir():
    os.environ.setdefault("FEAD_MODAL", str(LOCAL_MODAL))
import pipeline_study as study
import pipeline_adapter as pipeline


def solve_rve(kind="hg", grid=192, variant="auto", case="strong", warmup=True):
    """Return full unsymmetrized CH, true residuals, timings and provenance.

    The frozen runtime and first modal implementation must be available;
    FEAD_RUNTIME and FEAD_MODAL select their locations. B=1, periodic HEX8.
    """
    if variant == "auto":
        config = json.loads((HERE / "recommended.json").read_text())
        variant = config["variants"][kind]
    b = study.previous.load_runtime(kind)
    b.gpu_guard(); study.modal.extension(); pipeline.extension()
    original_build, original_solver = pipeline.configure(b, study.modal, kind, variant)
    try:
        data, geometry, material, digest = b.inputs(grid, case)
        if warmup:
            b.workload(data, material, "direct", "mixed")
        result, _ = b.workload(data, material, "direct", "mixed")
    finally:
        b.build, b.solve_pcg_batched = original_build, original_solver
    if max(v for c in result["calls"] for v in c["true_residuals"]) > b.TOL:
        raise RuntimeError("Final FP64 true residual failed")
    return dict(kind=kind, grid=grid, variant=variant, case=case,
                warmup=warmup, input_hash=digest, geometry=geometry, material=material,
                tolerance=b.TOL, measurement=result, provenance=study.provenance(b))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--grid",type=int,default=192)
    parser.add_argument("--variant",choices=("auto","paper","modal","lean","energy","analytic","combined"),default="auto")
    parser.add_argument("--case",default="strong")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result=solve_rve(args.kind,args.grid,args.variant,args.case)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(kind=result["kind"],variant=result["variant"],
                         seconds=result["measurement"]["seconds"],
                         CH=result["measurement"]["CH"]),indent=2))


if __name__=="__main__":main()

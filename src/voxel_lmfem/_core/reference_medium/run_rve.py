"""Run one explicitly selected formula experiment in a fresh Python process."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import formula_study as study

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hg", "fi"), default="fi")
    parser.add_argument("--variant", choices=("auto", "baseline", "analytic", "contrast64", "single"), default="auto")
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--case", choices=("strong", "near"), default="strong")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.grid < 2:
        parser.error("--grid must be at least 2")
    variant = args.variant
    if variant == "auto":
        variant = json.loads((HERE / "recommended.json").read_text())["variants"][args.kind]
    valid = {"hg": ("baseline", "contrast64", "single"), "fi": ("baseline", "analytic")}
    if variant not in valid[args.kind]:
        parser.error(f"{args.kind} supports {valid[args.kind]}")
    b = study.previous.load_runtime(args.kind)
    b.gpu_guard()
    study.formula.extension(); study.prior.modal.extension(); study.formula.pipeline.extension()
    study.formula.configure(b, args.kind, variant)
    data, geometry, material, digest = b.inputs(args.grid, args.case)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    measurement, _ = b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    if max(v for c in measurement["calls"] for v in c["true_residuals"]) > b.TOL:
        raise RuntimeError("Final FP64 true residual exceeds tolerance")
    result = dict(kind=args.kind, variant=variant, grid=args.grid, case=args.case,
                  tolerance=b.TOL, input_hash=digest, geometry=geometry, material=material,
                  protocol="fresh process; full six-load warmup then full GPU workflow measurement",
                  measurement=measurement, provenance=study.provenance(b))
    study.save(args.out, result)
    print(json.dumps(dict(kind=args.kind, variant=variant, seconds=measurement["seconds"],
                          CH=measurement["CH"]), indent=2))


if __name__ == "__main__":
    main()

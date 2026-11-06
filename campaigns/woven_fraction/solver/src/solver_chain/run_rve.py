"""Run the validated complete chain, with optional closed-form reference."""
import argparse
import json
from pathlib import Path
import chain_study as cs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("hg", "fi"), required=True)
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--case", choices=("strong", "near"), default="strong")
    p.add_argument("--variant", choices=("auto", *cs.VARIANTS), default="auto")
    p.add_argument("--reference", choices=("auto", "baseline", "closed"), default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.grid < 2:
        raise ValueError("Grid must have at least two nodes on each axis")
    variant = ("compact_combined" if args.kind == "hg" else "combined") if args.variant == "auto" else args.variant
    reference = args.reference
    if reference == "auto":
        reference = "closed" if args.grid == 192 and args.case == "strong" else "baseline"
    b = cs.s.load(args.kind)
    cs.a.native.extension()
    if reference == "closed":
        import closed_study
        closed_study.install(b, args.kind)
    cs.a.configure(b, args.kind, variant)
    data, geometry, material, digest = b.inputs(args.grid, args.case)
    # This entry only supplies the frozen strong/near material definitions.
    # The compact representation is restricted to their tetragonal yarn law.
    if variant.startswith("compact"):
        y = material["yarn"]
        for x, z in (("E22", "E33"), ("Nu12", "Nu13"), ("G12", "G13")):
            if abs(y[x]-y[z]) > 1e-12*max(abs(y[x]), abs(y[z]), 1.):
                raise ValueError("Compact HG requires the verified tetragonal material class")
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    result, _ = b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    maximum = max(v for call in result["calls"] for v in call["true_residuals"])
    if maximum > b.TOL:
        raise RuntimeError("Final true residual exceeds tolerance")
    cs.save(args.out, dict(complete=True, kind=args.kind, grid=args.grid, case=args.case,
        variant=variant, reference_mode=reference, input_hash=digest, geometry=geometry,
        material=material, tolerance=b.TOL, measurement=result,
        reference_design=getattr(b, "_reference_design", None), provenance=cs.provenance(b)))
    print(json.dumps(dict(output=str(args.out), seconds=result["seconds"],
                         max_true_residual=maximum, variant=variant, reference=reference)), flush=True)


if __name__ == "__main__":
    main()

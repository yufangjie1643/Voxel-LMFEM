"""Opt-in complete RVE solve with the measured fixed-grid FFT implementation."""
import argparse
import json
from pathlib import Path
import study as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--grid",type=int,default=192)
    parser.add_argument("--case",choices=("strong","near"),default="strong")
    parser.add_argument("--variant",choices=("auto","baseline","fused","inplace","cached","dense","native_cached","native_dense"),default="auto")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if args.grid<2:parser.error("--grid must be at least 2")
    variant=args.variant
    if variant=="auto":
        variant=json.loads(Path(__file__).with_name("recommended.json").read_text())["variants"][args.kind]
    b=s.load(args.kind)
    if variant!="baseline":
        if variant.startswith("native_"):
            import native_adapter
            native_adapter.extension()
        original=b.build
        def build(data,material,route,precision,stages):
            op=original(data,material,route,precision,stages)
            if variant.startswith("native_"):native_adapter.attach(op,args.kind,variant.removeprefix("native_"))
            else:s.c.attach(op,args.kind,variant)
            stages.end("contest_attachment",op)
            return op
        b.build=build
    data,geometry,material,digest=b.inputs(args.grid,args.case)
    b.workload(data,material,"direct","mixed");b.gpu_guard()
    record,_=b.workload(data,material,"direct","mixed");b.gpu_guard()
    assert max(v for call in record["calls"] for v in call["true_residuals"])<=b.TOL
    s.save(args.out,dict(kind=args.kind,grid=args.grid,case=args.case,variant=variant,tolerance=b.TOL,
        input_hash=digest,geometry=geometry,material=material,measurement=record,provenance=s.provenance(b)))
    print(json.dumps(dict(kind=args.kind,variant=variant,seconds=record["seconds"],CH=record["CH"]),indent=2))


if __name__=="__main__":main()

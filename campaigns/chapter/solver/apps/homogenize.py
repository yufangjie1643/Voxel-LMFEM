"""Six-load exact FI homogenization: generated fixture or generic phase input.

NPZ inputs: phase_id[z,y,x], domain_size[3] (or [1,3]), optional orientation.
JSON material dictionary uses integer phase labels and material_field schema.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--case", choices=["strong", "near"], default="strong")
    p.add_argument("--input", type=Path)
    p.add_argument("--materials", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", action="store_true", help="Full six-load same-size warmup")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if bool(a.input) != bool(a.materials):
        p.error("--input and --materials must be supplied together")
    import chain_study as cs
    import operator_adapter as latest
    b = cs.s.load("fi")
    mf = latest.prior.install(b, cs)
    cs.a.configure(b, "fi", "combined")
    b.TOL = 1e-8
    b.gpu_guard()
    latest.install(b, "split_delta", 64)
    if a.input:
        with np.load(a.input) as z:
            phase = z["phase_id"]
            data = dict(phase_id=phase, grid_shape=phase.shape[-3:], domain_size=z["domain_size"].reshape(1, 3))
            if "orientation" in z:
                data["orientation"] = z["orientation"]
        materials = {int(k): v for k, v in json.loads(a.materials.read_text()).items()}
        identity = dict(input=str(a.input.resolve()), materials=str(a.materials.resolve()))
    else:
        original, geometry, material, sha = b.inputs(a.n, a.case)
        phase, materials, orientation = mf.from_legacy_yarn(original, material)
        data = dict(phase_id=phase, orientation=orientation, grid_shape=original["grid_shape"], domain_size=original["domain_size"])
        identity = dict(case=a.case, n=a.n, input_hash=sha, geometry=geometry)
    if a.warmup:
        b.workload(data, materials, "direct", "mixed")
    start = time.perf_counter()
    result, _ = b.workload(data, materials, "direct", "mixed")
    result["complete_call_seconds"] = time.perf_counter() - start
    result["input"] = identity
    result["storage"] = b._storage
    result["operator"] = b._operator_metadata
    result["scope"] = "FP64 physical FI, FP32 FFT, combined PCG, hybrid background/delta, six loads"
    rr = max(r for c in result["calls"] for r in c["true_residuals"])
    if rr > 1e-8 or not np.isfinite(rr):
        raise RuntimeError(f"True residual {rr}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(a.out), max_true_residual=rr)))


if __name__ == "__main__":
    main()

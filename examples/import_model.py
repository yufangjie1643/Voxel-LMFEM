"""Create, import and solve a small external NPY model. Usage: python examples/import_model.py --out runs/import_demo"""
import argparse
import json
from pathlib import Path

import numpy as np

from voxel_lmfem import homogenize, import_model
from voxel_lmfem.inputs import load_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/import_demo"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / "source"
    source.mkdir()
    # An illustrative homogeneous cell in the external program's x,y,z order.
    np.save(source / "phase.npy", np.full((2, 3, 4), 17, dtype=np.uint8))
    (source / "materials.json").write_text(json.dumps({"17": {"kind": "isotropic", "E": 3500., "nu": .35}}),
                                           encoding="utf-8")
    target = args.out / "model"
    report = import_model(source / "phase.npy", source / "materials.json", target,
                          axes="xyz", spacing=[.5, 2., 3.])
    phase, materials, domain, orientation = load_input(target / "model.npz", target / "materials.json")
    result = homogenize(phase, materials, domain, orientation=orientation, backend="cpu")
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.out.resolve()), shape=report["shape"],
                          domain_size=report["domain_size"], max_true_residual=result["max_true_residual"]), indent=2))


if __name__ == "__main__":
    main()

"""A complete CPU example: python examples/homogeneous.py."""
import numpy as np
from voxel_lmfem import homogenize

result = homogenize(
    np.zeros((4, 4, 4), dtype=np.uint8),
    {0: {"kind": "isotropic", "E": 3500., "nu": .35}},
    [1., 1., 1.],
    backend="cpu",
)
print(np.array(result["CH"]))
print("Maximum true residual:", result["max_true_residual"])

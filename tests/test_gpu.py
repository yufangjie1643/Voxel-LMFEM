"""Opt-in GPU integration: VOXEL_LMFEM_TEST_GPU=1 pytest tests/test_gpu.py."""
import os

import numpy as np
import pytest

from voxel_lmfem import homogenize


@pytest.mark.skipif(os.environ.get("VOXEL_LMFEM_TEST_GPU") != "1",
                    reason="Set VOXEL_LMFEM_TEST_GPU=1 on a configured CUDA machine")
def test_gpu_against_independent_cpu_reference():
    pytest.importorskip("torch")
    z,y,x = np.indices((4,4,4))
    phase = ((x+2*y+z)%3 == 0).astype(np.uint8)
    table = {0: dict(kind="isotropic",E=3500.,nu=.35),
             1: dict(kind="anisotropic",C=np.diag([135000.,9000.,9000.,3500.,5000.,5000.]).tolist(),
                     frame="local")}
    angle = .2*x+.1*y
    rotation = np.zeros(phase.shape+(3,3))
    rotation[...,0,0] = rotation[...,1,1] = np.cos(angle)
    rotation[...,0,1] = -np.sin(angle)
    rotation[...,1,0] = np.sin(angle)
    rotation[...,2,2] = 1.
    reference = homogenize(phase,table,[1.,1.5,2.],orientation=rotation,backend="cpu")
    result = homogenize(phase,table,[1.,1.5,2.],orientation=rotation,backend="cuda",warmup=True)
    error = np.linalg.norm(np.array(result["CH"])-reference["CH"])/np.linalg.norm(reference["CH"])
    assert error < 1e-8
    assert result["max_true_residual"] <= 1e-8

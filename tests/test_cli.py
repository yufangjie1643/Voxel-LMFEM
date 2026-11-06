import json
import subprocess
import sys

import numpy as np


def invoke(*args, cwd):
    return subprocess.run([sys.executable,"-m","voxel_lmfem",*map(str,args)],cwd=cwd,
                          capture_output=True,text=True)


def test_installed_cli_works_outside_checkout_and_preserves_output(tmp_path):
    example = tmp_path/"input"
    output = tmp_path/"result.json"
    assert invoke("example","--out",example,cwd=tmp_path).returncode == 0
    result = invoke("solve","--input",example/"homogeneous.npz",
        "--materials",example/"materials.json","--backend","cpu","--output",output,cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert np.shape(report["CH"]) == (6,6)
    before = output.read_bytes()
    duplicate = invoke("solve","--input",example/"homogeneous.npz",
        "--materials",example/"materials.json","--backend","cpu","--output",output,cwd=tmp_path)
    assert duplicate.returncode != 0
    assert output.read_bytes() == before


def test_help_does_not_require_a_cuda_runtime(tmp_path):
    result = invoke("--help",cwd=tmp_path)
    assert result.returncode == 0
    assert "solve" in result.stdout and "reproduce" in result.stdout

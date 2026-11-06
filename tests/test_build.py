import os
import sys
import threading
from types import SimpleNamespace

import pytest

from voxel_lmfem.build import _stream, extension_progress
from test_cli import invoke


def test_build_help_works_without_cuda(tmp_path):
    result = invoke("build", "--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "execution_fp32" in result.stdout and "--jobs" in result.stdout


def test_progress_preserves_build_arguments_and_reports_long_load(capsys):
    sentinel = object()
    calls = []

    def load(name, sources, extra_cuda_cflags=None, verbose=False):
        calls.append((name, sources, extra_cuda_cflags, verbose))
        threading.Event().wait(.04)
        return sentinel

    extension = SimpleNamespace(load=load)
    with extension_progress(extension, interval=.01) as ready:
        assert extension.load("tiny", ["tiny.cu"], ["-O3"], False) is sentinel
    assert extension.load is load
    assert calls == [("tiny", ["tiny.cu"], ["-O3"], True)]
    assert ready == ["tiny"]
    output = capsys.readouterr().out
    assert "working," in output and "[READY] tiny" in output


def test_failed_compiler_is_reported_and_loader_restored(capsys):
    def load(name, sources, verbose=False):
        raise RuntimeError("nvcc failed")

    extension = SimpleNamespace(load=load)
    with pytest.raises(RuntimeError, match="nvcc failed"):
        with extension_progress(extension):
            extension.load("broken", ["broken.cu"])
    assert extension.load is load
    assert "[FAILED] broken" in capsys.readouterr().out


def test_stream_keeps_compiler_output_and_nonzero_exit(tmp_path, capsys):
    log = tmp_path / "build.log"
    code = _stream([sys.executable, "-u", "-c",
                    "import sys; print('[1/2] nvcc'); print('compile error', file=sys.stderr); sys.exit(7)"],
                   dict(os.environ), log)
    assert code == 7
    output = capsys.readouterr().out
    assert "[1/2] nvcc" in output and "compile error" in output
    assert log.read_text() == output

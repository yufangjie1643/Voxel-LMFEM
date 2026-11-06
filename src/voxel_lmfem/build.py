"""Prepare CUDA extensions with live compiler output and a small solver check."""
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from .paths import configure, require_checkout


@contextmanager
def extension_progress(extension_module, interval=15):
    """Report extension loading while preserving the original build arguments."""
    original = extension_module.load
    signature = inspect.signature(original)
    ready = []

    def load(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        name = bound.arguments["name"]
        bound.arguments["verbose"] = True
        started = time.monotonic()
        stop = threading.Event()
        print(f"[CUDA] Compiling/loading {name}", flush=True)

        def heartbeat():
            while not stop.wait(interval):
                print(f"[CUDA] {name}: working, {time.monotonic()-started:.0f}s elapsed", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            module = original(*bound.args, **bound.kwargs)
        except BaseException:
            print(f"[FAILED] {name} ({time.monotonic()-started:.1f}s)", flush=True)
            raise
        else:
            ready.append(name)
            print(f"[READY] {name} ({time.monotonic()-started:.1f}s)", flush=True)
            return module
        finally:
            stop.set()
            thread.join()

    extension_module.load = load
    try:
        yield ready
    finally:
        extension_module.load = original


@contextmanager
def build_progress():
    """Enable progress before the worker imports its numerical modules."""
    print("[1/3] Checking CUDA and compiler tools", flush=True)
    try:
        import torch
        from torch.utils import cpp_extension
    except ImportError as exc:
        raise RuntimeError('Install GPU dependencies with uv pip install --torch-backend=auto -e ".[gpu]"') from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and an NVIDIA GPU are required")
    if cpp_extension.CUDA_HOME is None:
        raise RuntimeError("Install the CUDA Toolkit and set CUDA_HOME to its installation directory")
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(f"CUDA Toolkit: {cpp_extension.CUDA_HOME}", flush=True)
    print("[2/3] Preparing extensions and running an 8³ woven-cell check", flush=True)
    with extension_progress(cpp_extension) as ready:
        yield
    print(f"[3/3] Ready: {len(set(ready))} CUDA extensions; solver check passed", flush=True)


def _stream(command, environment, log_path):
    """Tee native compiler output and Python output into one UTF-8 log."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise


def run_build(variant="execution_fp32", jobs=None):
    root = require_checkout()
    configure()
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be a positive integer")
    parent = root / "runs/build"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cuda-", dir=parent))
    log = output / "build.log"
    environment = dict(os.environ, VOXEL_LMFEM_BUILD_PROGRESS="1", PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8")
    if jobs is not None:
        environment["MAX_JOBS"] = str(jobs)
    print(f"Preparing {variant}; first compilation can take several minutes.", flush=True)
    print(f"Extension cache: {environment['TORCH_EXTENSIONS_DIR']}", flush=True)
    print(f"Compiler jobs: {environment['MAX_JOBS']}", flush=True)
    print(f"Build log: {log}", flush=True)
    command = [sys.executable, "-u", "-m", "voxel_lmfem", "chapter-study",
               "--phase", "worker", "--case", "woven", "--grid", "8",
               "--variant", variant, "--repeat", "0", "--out", str(output)]
    started = time.monotonic()
    code = _stream(command, environment, log)
    if code:
        raise RuntimeError(f"CUDA preparation failed (exit {code}). Build log: {log}")
    print(f"CUDA preparation completed in {time.monotonic()-started:.1f}s.", flush=True)
    print(f"Build log and check result: {output}", flush=True)

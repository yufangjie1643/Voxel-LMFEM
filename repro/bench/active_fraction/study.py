"""Controlled layered-cell active-fraction sweep; launch with ./run active-study."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from project_paths import ROOT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', type=int, required=True, choices=range(10, 100, 10))
    parser.add_argument('--variant', choices=('hybrid', 'latest'), required=True)
    parser.add_argument('--repeat', type=int, required=True)
    args = parser.parse_args()
    out = Path(os.environ['VOXEL_RUN_DIR'])
    name = f'fraction{args.fraction}_128_{args.variant}_r{args.repeat}'
    target = out/'results'/f'{name}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    old = load('active_base_study', ROOT/'repro/bench/solver_comparison/study.py')
    b, mf = old.configure(args.variant)
    adapt, digest, meta = old.inputs(b, mf, f'fraction{args.fraction}', 128)

    def task():
        start = time.perf_counter()
        data, material = adapt()
        result, _ = b.workload(data, material, 'direct', 'mixed')
        return result, time.perf_counter()-start

    b.gpu_guard()
    task()
    b.gpu_guard()
    print('MEASUREMENT_BEGIN', flush=True)
    result, elapsed = task()
    print('MEASUREMENT_END', flush=True)
    residual = max(v for call in result['calls'] for v in call['true_residuals'])
    assert len(result['calls']) == 6 and old.np.isfinite(residual) and residual <= 1e-8
    assert old.np.isfinite(result['CH']).all()
    data, material = adapt()
    b.cleanup()
    op = b.build(data, material, 'direct', 'mixed', b.Stages())
    micro = old.microbenchmark(op)
    operator = getattr(op, '_operator_metadata', None)
    del op
    gc.collect()
    b.cleanup()
    report = dict(complete=True, case=f'fraction{args.fraction}', grid=128,
        variant=args.variant, repeat=args.repeat, input_hash=digest, input=meta,
        measurement=result, workflow_seconds=elapsed, max_true_residual=residual,
        microbenchmark=micro, storage=getattr(b, '_storage', None), operator=operator,
        reference_design=getattr(b, '_reference_design', None),
        environment=dict(gpu=old.torch.cuda.get_device_name(), torch=old.torch.__version__,
            cuda=old.torch.version.cuda, python=sys.version, threads=old.torch.get_num_threads()),
        sources=old.sources(), protocol='128^3 periodic layered cell; six engineering unit strains; '
        'FP64 physics and FP32 FFT; true residual <=1e-8; complete six-load warmup; '
        'fresh process; identical material, hybrid storage, reference and GPU execution within each pair; '
        'complete-call time includes material construction, operator preparation, load, solve, certification and stress; '
        'resident Ku CUDA-event median of five batches of twenty applications, outside workflow window.')
    b.gpu_guard()
    with target.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('PASS', name, elapsed, micro['Ku']['median_ms'], residual, flush=True)


if __name__ == '__main__':
    main()

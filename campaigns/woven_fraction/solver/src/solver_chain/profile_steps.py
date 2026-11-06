"""Isolated memory evidence for eliminating the unpacked FP64 z temporary."""
import json
import random
import statistics
import torch
import chain_study as cs


def main():
    b = cs.s.load("hg")
    from femlib.batched import pcg
    old = pcg._compiled_pcg_steps()[1]
    new = cs.a.compiled("hg", True, True)[1]
    grid = (192, 192, 192)
    n = 192**3
    r = torch.randn((1, 3*n), device="cuda", dtype=torch.float64)
    p = torch.randn_like(r)
    raw = torch.randn((1, 3, *grid), device="cuda", dtype=torch.float32)
    rho = torch.ones(1, device="cuda", dtype=torch.float64)
    eps = torch.full_like(rho, 1e-30)
    def baseline():
        z = cs.s.c.extension().unpack(raw, *grid, False)
        return old(r, p, z, rho, eps)
    def boundary():
        return new(r, p, raw, rho, eps)
    functions = dict(baseline=baseline, boundary=boundary)
    reference = baseline()
    value = boundary()
    errors = [cs.s.relative(x, y) for x, y in zip(value, reference)]
    assert max(errors) < 1e-10
    del reference, value
    for fn in functions.values():
        for _ in range(3):
            output = fn()
    del output
    rows = {}
    for name, fn in functions.items():
        b.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        b.synchronize()
        rows[name] = dict(incremental_peak_bytes=torch.cuda.max_memory_allocated()-before,
                          samples_ms=[])
        del out
    for i in range(9):
        names = list(functions)
        random.Random(197+i).shuffle(names)
        for name in names:
            b.gpu_guard()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                out = functions[name]()
            end.record()
            end.synchronize()
            rows[name]["samples_ms"].append(begin.elapsed_time(end)/10)
            del out
    for row in rows.values():
        row["median_ms"] = statistics.median(row["samples_ms"])
    cs.save(cs.OUT/"post_profile.json", dict(complete=True, grid=grid, rows=rows,
        relative_errors=errors, scope="IFFT FP32 output to FP64 rho and new p; includes unpack in baseline; excludes preexisting r,p,raw arrays"))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()

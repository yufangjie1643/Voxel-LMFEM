"""Matched full-27 / symmetric-14 / modal / final FI benchmark.

Invoked through the canonical project run command. Existing campaign timing,
six-load warmup, residual checks and memory instrumentation are reused.
"""
from pathlib import Path
import sys


def main():
    from project_paths import ROOT
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT/'experiments/solver_comparison'))
    # Keep this file's stem distinct from the imported study module.
    import importlib.util
    spec = importlib.util.spec_from_file_location('comparison_study', ROOT/'experiments/solver_comparison/study.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    original = comparison.configure
    def configure(variant):
        if variant != 'stencil27':
            return original(variant)
        b, mf = original('paper')
        sys.path.insert(0, str(here))
        spec = importlib.util.spec_from_file_location('stencil27_adapter', here/'adapter.py')
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        source = adapter.install()
        original_build = b.build
        def build(*args, **kwargs):
            op = original_build(*args, **kwargs)
            assert op.S_flat.shape[1] == 27
            op._operator_metadata = dict(representation='full 27-point FI stencil',
                                        assembly='same spatial-slab C21 contraction as 14-point',
                                        assembly_source=source)
            return op
        b.build = build
        return b, mf
    comparison.configure = configure
    comparison.VARIANTS = (*comparison.VARIANTS, 'stencil27')
    comparison.main()


if __name__ == '__main__':
    main()

"""Launch the paper-specific fully integrated HEX8 solver and checks."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from project_paths import ROOT, configure, source_path

COMMANDS = {
    "active-study": ROOT / "experiments/active_fraction/study.py",
    "active-sweep": ROOT / "experiments/active_fraction/launch.py",
    "chapter-study": ROOT / "experiments/chapter_aligned/study.py",
    "fft-precision": ROOT / "experiments/solver_comparison/fft_precision.py",
    "verify-stencils-cpu": ROOT / "experiments/stencil27_baseline/validate_cpu.py",
    "compare-stencils": ROOT / "experiments/stencil27_baseline/launch.py",
    "stencil-study": ROOT / "experiments/stencil27_baseline/study.py",
    "compare": ROOT / "apps/reproduce_comparison.py",
    "fi": ROOT / "apps/homogenize.py",
    "verify-fi": source_path("hybrid_operator/validate.py"),
    "damage-prototype": ROOT / "experimental/progressive_storage/verify.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Pass COMMAND --help for its options.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    configure()
    os.chdir(ROOT)
    for i, value in enumerate(rest):
        if value in ("--out", "--output", "--fields") and i + 1 < len(rest):
            destination = Path(rest[i + 1]).resolve()
            if not destination.is_relative_to(ROOT):
                parser.error("Experiment outputs must stay inside " + str(ROOT))
    target = COMMANDS[args.command]
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *rest]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

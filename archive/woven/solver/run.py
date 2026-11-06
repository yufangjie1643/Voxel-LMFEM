"""Launch the paper-specific fully integrated HEX8 solver and checks."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from project_paths import ROOT, configure, source_path

COMMANDS = {
    "woven-fraction": ROOT / "experiments/woven_fraction/study.py",
    "fft-precision": ROOT / "experiments/solver_comparison/fft_precision.py",
    "verify-fi": source_path("hybrid_operator/validate.py"),
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

"""Audit every versioned file/directory name for Windows-portable checkout paths.

Usage: python .github/tools/check_paths.py [--json REPORT] [--report-only]
Includes new non-ignored files; excludes dependencies, caches and run outputs.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
MAX_PATH = 120
MAX_NAME = 64
MAX_DEPTH = 12
ROOT_ENTRIES = {".gitattributes", ".github", ".gitignore", "CITATION.cff", "LICENSE",
                "README.md", "docs", "examples", "pyproject.toml", "repro", "src", "tests"}


def audit(paths):
    paths = sorted(set(paths))
    directories = sorted({str(parent) for name in paths for parent in PurePosixPath(name).parents
                          if str(parent) != "."})
    issues = [dict(path=name, problem="Place project material in an existing directory")
              for name in sorted({PurePosixPath(p).parts[0] for p in paths} - ROOT_ENTRIES)]
    seen = {}
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.I)
    for path in sorted(set(paths + directories)):
        name = PurePosixPath(path).name
        if any(char in '<>:"\\|?*' or ord(char) < 32 for char in name):
            issues.append(dict(path=path, problem="Windows-invalid character"))
        if name.endswith((".", " ")) or reserved.match(name.split(".")[0]):
            issues.append(dict(path=path, problem="Windows-reserved name or suffix"))
        if len(name.encode("utf-16-le")) // 2 > MAX_NAME:
            issues.append(dict(path=path, problem=f"Component exceeds {MAX_NAME} characters"))
        key = unicodedata.normalize("NFC", path).casefold()
        if key in seen and seen[key] != path:
            issues.append(dict(path=path, problem="Case/Unicode collision", other=seen[key]))
        seen[key] = path
    inventory = []
    for path in paths:
        length = len(path.encode("utf-16-le")) // 2
        depth = len(PurePosixPath(path).parts)
        if length > MAX_PATH:
            issues.append(dict(path=path, problem=f"Relative path exceeds {MAX_PATH} characters"))
        if depth > MAX_DEPTH:
            issues.append(dict(path=path, problem=f"Path exceeds {MAX_DEPTH} components"))
        inventory.append(dict(path=path, length=length, depth=depth))
    return dict(passed=not issues, files=len(paths), directories=len(directories),
                limits=dict(relative_path=MAX_PATH, component=MAX_NAME, depth=MAX_DEPTH),
                max_relative_length=max((row["length"] for row in inventory), default=0),
                max_depth=max((row["depth"] for row in inventory), default=0),
                max_component_length=max((len(PurePosixPath(p).name) for p in paths+directories), default=0),
                issues=issues, inventory=inventory, directory_names=directories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, stdout=subprocess.PIPE)
    paths = [name for name in result.stdout.decode("utf-8").split("\0")
             if name and (ROOT/name).is_file()]
    report = audit(paths)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("inventory", "directory_names")}, indent=2))
    return 0 if report["passed"] or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())

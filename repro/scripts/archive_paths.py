"""Restore unchanged historical analyzer paths in a private temporary directory."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


@contextmanager
def staged_campaigns(repository):
    repository = Path(repository).resolve()
    mapping = json.loads((repository/"repro/path_map.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="vlm-") as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != Path(tempfile.gettempdir()).resolve():
            raise ValueError("Unexpected temporary directory location")
        destinations = set()
        for row in mapping["files"]:
            original = PurePosixPath(row["original_path"])
            if original.is_absolute() or original.parts[0] != "campaigns" or ".." in original.parts:
                raise ValueError(f"Invalid historical path: {original}")
            source = (repository/row["path"]).resolve()
            target = (stage/Path(*original.parts[1:])).resolve()
            if not source.is_relative_to(repository/"archive") or not target.is_relative_to(stage):
                raise ValueError("Archive path escaped its containing directory")
            if target in destinations:
                raise ValueError(f"Duplicate archive destination: {target}")
            destinations.add(target)
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive mapping hash mismatch: {row['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        yield stage

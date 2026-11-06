"""Import labelled voxel volumes into the solver's NPZ/JSON input format."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .inputs import validate


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _orientation(archive):
    keys = set(archive.files)
    if "orientation" in keys:
        if keys & {"axis1", "axis2"}:
            raise ValueError("Use orientation or axis1/axis2, rather than both representations")
        return archive["orientation"]
    if keys & {"axis1", "axis2"}:
        if not {"axis1", "axis2"} <= keys:
            raise ValueError("Both axis1 and axis2 are required")
        return {key: archive[key] for key in ("axis1", "axis2")}
    return None


def _read_volume(path, key):
    domain, orientation = None, None
    if path.suffix.lower() == ".npy":
        phase = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise ValueError(f"NPZ has no {key!r} array; select its label array with --key")
            phase = archive[key]
            domain = archive["domain_size"] if "domain_size" in archive.files else None
            orientation = _orientation(archive)
    else:
        raise ValueError("Supported volume formats: .npy and .npz")
    return phase, domain, orientation


def _reorder_orientation(orientation, source_shape, permutation):
    if orientation is None:
        return None

    def reorder(array, tail):
        value = np.asarray(array, dtype=float)
        if value.shape == tail:
            return value
        if value.shape != tuple(source_shape) + tail:
            raise ValueError(f"Orientation shape must be {tail} or {tuple(source_shape) + tail} in source axis order")
        return value.transpose((*permutation, *range(3, value.ndim)))

    if isinstance(orientation, dict):
        return {name: reorder(orientation[name], (3,)) for name in ("axis1", "axis2")}
    return reorder(orientation, (3, 3))


def import_model(input_path, materials_path, out, *, axes="zyx", spacing=None,
                 domain_size=None, key="phase_id", orientation_path=None):
    """Convert an external labelled volume and validate its material definitions.

    axes names the source array's three dimensions. Physical lengths, spacings
    and vector/tensor components always use the global x,y,z coordinate system.
    The output is a new directory containing model.npz, materials.json and
    import.json. Geometry is preserved exactly, with axis permutation only.
    """
    source, materials_path, output = Path(input_path), Path(materials_path), Path(out).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not isinstance(axes, str) or len(axes) != 3 or set(axes) != set("xyz"):
        raise ValueError("axes must be a permutation of xyz, such as zyx or xyz")
    if spacing is not None and domain_size is not None:
        raise ValueError("Specify spacing or domain_size, rather than both")
    phase, stored_domain, orientation = _read_volume(source, key)
    if phase.ndim != 3 or phase.dtype.kind not in "biu" or min(phase.shape) < 2:
        raise ValueError("Input must be a 3D integer label array with every dimension >= 2; segment image intensities first")
    if phase.dtype.kind == "b":
        phase = phase.astype(np.uint8)
    if int(phase.max()) > np.iinfo(np.int64).max:
        raise ValueError("Material IDs must fit a signed 64-bit integer")
    if orientation_path is not None:
        if orientation is not None:
            raise ValueError("Orientation is already present in the source NPZ")
        orientation_path = Path(orientation_path)
        if orientation_path.suffix.lower() == ".npy":
            orientation = np.load(orientation_path, allow_pickle=False)
        elif orientation_path.suffix.lower() == ".npz":
            with np.load(orientation_path, allow_pickle=False) as archive:
                orientation = _orientation(archive)
            if orientation is None:
                raise ValueError("Orientation NPZ needs orientation or axis1/axis2 arrays")
        else:
            raise ValueError("Orientation input must be .npy or .npz")
    source_shape = phase.shape
    permutation = tuple(axes.index(axis) for axis in "zyx")
    phase = phase.transpose(permutation).astype(phase.dtype.newbyteorder("="), copy=False)
    orientation = _reorder_orientation(orientation, source_shape, permutation)
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing must be three finite positive values in x,y,z order")
        domain_size = spacing * np.array(phase.shape[::-1])
    elif domain_size is None:
        domain_size = stored_domain
    if domain_size is None:
        raise ValueError("Provide --spacing or --domain-size, or domain_size in the source NPZ")
    materials = json.loads(materials_path.read_text(encoding="utf-8-sig"))
    if not isinstance(materials, dict) or not materials or any(not isinstance(v, dict) for v in materials.values()):
        raise ValueError("Materials JSON must map phase IDs to material property objects")
    required = {"isotropic": {"E", "nu"}, "orthotropic": {"E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23"},
                "anisotropic": {"C"}}
    for label, spec in materials.items():
        if spec.get("kind") not in required:
            raise ValueError(f"Material {label}: kind must be isotropic, orthotropic or anisotropic")
        missing = required[spec["kind"]] - spec.keys()
        if missing:
            raise ValueError(f"Material {label}: missing {', '.join(sorted(missing))}")
        if not np.iinfo(np.int64).min <= int(label) <= np.iinfo(np.int64).max:
            raise ValueError(f"Material ID {label} must fit a signed 64-bit integer")
    try:
        phase, table, domain, orientation, *_ = validate(phase, materials, domain_size, orientation)
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError(f"Invalid material definition: {exc}") from exc
    arrays = dict(phase_id=phase, domain_size=domain)
    if isinstance(orientation, dict):
        arrays.update(orientation)
    elif orientation is not None:
        arrays["orientation"] = orientation
    labels, counts = np.unique(phase, return_counts=True)
    sources = dict(volume=source, materials=materials_path)
    if orientation_path is not None:
        sources["orientation"] = orientation_path
    report = dict(schema_version=1, input_axes=axes, input_shape=list(source_shape),
                  axes="zyx", shape=list(phase.shape), domain_size=domain.tolist(),
                  spacing=(domain / np.array(phase.shape[::-1])).tolist(),
                  phases=[dict(id=int(label), voxels=int(count), fraction=float(count / phase.size))
                          for label, count in zip(labels, counts)],
                  sources={name: dict(file=str(path.resolve()), sha256=_digest(path)) for name, path in sources.items()})
    material_text = json.dumps(table, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".voxel-import-", dir=output.parent) as staging:
        stage = Path(staging)
        np.savez_compressed(stage / "model.npz", **arrays)
        (stage / "materials.json").write_text(material_text, encoding="utf-8")
        (stage / "import.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Choose a new output directory: {output}")
        stage.rename(output)
    return report

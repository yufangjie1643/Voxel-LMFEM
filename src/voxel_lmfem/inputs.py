"""Validate host-side phase/material inputs before invoking either solver."""
import json
from pathlib import Path

import numpy as np

from .paths import configure


def material_module():
    configure()
    from adapter import load_material_module
    return load_material_module()


def validate(phase_id, materials, domain_size, orientation=None):
    phase = np.asarray(phase_id)
    if phase.ndim != 3 or phase.dtype.kind not in "iu" or min(phase.shape) < 2:
        raise ValueError("phase_id must be an integer (Nz, Ny, Nx) array with every dimension >= 2")
    domain = np.asarray(domain_size, dtype=float)
    if domain.shape not in ((3,), (1, 3)) or not np.isfinite(domain).all() or (domain <= 0).any():
        raise ValueError("domain_size must contain three finite positive lengths in (x, y, z) order")
    domain = domain.reshape(3)
    if not isinstance(materials, dict) or not materials:
        raise ValueError("materials must be a nonempty dictionary")
    table = {}
    for key, value in materials.items():
        if isinstance(key, bool) or not isinstance(key, (str, int, np.integer)):
            raise ValueError("Material labels must be integers or integer JSON keys")
        try:
            label = int(key)
        except ValueError as exc:
            raise ValueError(f"Invalid material label: {key!r}") from exc
        if label in table:
            raise ValueError(f"Duplicate material label after conversion: {label}")
        table[label] = value
    mf = material_module()
    labels, tensors, local = mf.phase_table(table)
    used = np.unique(phase)
    if not np.isin(used, labels).all():
        raise ValueError(f"Undefined material labels: {used[~np.isin(used, labels)].tolist()}")
    for index in np.flatnonzero(local):
        selected = np.flatnonzero(phase.ravel() == labels[index])
        if selected.size and orientation is None:
            raise ValueError("Local-frame anisotropic phases require orientation")
        for start in range(0, len(selected), 65536):
            mf._orientation_values(orientation, phase.shape, selected[start:start+65536])
    return phase, table, domain, orientation, labels, tensors, local


def load_input(input_path, materials_path):
    with np.load(input_path, allow_pickle=False) as archive:
        if not {"phase_id", "domain_size"} <= set(archive.files):
            raise ValueError("Input NPZ must contain phase_id and domain_size")
        phase, domain = archive["phase_id"], archive["domain_size"]
        orientation = None
        if "orientation" in archive:
            orientation = archive["orientation"]
        elif "axis1" in archive or "axis2" in archive:
            if not {"axis1", "axis2"} <= set(archive.files):
                raise ValueError("Both axis1 and axis2 are required")
            orientation = dict(axis1=archive["axis1"], axis2=archive["axis2"])
    materials = json.loads(Path(materials_path).read_text(encoding="utf-8"))
    return phase, materials, domain, orientation

"""Compatibility import for the maintained experiment drivers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from voxel_lmfem.paths import ROOT, configure, source_path

__all__ = ["ROOT", "configure", "source_path"]

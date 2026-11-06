# Benchmark data

Run python run.py prepare-inputs to populate data/cases/ with the exact 192³ sphere and Grosmont fields. Array and NPY-file hashes are checked against metadata.

Working inputs are ignored by Git. Frozen originals remain in:

- archive/paper/solver/remote/inputs/
- archive/chapter/solver/data/grosmont/

Small spheres can be generated with python run.py prepare-inputs --cases sphere --grids 8 32. Bundled Grosmont arrays support 192³ and 400³; the original full-volume digital-rock image is not included. See NOTICE.md for attribution and docs/usage.md for user-supplied NPZ models.

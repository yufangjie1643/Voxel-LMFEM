# Development

Create the environment following [Installation](installation.md), then install the development tools:

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
uv run --no-sync python -m ruff check src tests examples
uv build
```

Build outputs are written to `dist/`. The wheel contains the Python API and numerical implementation; the full repository also contains the paper's experiments and data.

## Working directories

- `src/`: solver implementation and public interfaces.
- `tests/`: numerical and interface tests.
- `examples/`: small examples and material configurations.
- `repro/bench/`: benchmark protocols and measurement drivers.
- `runs/`: local computation results.

Use analytical solutions, mechanics invariants and independent numerical references to verify solver changes. The [validation guide](validation.md) describes the available checks.

## Names and paths

Use descriptive file names and group files by purpose. Repository-relative paths are limited to 120 characters, individual names to 64 characters, and depth to 12 components. Run the project checks with:

```bash
uv run --no-sync python .github/tools/check_paths.py
```

CI runs path checks, numerical tests, input preparation, table generation and package installation on Windows and Linux.

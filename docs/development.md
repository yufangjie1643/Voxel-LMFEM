# Development

Install with python -m pip install -e ".[dev]". New outputs belong under runs/; working benchmark inputs belong under data/cases/.

Public API changes need tests against physical or analytical invariants. GPU execution changes need the opt-in GPU comparison as well as CPU checks. Never edit frozen campaigns to make a current test pass. A changed method or protocol needs a new measured campaign and its source-to-claim map.

The maintained layer changes the original organization as follows:

1. Consolidate identical numerical modules in src/voxel_lmfem/_core/ without changing their bytes.
2. Replace machine-specific source discovery with installed-package paths.
3. Dispatch both main and yarn-fraction experiments through one root entry.
4. Read isotropic inputs from data/cases/, populated by hash-verified preparation.
5. Expose the existing independent assembly through a small CPU adapter, with the exact homogeneous zero-fluctuation solution.
6. Keep public input validation and CLI errors separate from measured numerical kernels.

Build distributions with:

    python -m build

The wheel contains the public API and numerical core. The source distribution also contains maintained examples, tests, protocols and documentation. The full GitHub release ZIP additionally contains the measurement archive.

v0.1.0 is the first maintained-source release. Earlier fead-2026-10-25-v1 and v2 tags identify review packages. Use forward commits and new tags; preserve published snapshots.

The one-time deploy_entrypoints helper and old absolute-path validation plan stay only in campaigns/. The maintained source snapshot helper discovers the current checkout, and the chapter guide points to the current table regeneration entry.

# Repository paths

The repository uses ordinary Git cloning on Windows, Linux and macOS. Published file and directory names are audited with:

    uv run --no-sync python scripts/check_paths.py

For an inventory of every file and directory, including lengths and any failures:

    uv run --no-sync python scripts/check_paths.py --json runs/path_audit.json

The checker includes versioned files and new non-ignored files. Environments, build products, caches and local run outputs are outside the source inventory.

## Naming limits

- Maximum repository-relative path: 120 characters.
- Maximum file or directory name: 64 characters.
- Maximum depth, including the filename: 12 components.
- No Windows-reserved names, forbidden characters, trailing periods/spaces, or case/Unicode collisions.

These budgets leave room for the checkout's parent directory. CI uses standard Windows path handling and checks the inventory before installation. Keep future experiment dates and run parameters in metadata when they would create repeated directory layers.

## Historical archive migration

The layout was audited against the MIT-license commit abe795b on 6 November 2026. The original tree contained 2,973 versioned files in 331 directories. Its longest relative path was 189 characters, maximum depth was 15, and 808 paths exceeded the new 120-character budget. At the author's 92-character checkout root, 67 complete file paths exceeded 259 characters.

The shortened layout has a maximum relative path of 117 characters, maximum depth of 12, and maximum individual name length of 47. All paths fit the budgets. A 92-character checkout root therefore yields a maximum complete file path of 210 characters.

| Directory | Purpose |
|---|---|
| archive/paper | Comparison, convergence, scaling and active-fraction evidence |
| archive/woven | Yarn-fraction evidence, analyzers and original solver snapshot |
| archive/chapter | Original chapter solver and material/input fixtures |
| archive/sources | Measured dependency snapshots with concise component names |
| src/voxel_lmfem | Maintained solver package |
| experiments | Maintained experiment protocols |

All 2,516 relocated archive files retain their original SHA-256 values. The 238 canonical numerical core files are unchanged. repro/path_map.json maps every original path to its current path and checksum. Original source-column identifiers remain in the numerical outputs for traceability.

The reproduction entry reconstructs the original analyzer layout in a temporary directory, runs the unchanged historical analyzers, and removes that temporary copy. The checked-in tree stays shallow. The prior releases and their directory layouts remain available for historical reference.

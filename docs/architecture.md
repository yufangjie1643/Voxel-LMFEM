# Repository architecture

    src/voxel_lmfem/    Installed package, input validation, API and CLI
      _core/           Canonical FI operators, materials, FFT and PCG modules
    experiments/       Maintained benchmark protocols and measurement drivers
    configs/           Material examples and material-state fixture
    examples/          Small runnable examples
    tests/             Public API, numerical-reference and CLI regression tests
    validation/        Independent material-state validation
    data/              Data guide; prepared inputs go in data/cases/
    baselines/         Third-party reference source and existing licenses
    docs/              Installation, usage and development documentation
    repro/             Paper map, checksums and current generated tables
    campaigns/         Frozen measured campaigns retained for provenance
    runs/              New local outputs, ignored by Git

## Canonical source

The two archived experiments share 238 identical core files, consolidated under src/voxel_lmfem/_core/. The yarn-fraction driver uses the same core. Unrelated extra comparison modules in the later snapshot are not part of the maintained solver.

The public solver adapter, input checks and source discovery live in voxel_lmfem.solver, inputs and paths respectively.

| Core component | Responsibility |
|---|---|
| materials | Constitutive definitions, orientation, shared/local storage |
| hex8_modal, constitutive_basis | Polynomial representation and element action |
| hybrid_operator | Homogeneous background and active contrast action |
| spectral, reference_medium, modal_reference | FFT precision and reference construction |
| solver_chain, assembly_pipeline, compact_operator | PCG execution and operator adapters |
| runtime | Required numerical implementation dependencies |

Internal filenames preserve the tested dependency chain. Applications should use the public API and root command line.

## Provenance

repro/provenance/canonical_sources.json maps maintained files to their frozen origins and original hashes. Numerical core files are copied byte-for-byte. Infrastructure changes are described in docs/development.md.

The public solver does not import campaigns/. The reproduction command reads that archive explicitly to regenerate measured observations. Existing v1/v2 releases remain available.

The broad v2 inventory is retained as repro/provenance/v2_file_manifest.json. The active evidence inventory covers frozen campaign files and the paper regeneration script; maintained code and documentation are identified by Git commits and release archives.

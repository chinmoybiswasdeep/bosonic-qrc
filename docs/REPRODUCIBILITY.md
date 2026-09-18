# Reproducibility

Outputs contain schema version, resolved config and hash, Git commit, UTC times,
platform/dependencies, seeds, raw CSV, append-only journal, checksums, figures,
and JSON/Markdown gates.

```bash
bosonic-qrc-dv-static-capacity configs/smoke/static_ipc.yaml --output results/calibration/static_ipc_smoke
bosonic-qrc-dv-static-capacity configs/calibration/static_ipc.yaml --output results/calibration/static_ipc
bosonic-qrc-dv-static-capacity configs/production/static_ipc.yaml --output results/production/static_ipc
```

Commands resume exact completed tuples and reject changed configs. Parallelize
with `--chunk-index I --chunk-count N`, one output directory per shard. Merge:

```bash
bosonic-qrc-dv-aggregate configs/production/static_ipc.yaml --output results/production/static_ipc results/shards/*
bosonic-qrc-dv-static-capacity configs/production/static_ipc.yaml --output results/production/static_ipc
```

The final command produces aggregate tables/figures without rerunning merged
backend work.

# EOC workflow

Install with `pip install -e '.[dev]'`, run `pytest`, then execute the smoke command in
`VALIDATION.md`. Calibration and publication configs are deliberately not run in CI. Independent
reservoir, data, measurement, and trajectory seeds are stored in YAML and copied verbatim into the
manifest. The manifest records dependency versions, platform, UTC times, base/current commits,
config SHA-256, and artifact SHA-256 values.

Long scans can be split deterministically with `--chunk-index I --chunk-count K`; each output
directory records its chunk identity. `--resume` returns only after matching the config/chunk and
verifying every artifact checksum. Use distinct output directories per chunk and merge only
manifests with the same config hash and disjoint interaction points.

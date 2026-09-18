# Reproducibility

Each output includes schema version, resolved config, config hash, Git commit,
UTC timestamps, platform and dependency versions, seeds, raw CSV, an append-only
JSONL journal, checksums, figures, and JSON/Markdown gates.

## Commands

```bash
bosonic-qrc-cv-ipc configs/smoke/ipc.yaml --output results/calibration/ipc_smoke
bosonic-qrc-cv-ipc configs/calibration/ipc.yaml --output results/calibration/ipc
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/production/ipc
```

Repeating the same command resumes completed seed tuples. A changed config in
the same directory fails with a hash mismatch.

Local parallel example:

```bash
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/shards/0 --chunk-index 0 --chunk-count 4 &
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/shards/1 --chunk-index 1 --chunk-count 4 &
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/shards/2 --chunk-index 2 --chunk-count 4 &
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/shards/3 --chunk-index 3 --chunk-count 4 &
wait
bosonic-qrc-cv-aggregate configs/production/ipc.yaml --output results/production/ipc results/shards/{0,1,2,3}
bosonic-qrc-cv-ipc configs/production/ipc.yaml --output results/production/ipc
```

The final IPC invocation regenerates aggregate tables, figures, and gates from
the merged journal without repeating completed backend runs.

# EOC workflow

Install with `pip install -e '.[dev]'`, run tests, then use the smoke command in `VALIDATION.md`.
YAML preserves independent reservoir/data/measurement/trajectory seeds and predeclared thresholds.
Each manifest includes exact base/current Git SHAs, resolved config plus SHA-256, dependency
versions, platform, UTC times, and checksums for CSV/JSON/PNG/PDF/SVG artifacts.

Use `--chunk-index I --chunk-count K` with distinct output directories to partition open-system
interaction points. `--resume` accepts an existing run only if config/chunk identity and every file
checksum match. Merge only equal-config, disjoint chunks.

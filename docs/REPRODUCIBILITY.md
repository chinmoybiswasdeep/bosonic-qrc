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


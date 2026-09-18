# Reproducibility

Install `pip install -e '.[dev,experiments]'`, run tests, then the smoke command in `VALIDATION.md`.
The YAML files preserve independent reservoir/data/measurement/trajectory seeds and predeclared
thresholds. Manifests record base/current Git SHAs, resolved config and hash, dependency versions,
platform, UTC times, and artifact checksums.

Split long open scans with `--chunk-index I --chunk-count K`, using a distinct output directory per
chunk. `--resume` checks config/chunk identity and all file hashes before reusing a result. Only
equal-config, disjoint chunks may be aggregated.


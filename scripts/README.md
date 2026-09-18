# Launch scripts

Serial, resume, and local-parallel commands are in
`docs/REPRODUCIBILITY.md`. Submit the array with
`sbatch scripts/production.slurm`. After all shards finish, merge them with
`bosonic-qrc-cv-aggregate`, rerun `bosonic-qrc-cv-ipc` on the merged output
to create aggregates, and run:

```bash
bosonic-qrc-cv-publication results/calibration/ipc results/production/ipc --output results/publication
```

Inspect `results/publication/gate_report.json` before making claims.


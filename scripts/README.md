# Launch scripts

Serial and resume commands are in `docs/REPRODUCIBILITY.md`. Submit with
`sbatch scripts/production.slurm`, merge with
`bosonic-qrc-dv-aggregate`, rerun the capacity command to build aggregate
artifacts, then generate a bundle:

```bash
bosonic-qrc-dv-publication results/calibration/static_ipc results/production/static_ipc --output results/publication
```

Inspect the machine-readable gate before making claims.

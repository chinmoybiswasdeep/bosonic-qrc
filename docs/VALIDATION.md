# Validation

Unit tests cover basis indexing/dimension, Hermiticity, number conservation, exact matrix versus
`expm_multiply`, Floquet unitarity, second-order refinement, Kerr phases, symmetry blocks, partial
trace, Kraus completeness, density-matrix physicality, recurrence, causal input dependence, reset,
seeded trajectories, leakage-safe chronological splits, and deterministic tasks.

The smoke workflow additionally requires candidate physical crossings, SFF and OTOC records,
monotonic Trotter refinement, physical open states, and predeclared boundary-shell/retained-norm
guards. A pass means pipeline validation only. It cannot upgrade the claims ledger.

Run:

```bash
python -m pytest
bosonic-qrc-eoc-dv configs/eoc/smoke.yaml --output results/eoc/smoke
bosonic-qrc-eoc-dv configs/eoc/smoke.yaml --output results/eoc/smoke --resume
```


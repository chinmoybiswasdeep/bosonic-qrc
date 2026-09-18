# Validation

Tests cover Fock indexing, Hamiltonian structure, exact/sparse agreement, exact Kerr phases,
unitarity, Floquet reduction, Strang refinement, reflection sectors, partial trace, state
physicality, every input-resource family, recurrence/causality/reset, Piquasso native-Kerr phase
agreement, cutoff-aligned state distances, backend separation, deterministic tasks, and no split
leakage.

The smoke gate additionally requires independently generated boundary candidates, stored SFF and
OTOC data, monotonic step refinement, physical open states, and predeclared cutoff-shell/retained
norm guards. `PASS_SMOKE` validates the pipeline only.

```bash
python -m pytest
bosonic-qrc-eoc-cv configs/eoc/smoke.yaml --output results/eoc/smoke
bosonic-qrc-eoc-cv configs/eoc/smoke.yaml --output results/eoc/smoke --resume
```


# Validation

Tests cover basis construction, Hermiticity, number conservation, exact/sparse agreement, Kerr
phases, unitarity, Floquet and Strang limits, reflection sectors, partial trace, state physicality,
Born-weighted unconditional evolution, conditional recurrence/causality, resource-injection
ablation, deterministic task generation, and split leakage. The corrected legacy vacuum test now
checks the documented nonredundant feature order: means, variances, then second moments.

The smoke gate also requires boundary candidates, SFF plus OTOC records, monotonic step refinement,
physical open states, and predeclared cutoff guards. A pass is pipeline validation, not an EOC claim.

```bash
python -m pytest
bosonic-qrc-eoc-cvmb experiments/eoc/configs/smoke.yaml --output experiments/eoc/results/smoke
bosonic-qrc-eoc-cvmb experiments/eoc/configs/smoke.yaml --output experiments/eoc/results/smoke --resume
```


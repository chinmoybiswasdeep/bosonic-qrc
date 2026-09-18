# bosonic-qrc

Research implementations of photonic quantum reservoir processing. The code is
kept on two independent branches; `main` is an index and does not contain either
incompatible Python package.

| Branch | Physical model | Capacity currently implemented | Status |
| --- | --- | --- | --- |
| [`CV`](https://github.com/chinmoybiswasdeep/bosonic-qrc/tree/CV) | Persistent Gaussian loop with fresh squeezed pulses, Piquasso | Temporal Legendre IPC | `PASS_CALIBRATION`; production/task matrix pending |
| [`DV`](https://github.com/chinmoybiswasdeep/bosonic-qrc/tree/DV) | Static Fock-input interferometric QRP, Perceval SLOS | Delay-zero nonlinear capacity | `PASS_CALIBRATION`; robustness matrix pending |

CV is temporal QRC because a reduced Gaussian loop state persists between input
steps. DV is currently static QRP/QELM: its features are sample-dependent but it
has no persistent temporal state. The DV branch therefore refuses delayed IPC.

## Installation

```bash
git switch CV
python -m pip install -e ".[dev]"
pytest
bosonic-qrc-cv-ipc configs/smoke/ipc.yaml --output results/calibration/ipc_smoke
```

```bash
git switch DV
python -m pip install -e ".[dev]"
pytest
bosonic-qrc-dv-static-capacity configs/smoke/static_ipc.yaml --output results/calibration/static_ipc_smoke
```

## Latest calibration status

CV calibration covers four reservoir/data runs and the complete 20-target bank
with 399 independently generated nulls per target. Mean significant temporal
capacity is 2.735 (hierarchical-bootstrap 95% CI 2.702–2.771) at numerical rank
3. DV calibration covers eight model/seed runs and degrees 1–6 with 119 nulls;
the reservoir mean is 5.99997 while the matched identity control is zero.

Both branch-level calibration gates pass. The generated publication gates are
still `INCOMPLETE`: production profiles were deliberately not run, paired
task/control/robustness matrices remain incomplete, and exact truncated-Fock
tomography is explicitly unsupported. These are calibration results, not claims
of quantum advantage or publication readiness.

Reference architectures:

- Garcia-Beni et al., *Scalable photonic platform for real-time quantum reservoir computing*, [arXiv:2207.14031](https://arxiv.org/abs/2207.14031).
- Lopez Carreno et al., *Quantum and classical processing with photonic quantum machine learning*, [arXiv:2605.10471](https://arxiv.org/abs/2605.10471).

Each research branch contains versioned schemas, exact configs, append-only
resume journals, raw per-target tables, checksummed manifests, PDF/SVG/PNG
figures, production/SLURM instructions, publication bundles, and CI workflows.

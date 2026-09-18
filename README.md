# bosonic-qrc - CV branch

Backend-native continuous-variable feedback-loop reservoir based on
Garcia-Beni et al. (arXiv:2207.14031). Every quantum preparation, gate, channel,
reduction, and measurement is executed by `piquasso.GaussianSimulator`; only a
classical ridge readout is trained.

At each step, N persistent Gaussian modes interact in parallel with N fresh
squeezed modes. Distinct fixed passive and two-mode-active networks act on the
retained loop and detector arms. Readout features are the upper triangle of the
detector-arm x-quadrature covariance, with dimension N(N+1)/2. Finite-shot mode
uses actual backend homodyne samples.

## Install and verify

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\ruff check src tests
.\.venv\Scripts\pytest
.\.venv\Scripts\bosonic-qrc-cv configs/smoke/memory.yaml --output results/calibration/smoke
.\.venv\Scripts\bosonic-qrc-cv-ipc configs/smoke/ipc.yaml --output results/calibration/ipc_smoke
```

Inference-resolved IPC calibration:

```powershell
.\.venv\Scripts\bosonic-qrc-cv-ipc configs/calibration/ipc.yaml --output results/calibration/ipc
```

The production profile is a cost declaration and is never launched by CI. Gaussian
steps scale approximately cubically with the joint 2N-mode covariance dimension.
This branch passes physical topology and smoke gates but does not yet implement
the complete IPC/task/ablation matrix requested for a manuscript. See
`docs/LIMITATIONS.md` and `docs/NOVELTY.md`.

Capacity is held-out `C=max(0,R²)`; raw `R²` and squared correlation are retained
as diagnostics. Reservoir, measurement, and data seeds are independent. The
smoke profile uses `B=20` only as a pipeline check. The committed calibration
uses the complete 20-target family and `B=399`, the minimum needed for its BH
resolution condition. It passes `PASS_CALIBRATION`, not publication readiness.
See `docs/REPRODUCIBILITY.md` for resume, chunk, merge, production, and reporting
commands.
## Corrected many-body EOC workflow

The fixed-number Floquet calibration and complete four-mode non-Gaussian Fock-channel study
are documented in [docs/EOC_METHODS_V2.md](docs/EOC_METHODS_V2.md). Run the CI profile with
`bosonic-qrc-eoc-cv run configs/eoc/smoke.yaml --output results/eoc/corrected-smoke-v2`.
Smoke validates implementation only and does not establish an EOC advantage.

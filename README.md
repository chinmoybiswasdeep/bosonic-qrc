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

Moderate calibration:

```powershell
.\.venv\Scripts\bosonic-qrc-cv configs/calibration/memory.yaml --output results/calibration/memory
```

The full profile is a cost declaration and is never launched by CI. Gaussian
steps scale approximately cubically with the joint 2N-mode covariance dimension.
This branch passes physical topology and smoke gates but does not yet implement
the complete IPC/task/ablation matrix requested for a manuscript. See
`docs/LIMITATIONS.md` and `docs/NOVELTY.md`.

Capacity output reports held-out squared correlation and raw held-out `test_r2`
separately. Optical, measurement, and data seeds are independent. The committed
IPC smoke uses only 20 null surrogates, whose minimum attainable p-value is
1/21; it verifies execution and the rank bound but is not a significant-capacity
claim. Calibration requires at least 100 surrogates and longer target banks.

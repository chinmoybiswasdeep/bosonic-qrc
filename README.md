# bosonic-qrc — CV branch

This branch contains a deliberately backend-strict, continuous-variable proof of
the feedback-loop reservoir in García-Beni *et al.* (arXiv:2207.14031).  Every
quantum state preparation and evolution runs via `piquasso.GaussianSimulator`.
The only trained component is a scikit-learn ridge readout.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\pytest
.\.venv\Scripts\bosonic-qrc-cv configs/smoke/memory.yaml --output results/cv-smoke
```

The reservoir retains reduced Gaussian memory modes between steps. Fresh encoded
ancilla pulses are coupled by a beam splitter, followed by fixed passive mixing,
fixed single-mode squeezing, and optional loss. Features are the upper triangle
of the backend-reported x-quadrature covariance. This initial implementation
provides a delayed-memory calibration only; finite-shot homodyne, the requested
full benchmark suite, and the DV branch remain future work.

See `docs/LIMITATIONS.md` for scientific boundaries.

Calibration (moderate laptop profile):

```powershell
.\.venv\Scripts\bosonic-qrc-cv configs/calibration/memory.yaml --output results/cv-calibration
```

Scaling is dominated by Gaussian covariance operations and is approximately
cubic in the total number of loop-plus-input modes per timestep. A `full`
profile is intentionally not shipped: it requires the unimplemented benchmark,
finite-shot, and multi-seed orchestration rather than pretending this smoke
baseline is publication-scale.

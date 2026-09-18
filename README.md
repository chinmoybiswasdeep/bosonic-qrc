# cv-mb-qrc

Stateful continuous-variable and qubit measurement-based quantum reservoir
computing, with leakage-controlled temporal benchmarks and reproducible artifacts.

This repository owns the reservoir orchestration, encoders, feature extraction,
ridge readout, diagnostics, benchmark datasets, experiments, tests and reports.
[PhotoGraphiQ](https://github.com/chinmoybiswasdeep/PhotoGraphiQ) remains the CV
execution and physics dependency. Graphix supplies the optional qubit backend;
MentPy is an optional independent corrected-wire reference.

The two temporal models are explicit:

- `CVMBReservoir` and `GraphixMBReservoir` carry the complete surviving quantum
  state between steps.
- `WindowedMBQELM` resets the quantum resource and receives an explicit classical
  input window. It is a feature-map baseline, not a recurrent reservoir.

The default experiments train only a regularized classical readout. Reservoir
parameters stay fixed after seeded initialization.

## Install

Install the audited PhotoGraphiQ checkout, then this repository:

```sh
python -m pip install -e "../softwares/PhotoGraphiQ[dev]"
python -m pip install -e ".[dev,docs,experiments,graphix,validation]"
```

Core CV use does not import Graphix or MentPy. Their adapters raise an informative
error only when selected.

## Quick start

```python
import numpy as np
from cv_mb_qrc.reservoirs import CVConfig, CVMBReservoir, RidgeReadout

inputs = np.random.default_rng(1729).uniform(-1, 1, 200)
reservoir = CVMBReservoir(CVConfig(seed=7, memory_modes=2, tier="B"))
features = reservoir.run_sequence(inputs, washout=20)

readout = RidgeReadout(1e-4).fit(
    inputs[20:], features.features, np.roll(inputs, 1)[20:]
)
prediction = readout.predict(inputs[20:], features.features)
```

Angles are radians. CV conventions are `[q,p]=2i`, interleaved quadratures,
vacuum statistical covariance `I`, and positive resource squeezing means momentum
squeezing.

## Reproduce

```sh
python -m pytest --cov=src/cv_mb_qrc --cov-fail-under=90
python -m ruff check src tests experiments
python -m ruff format --check src tests experiments
python -m mypy
python -m build
python experiments/measurement_based_reservoir/main.py
python experiments/measurement_based_reservoir/reproduce.py
```

The committed smoke-study artifacts contain 420 raw method/seed/task runs and 17
figures in SVG, PDF and PNG. The deterministic score records reproduced exactly.
These small results do not establish quantum advantage, a memory–nonlinearity
tradeoff violation, non-Gaussian task superiority or edge-of-chaos behavior.

Read the [design audit](docs/research/measurement_based_quantum_reservoir_design.md),
[tutorial](docs/tutorials/measurement-based-reservoir.md), [experiment protocol](experiments/measurement_based_reservoir/README.md),
and [implementation report](docs/research/measurement_based_quantum_reservoir_report.md).
## Corrected many-body EOC workflow

The fixed-number Floquet calibration and four-mode manual mixed-state adapter are documented
in [docs/EOC_METHODS_V2.md](docs/EOC_METHODS_V2.md). Run the CI profile with
`bosonic-qrc-eoc-cvmb run experiments/eoc/configs/smoke.yaml --output experiments/eoc/results/corrected-smoke-v2`.
Smoke validates implementation only and does not establish an EOC advantage.

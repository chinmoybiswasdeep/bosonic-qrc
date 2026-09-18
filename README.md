# bosonic-qrc

Research implementations of photonic quantum reservoir processing. The code is
kept on two independent branches; `main` is an index and does not contain either
incompatible Python package.

| Branch | Physical model | Capacity currently implemented | Status |
| --- | --- | --- | --- |
| [`CV`](https://github.com/chinmoybiswasdeep/bosonic-qrc/tree/CV) | Persistent Gaussian loop with fresh squeezed pulses, Piquasso | Genuine temporal Legendre IPC smoke | Architecture and smoke gates pass; long calibration pending |
| [`DV`](https://github.com/chinmoybiswasdeep/bosonic-qrc/tree/DV) | Static Fock-input interferometric QRP, Perceval SLOS | Delay-zero nonlinear capacity smoke | Static gate passes; tomography and recurrent DV channel pending |

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

The CV two-seed IPC smoke respects the numerical-rank capacity bound but uses
only 20 null surrogates, so FDR significance is resolution-limited and no
significant total is claimed. The DV smoke exposes degrees 1-4 with a random
four-mode, two-photon reservoir; it reports 15 nominal outcomes, 10 nonzero
lossless columns, and numerical rank 8. Its identity control has rank zero after
scale-aware SVD filtering. These are pipeline checks, not publication results or
claims of quantum advantage.

Reference architectures:

- Garcia-Beni et al., *Scalable photonic platform for real-time quantum reservoir computing*, [arXiv:2207.14031](https://arxiv.org/abs/2207.14031).
- Lopez Carreno et al., *Quantum and classical processing with photonic quantum machine learning*, [arXiv:2605.10471](https://arxiv.org/abs/2605.10471).

Each research branch contains its own `CITATION.cff`, bibliography, exact
configuration, raw per-target results, manifests, limitations, and CI workflow.

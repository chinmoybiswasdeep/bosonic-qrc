# bosonic-qrc - DV branch

This branch implements backend-native discrete-variable static quantum reservoir
processing (QRP/QELM) with Perceval 1.2 and SLOS. It does not present the static
model as recurrent QRC.

The configurable fixed Haar-like unitary is decomposed by Perceval into beam
splitters and phase shifters. Fock inputs, sample-dependent dual-rail encoding,
optical evolution, exact probabilities, and finite-shot counts are handled by
Perceval. The four-mode, two-photon index includes all 1+4+10 zero-, one-, and
two-photon outcomes, retaining collision events and zero-valued lower sectors
in the lossless model.

## Install and verify

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\ruff check src tests
.\.venv\Scripts\pytest
.\.venv\Scripts\bosonic-qrc-dv configs/smoke/xor.yaml --output results/calibration/xor
```

The HOM limit remains an independent backend sanity test. The smoke XOR run
checks encoding and artifact generation; training accuracy is not a held-out
scientific claim. Tomography, source-ancilla reduction, loss models, spiral
calibration, and genuine temporal collision-model QRC remain calibration-gate
failures documented in `docs/LIMITATIONS.md`.

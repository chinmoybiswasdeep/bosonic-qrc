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
.\.venv\Scripts\bosonic-qrc-dv-static-capacity configs/smoke/static_ipc.yaml --output results/calibration/static_ipc_smoke
```

The HOM limit remains an independent backend sanity test. The smoke XOR run
checks encoding and artifact generation; training accuracy is not a held-out
scientific claim. Tomography, source-ancilla reduction, loss models, spiral
calibration, and genuine temporal collision-model QRC remain calibration-gate
failures documented in `docs/LIMITATIONS.md`.

The static-capacity command permits delay-zero Legendre targets only. It reports
15 nominal indexed outcomes, 10 nonzero lossless two-photon columns, and the
numerical rank separately. The committed smoke resolves degrees 1-4 for the
random reservoir and correctly gives rank zero to the identity control. Its 20
null surrogates are CI-scale, not calibration-scale inference. The committed
degree-1–6 calibration uses `B=119`, passes `PASS_CALIBRATION`, and finds nearly
perfect reservoir capacity while the identity control remains zero. That result
is explicitly held behind finite-shot/noise and stronger-control stress tests.
See `docs/REPRODUCIBILITY.md` for resume, chunk, merge, production, and reporting
commands.

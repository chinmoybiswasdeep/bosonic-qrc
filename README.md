# bosonic-qrc — DV branch

This branch provides a backend-native discrete-variable **static quantum
reservoir processing (QRP/QELM)** seed using Perceval 1.2 and its SLOS backend.
It explicitly does not label this static feature map as recurrent QRC.

Every PNR feature comes from `perceval.algorithm.Sampler(...).probs()`. The
fixed reservoir is a two-mode beam splitter and phase shifter; Fock input
states, circuit evolution, and photon-number outcome probabilities remain in
Perceval. Classical code only orders the backend-returned probabilities.

```powershell
.\.venv\Scripts\python -m pip install -e . -c dv_pyproject.toml
.\.venv\Scripts\pytest -c dv_pyproject.toml dv_tests
```

The included integration test checks backend invocation, normalization, and the
Hong--Ou--Mandel two-photon interference limit. Temporal DV recurrence,
tomography, loss, finite shots, and the requested experiment/ablation suite are
not implemented; this is a constrained physical baseline, not a claim of the
complete research repository described in the task brief.

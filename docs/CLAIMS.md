# Claims and non-claims

Calibration supports near-complete static degree-1–6 Legendre reconstruction
for the tested encoding/reservoir, while the identity control has rank and
capacity zero. Because this result is near perfect, it requires larger held-out
sets, finite-shot and noise sweeps, and stronger classical controls before
publication.

No temporal-memory or tomography claim is made. DV static capacity and CV
temporal IPC are different estimands and are not plotted as a direct winner.

# EOC claims ledger

| Statement | Status | Evidence needed / available |
|---|---|---|
| The added DV model is genuinely recurrent. | Supported in implementation/tests | Full memory density matrix survives fresh-ancilla collisions; causality and reset tests pass. |
| The interacting core is exact at smoke size. | Supported | Independent matrix, sparse action, phase, conservation, and unitarity tests. |
| Smoke finds candidate regular-to-chaotic crossovers. | Pipeline observation only | Reflection-resolved ratios plus stored SFF/OTOC curves at one small size. |
| Performance peaks at the EOC. | Not supported | Calibration/production paired task and IPC scans are not run. |
| An edge of many-body quantum chaos is established. | Not supported | Size, cutoff, time-step, controls, and statistical replication remain blocking gates. |

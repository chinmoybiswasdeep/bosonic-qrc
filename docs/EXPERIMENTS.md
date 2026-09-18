# Experiment protocol

## Core experiment

The DV implementation is a static Perceval SLOS quantum random processor. Its
core metric is complete delay-zero Legendre nonlinear capacity; it is not
temporal IPC. Reservoir and identity interferometers are evaluated on identical
input streams.

| Claim | Experiment | Controls | Required evidence | Current state |
|---|---|---|---|---|
| Static nonlinear capacity | complete degree basis | identity/direct PNR | production seeds and B | calibration only |
| Boundary enrichment | XOR/rings/spiral | raw input, identity, random features, RBF-SVM | held-out paired tests | XOR smoke only |
| Robustness | exact/finite shots/loss/drift/mismatch | matched seeds | production sweep | not implemented |
| Temporal memory | recurrent DV channel | delay controls | validated recurrence | unsupported |
| Tomography | truncated-Fock reconstruction | direct PNR | fidelity/error bars | unsupported |

Unsupported items are explicit blockers and are not replaced by proxies.


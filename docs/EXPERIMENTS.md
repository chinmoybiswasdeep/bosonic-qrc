# Experiment protocol

## Core experiment

The core CV claim is temporal information-processing capacity from the recurrent
Gaussian loop. Targets are a complete normalized-Legendre basis, grouped by
degree, maximum delay, and interaction order. Capacity is held-out
`C=max(0,R²)`; squared correlation is secondary.

Smoke validates plumbing. Calibration validates inference at modest scale.
Production is preregistered in `configs/production/ipc.yaml` and is not run by CI.

## Claim matrix

| Claim | Experiment | Controls | Required evidence | Current state |
|---|---|---|---|---|
| Temporal memory | CV IPC | delay-zero split, complete target Gram matrix | production seeds and B | calibration only |
| Quantum feature advantage | NARMA/channel/Mackey–Glass | input-only, tapped delay, ESN, random features, NVAR, classical Gaussian | paired hierarchical CI | not implemented |
| Robustness | IPC/task sweeps | loss, shots, drift, mismatch | predefined grid | not implemented |
| Tomography | truncated-Fock reconstruction | direct moments | fidelity/error bars | unsupported |

Unsupported items are blockers; they are not silently replaced by proxies.


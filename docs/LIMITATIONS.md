# Scientific limitations

The implemented QRP is a configurable, backend-native fixed interferometer with
PNR features and a dual-rail classical encoder. It is not yet the paper's mixed
state tomography pipeline. Lower photon sectors are present in the deterministic
15-element index but have zero weight in the current lossless two-photon run.
Finite-shot sampling uses Perceval but deterministic sampler seeding is not
exposed by this local workflow. No temporal recurrence, source impurity,
distinguishability, or detector loss is claimed.

Temporal IPC is deliberately unavailable because no recurrent Perceval channel
has passed trace-preservation, causality, and fading-memory tests. Mixed-state
source/ancilla tomography is also not implemented, so this branch is not yet a
reproduction of the reference tomography architecture.

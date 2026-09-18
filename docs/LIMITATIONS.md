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
# EOC-specific limitations

- Only open-chain reflection symmetry is currently resolved. Ring construction exists, but
  translation/momentum resolution must be added before ring level statistics are interpreted.
- The smoke scan is too small for a thermodynamic or universality claim, and candidate crossings
  can be multiple or finite-size artifacts.
- Perceval is a static passive control; the recurrent Kerr channel is a SciPy Fock implementation.
- Wigner negativity is not applicable to this DV route. Finite-shot/noisy task matrices and all
  confound controls remain for calibration/production.
- SFF and OTOC curves are recorded, but no Thouless or Lyapunov fit is claimed.

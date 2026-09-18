# Claims and non-claims

Supported at calibration scale: the implemented recurrent Gaussian loop exposes
nonzero held-out temporal IPC under the declared feature map.

Not established: advantage over classical baselines, hardware robustness,
production-scale uncertainty, or tomography. Cross-branch numbers are not
compared as the same quantity: CV is temporal IPC, while the DV branch currently
reports static delay-zero capacity.

# EOC claims ledger

| Statement | Status | Evidence |
|---|---|---|
| The new CV route is recurrent and non-Gaussian. | Supported in implementation/tests | Full Fock state persists; Kerr/Bose–Hubbard interaction and state-transfer tests pass. |
| Piquasso Kerr convention matches the reference. | Supported | Native phase test against \(e^{-iU\Delta t n(n-1)/2}\). |
| Smoke finds finite-size candidate chaos crossings. | Pipeline observation only | Reflection-resolved ratio, SFF, and OTOC artifacts. |
| Performance peaks near an EOC. | Not supported | Paired IPC/task calibration is not run. |
| Edge of many-body quantum chaos established. | Not supported | Size/cutoff/time-step/control/statistical gates are open. |

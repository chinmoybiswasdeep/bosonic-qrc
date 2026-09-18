# EOC-CV architecture

The original Piquasso Gaussian loop remains unchanged. `bosonic_qrc_cv.eoc.fock_loop` is a separate
non-Gaussian route with persistent modes; fresh coherent, squeezed, cat, number, or configured Fock
resources; input encoding only on the fresh mode; Bose–Hubbard/Floquet/Strang interaction; and
configurable loss. It transfers the complete surviving state.

`core.py` supplies independently validated Fock matrices and state metrics. Piquasso 8.0.1 is used
for the established Gaussian baseline and a native-Kerr convention cross-check; the multimode
recurrent interacting layer uses explicit SciPy matrices so backend attribution stays exact.
`chaos.py`, `tasks.py`, and `study.py` provide physical scans, leakage-safe datasets/baselines,
figures, checksummed manifests, deterministic chunks, and resume.

Features include photon means/correlations, \(g^{(2)}\), parity, PNR probabilities, selected moments
through fourth order, and a matched Gaussian subset. Wigner negativity is optional/not implemented
in the smoke path and is never treated as a chaos diagnostic.


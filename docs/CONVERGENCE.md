# Numerical convergence

Variable-number basis states satisfy total photon number `< cutoff`. Reports include trace,
Hermiticity error, minimum eigenvalue, boundary-shell population, retained resource norm, mean and
maximum photon number, and aligned fidelity/trace distance between successive cutoffs. Thresholds
are predeclared in YAML.

The smoke cutoff-3→4 comparison is a diagnostic, not convergence: its state trace distance is about
0.039 despite a small terminal shell population. Calibration/publication therefore extend both
cutoff and fixed-number size before any claim. Exact evolution is also compared against 1/2/4
Strang substeps to detect artificial Floquet/Trotter behavior.

Fixed-number dense storage is roughly \(16D^2\) bytes, \(D={M+N-1\choose N}\). A total-cutoff
Fock density matrix has \({M+c-1\choose M}^2\) complex entries. Chunk open interaction points and
monitor measured runtime/memory before moving from calibration to publication.


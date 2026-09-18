# Numerical convergence

The config fixes tolerances before execution. Each manifest records Hilbert dimension, exact model,
cutoff sequence, boundary population, trace, minimum eigenvalue, runtime, environment, and Git SHA.
Exact continuous evolution is compared with one, two, and four Strang substeps. Local-Fock cutoff
states are aligned before fidelity and trace-distance comparisons.

Smoke uses very small systems and only cutoffs 3 and 4. Its current terminal boundary population is
below the configured 0.05 guard, but two points do not establish cutoff convergence. Calibration
adds cutoff 5/6 and a larger fixed-number sector; publication adds a further size and cutoff. A
reported boundary or performance trend must remain stable under all three refinements.

Approximate fixed-number dense storage scales as \(16D^2\) bytes for a complex matrix, with
\(D={M+N-1\choose N}\); diagonalization needs additional work arrays. Local-Fock collision space
scales as \(c^{M+1}\). Use chunks across open-system interaction points, never within a correlated
trajectory.


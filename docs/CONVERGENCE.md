# Numerical convergence

The total-photon basis uses exclusive cutoff semantics. Manifests record trace, Hermiticity,
positivity, boundary-shell population, retained norm, photon statistics, aligned state distances,
Hilbert dimension, timing, exact/Floquet/Trotter label, and 1/2/4-step refinement.

Smoke cutoffs 3 and 4 satisfy the configured shell/norm guard but do not establish convergence.
PhotoGraphiQ truncation warnings are preserved as evidence rather than suppressed. Calibration and
publication increase both cutoff and fixed-number size; any boundary/performance conclusion must be
stable to both.

Fixed-number dimension is \({M+N-1\choose N}\); total-cutoff dimension is
\({M+c-1\choose M}\), and a dense mixed state squares that dimension. Use interaction chunks and
measure resource use before production. Conditional cutoff comparisons must specify the same
physical outcome; smoke uses unconditional evolution.


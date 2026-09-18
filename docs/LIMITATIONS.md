# Scientific limitations

This branch implements the corrected N+N Gaussian topology and supports exact
detector covariance and Piquasso-shot homodyne estimates, but not the paper's
complete capacity study. The retained state is re-prepared from Piquasso's
reduced moments at each causal step. Finite-shot measurement is separate from
unconditional loop reduction, representing ensemble-averaged rather than
trajectory-conditioned back-action. Detector inefficiency and higher-order
observables are absent. No result is a claim of quantum advantage.

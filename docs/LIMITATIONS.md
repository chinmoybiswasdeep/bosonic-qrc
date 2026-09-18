# Scientific limitations

This branch is an executable, backend-native Gaussian feedback-loop baseline,
not a reproduction of every experiment in arXiv:2207.14031. It uses exact
Gaussian expectation covariances, so it does not yet model finite-shot homodyne
noise, detector inefficiency, or non-Gaussian observables. The retained state is
re-prepared from Piquasso's reduced state moments at each causal step; all optical
evolution remains Piquasso execution. No result should be construed as a claim
of quantum advantage.

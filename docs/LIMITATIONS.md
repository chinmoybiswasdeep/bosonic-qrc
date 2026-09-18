# Scientific limitations

This branch implements the corrected N+N Gaussian topology and supports exact
detector covariance and Piquasso-shot homodyne estimates, but not the paper's
complete capacity study. The retained state is re-prepared from Piquasso's
reduced moments at each causal step. Finite-shot measurement is separate from
unconditional loop reduction, representing ensemble-averaged rather than
trajectory-conditioned back-action. Detector inefficiency and higher-order
observables are absent. No result is a claim of quantum advantage.

The committed IPC run is a two-reservoir-seed smoke test with 20 null
surrogates. Benjamini-Hochberg inference is resolution-limited at that surrogate
count, so zero significant CV targets is recorded rather than reinterpreted.
# EOC-specific limitations

- Ring Hamiltonians are available, but translation sectors are not yet resolved; current chaos
  configs therefore use open chains and reflection sectors.
- Smoke sizes and two cutoff points cannot support a phase-boundary claim. Boundary-shell weight
  alone is insufficient; the larger cutoff state distance must converge too.
- Native Piquasso Kerr is convention-tested, but the recurrent multimode engine is explicitly a
  SciPy Fock adapter. Wigner negativity is not implemented in the smoke workflow.
- Full task/IPC controls, finite-shot/noisy matrices, energy/effective-rank matching, Thouless-time
  inference, and any stable Lyapunov fit remain production work.

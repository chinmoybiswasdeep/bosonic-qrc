# Edge-of-chaos methodology (EOC-CV)

The full common methodology, formulas, claim policy, and references are implemented in
`science.py`, `channel.py`, `performance.py`, and `workflow.py`.  This branch uses exactly the
number-conserving two-kick Bose--Hubbard calibration
`U_F=exp(-i H_U T_U) exp(-i H_J T_J)` in fixed-`N` sectors.  Seeded zero-mean disorder and
phase kicks break spatial symmetries; the real protocol supports COE only.  Circular ratios
include the wrap gap and fail on unresolved degeneracy.  SFF ensembles never mix `U` values.
IPR/participation entropy, bipartite entropy, and OTOC independently accompany spectral data.
The finite-size entry crossing and bootstrap CI define the same
`d_EOC=(U-U_EOC)/max(closed grid spacing, half CI width)` used by the open scan.

## CV implementation

The open system has four memory modes in a total-Fock basis.  A fresh-mode projected-unitary
coherent encoding and a beam-splitter collision are followed by unconditional PNR reduction,
the same hopping plus onsite `n(n-1)` Kerr Floquet kicks, and complete multimode loss.  The
Kerr term is deterministic and non-Gaussian; it is not postselection and its feature-change
witness is not itself called chaos.  The projected preparation is evaluated from a cached
eigendecomposition and every parameter-independent matrix is cached.  Piquasso 8.0.1 is the
convention/reference backend; the explicit total-Fock adapter is used where mixed composite
channels and diagnostics require direct Kraus access.

The raw output trace is tested before normalization.  CP uses stage Choi Gram certificates,
TP uses the full adjoint, and fixed point, traceless-subspace `|lambda_2|`, Arnoldi Ritz
residual, traceless HS contraction, and non-normality are saved.  Echo curves compare vacuum
and maximally mixed initial states under identical drives.

Total cutoffs are compared under identical inputs.  Raw trace error, highest shell,
common-space trace distance/fidelity, observables, and task scores are gated; 0.04 trace
distance cannot pass.  Smoke uses cutoffs 3/4, calibration 3/4/5, publication 4/5/6.

Tasks are linear memory, nonlinear/cross-delay IPC, parity, NARMA10, Mackey--Glass, and
nonlinear equalization with independent washout splits, training-only scaling,
validation-only regularization, and untouched test data.  Smoke is not a scientific claim;
publication requires ten seeds and robustness to size and cutoff.

```bash
bosonic-qrc-eoc-cv run configs/eoc/smoke.yaml --output results/eoc/corrected-smoke-v2
bosonic-qrc-eoc-cv run configs/eoc/calibration.yaml --output results/eoc/calibration-v2
bosonic-qrc-eoc-cv run configs/eoc/publication.yaml --output results/eoc/pub-0 --chunk-index 0 --chunk-count 4
bosonic-qrc-eoc-cv merge results/eoc/pub-0 results/eoc/pub-1 results/eoc/pub-2 results/eoc/pub-3 --output results/eoc/publication-v2
```

Expected scale is under one minute/under 0.5 GB for smoke, hours/1--4 GB for calibration,
and several CPU-days/4--16 GB for publication.  Limitations are small exact sectors,
explicit disorder instead of symmetry projection, COE-only hopping, total-cutoff artifacts,
approximate but residual-gated Arnoldi eigenvalues, and no demonstrated EOC advantage or
hardware quantum advantage.

Primary sources: Martínez-Peña et al., PRL 127, 100502 (2021),
[DOI](https://doi.org/10.1103/PhysRevLett.127.100502); Kolovsky and Buchleitner, EPL 68,
632 (2004), [DOI](https://doi.org/10.1209/epl/i2004-10265-7); Bertini, Kos, and Prosen,
PRL 121, 264101 (2018), [DOI](https://doi.org/10.1103/PhysRevLett.121.264101); Nokkala et
al., Communications Physics 4, 53 (2021),
[DOI](https://doi.org/10.1038/s42005-021-00556-w); García-Beni et al.,
[arXiv:2207.14031](https://arxiv.org/abs/2207.14031); Dambre et al., Scientific Reports 2,
514 (2012), [DOI](https://doi.org/10.1038/srep00514); Yusipov et al.,
[arXiv:1806.09295](https://arxiv.org/abs/1806.09295); Oganesyan and Huse,
[DOI](https://doi.org/10.1103/PhysRevB.75.155111).

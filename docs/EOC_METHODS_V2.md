# Edge-of-chaos methodology (EOC-CVMB)

This branch uses the same fixed-number, exact two-kick Bose--Hubbard Floquet calibration and
the same signed `d_EOC` as EOC-DV and EOC-CV.  Seeded onsite disorder and kick phases break
spatial symmetries while particle number selects the sector.  The real protocol is COE-only.
Circular adjacent-gap ratios fail on degeneracy; fixed-parameter SFF ensembles, bootstrap
bands, IPR/participation entropy, bipartite entropy, OTOC, all entry/recrossings, and
inverse-dimension edge extrapolation are stored.  Smoke output never supports a physics claim.

## Backend capability decision and adapter

PhotoGraphiQ was inspected as the resource/measurement reference.  Its public workflow does
not expose the arbitrary four-mode mixed-state Kerr--hopping--loss composite channel needed
for raw Kraus/superoperator validation.  The public reservoir interface is therefore backed
by a manual total-Fock adapter.  It supports four interacting modes, beam-splitter hopping,
onsite Kerr, mixed density matrices, all loss sectors, unconditional PNR averaging, and a
seeded optional conditional trajectory.  The fresh measured mode is mode 4; the memory modes
are 0--3.  The Kerr interaction is deterministic and non-Gaussian, not a heralded chaos test.
All propagators, embeddings, loss matrices, and preparation eigendecompositions are cached.

The full injection--collision--measurement--Floquet--loss channel is diagnosed before any
normalization.  Results include stage Choi-Gram CP certification, full-adjoint TP residual,
fixed point, traceless-subspace `|lambda_2|` with Arnoldi residual, traceless HS contraction,
non-normality, and common-input trace-distance echo bands.  Cutoffs use common inputs and
report raw trace, shell weight, projected trace distance/fidelity, observable differences,
and calibration/publication task-score differences; 0.04 cannot pass.

The task runner covers linear memory, nonlinear/cross-delay IPC, delayed parity, NARMA10,
Mackey--Glass, and nonlinear channel equalization.  Splits reset independently; preprocessing
uses training only, ridge selection validation only, and test once.  Multiple seeds and
regional effect sizes are evaluated against `d_EOC`, not raw `U`.

```bash
bosonic-qrc-eoc-cvmb run experiments/eoc/configs/smoke.yaml --output experiments/eoc/results/corrected-smoke-v2
bosonic-qrc-eoc-cvmb run experiments/eoc/configs/calibration.yaml --output experiments/eoc/results/calibration-v2
bosonic-qrc-eoc-cvmb run experiments/eoc/configs/publication.yaml --output experiments/eoc/results/pub-0 --chunk-index 0 --chunk-count 4
bosonic-qrc-eoc-cvmb merge experiments/eoc/results/pub-0 experiments/eoc/results/pub-1 experiments/eoc/results/pub-2 experiments/eoc/results/pub-3 --output experiments/eoc/results/publication-v2
```

Expected scale is under one minute/under 0.5 GB for smoke, hours/1--4 GB for calibration,
and several CPU-days/4--16 GB for publication.  Limitations are the manual adapter, small
exact sectors, explicit symmetry breaking, COE-only hopping, total-cutoff artifacts,
residual-gated approximate Arnoldi values, and no demonstrated EOC or quantum advantage.

Primary sources: Martínez-Peña et al., PRL 127, 100502 (2021),
[DOI](https://doi.org/10.1103/PhysRevLett.127.100502); Kolovsky and Buchleitner, EPL 68,
632 (2004), [DOI](https://doi.org/10.1209/epl/i2004-10265-7); Bertini, Kos, and Prosen,
[DOI](https://doi.org/10.1103/PhysRevLett.121.264101); Nokkala et al.,
[DOI](https://doi.org/10.1038/s42005-021-00556-w); García-Beni et al.,
[arXiv:2207.14031](https://arxiv.org/abs/2207.14031); Dambre et al.,
[DOI](https://doi.org/10.1038/srep00514); Yusipov et al.,
[arXiv:1806.09295](https://arxiv.org/abs/1806.09295); Oganesyan and Huse,
[DOI](https://doi.org/10.1103/PhysRevB.75.155111).

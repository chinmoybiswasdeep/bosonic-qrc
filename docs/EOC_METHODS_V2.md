# Edge-of-chaos methodology (EOC-DV)

## Scope and claim policy

This workflow tests whether reservoir performance is largest near the *entry* into a
many-body-chaotic Floquet regime.  A smoke result is a code-path validation, not evidence
for an edge-of-chaos advantage.  Gaussian instability, nonlinearity, a small channel gap,
or good task performance is never relabelled as many-body quantum chaos.

## Common closed and open dynamics

The calibration sector uses the fixed-number Fock basis `sum(n_i)=N` and

```text
H_J = -J sum_<ij> (a_i^dagger a_j + h.c.) + sum_i epsilon_i n_i
H_U = (U/2) sum_i n_i(n_i-1) + sum_i phi_i n_i
U_F = exp(-i H_U T_U) exp(-i H_J T_J).
```

Both exponentials are exact dense matrix exponentials; there is no Trotter error.  Open
boundaries plus seeded, zero-mean onsite disorder and phase kicks break translation and
reflection reproducibly.  Therefore spectra are never concatenated across spatial sectors.
The remaining exact rule is fixed total particle number, whose dimension and basis rule are
saved.  The generators are real and admit a symmetric Floquet time origin, so the supported
benchmark is COE (GOE-like adjacent-gap mean 0.5307), not CUE.  A CUE setting is rejected
until a Peierls-flux extension is implemented.

The open DV reservoir is the local truncated-boson representation of the same four-mode
model, with local occupations `0..q-1`.  Each step injects a fresh qudit ancilla, applies a
number-preserving memory--ancilla collision, averages all PNR outcomes, applies the same
two-kick memory Floquet propagator, then all local amplitude-damping sectors.  The principal
development point is four modes and `q=3`; `q=4` is the convergence comparison.  Collision,
Floquet, and loss operators are cached.

## Chaos and edge estimators

Eigenphases are sorted on the circle, including the wraparound gap.  Every gap below the
configured tolerance is counted.  If any realization has such a gap, ratios are not computed
for it and the calibration fails rather than deleting the gap.  The SFF
`K(t)=mean(|Tr(U_F^t)|^2)` is formed from independent disorder/kick realizations at one fixed
`U`, size, and sector.  Raw, dimension-normalized, connected, bootstrap intervals, ensemble
size, and the COE curve are stored.  Small profiles do not estimate a Thouless time.

Independent evidence comprises Fock-basis IPR/participation entropy, bipartite eigenstate
entropy, and a number-operator OTOC.  Entry crossings of a normalized gap-ratio score are
bootstrapped at every size.  All later recrossings are retained as ambiguity data.  The entry
is extrapolated linearly in inverse sector dimension.  This is a finite-size candidate, not a
critical-point claim.

The shared signed coordinate is

```text
d_EOC = (U - U_EOC) / max(minimum closed-grid spacing, half the 95% CI width).
```

Negative values are on the regular side.  The same closed-grid scale is used in open output.

## Complete open channel and cutoff gate

Diagnostics act on the composed injection--collision--PNR-average--Floquet--loss channel.
The raw trace is checked before any normalization.  CP is certified by positive stage Choi
Gram matrices and closure under composition; TP is evaluated by applying the full adjoint to
identity.  The stationary state is refined and checked.  The subleading eigenvalue is found
on the traceless invariant subspace; large spaces use deterministic Arnoldi and save the Ritz
residual.  A separate traceless Hilbert--Schmidt contraction and non-normality diagnostic are
reported, so a singular value is not called a channel gap.

Cutoffs use identical inputs and seeds and report raw-trace error, boundary occupation,
common-space trace distance/fidelity, observable error, and (calibration/publication) task
score error.  The DV closed-sector ratio is also compared between local cutoffs.  A trace
distance of 0.04 fails every supplied profile.

## Task protocol and reproducibility

The runner includes linear memory, nonlinear polynomial IPC including cross-delays, delayed
parity, NARMA10, Mackey--Glass prediction, and nonlinear channel equalization.  Train,
validation, and test trajectories are independently reset; scaling is fitted on training
only, ridge strength on validation only, and test is used once.  Calibration has three seeds
and publication has ten.  Seed intervals, regional effects, correlations, and peak location
are functions of `d_EOC`.

Writes are atomic.  Resume verifies hashes, chunks are deterministic, merge requires exact
grid coverage, and a closed-calibration JSON can be shared by chunks.  Exact commands:

```bash
bosonic-qrc-eoc-dv run configs/eoc/smoke.yaml --output results/eoc/corrected-smoke-v2
bosonic-qrc-eoc-dv run configs/eoc/calibration.yaml --output results/eoc/calibration-v2
bosonic-qrc-eoc-dv run configs/eoc/publication.yaml --output results/eoc/pub-0 --chunk-index 0 --chunk-count 4
bosonic-qrc-eoc-dv merge results/eoc/pub-0 results/eoc/pub-1 results/eoc/pub-2 results/eoc/pub-3 --output results/eoc/publication-v2
```

Profile estimates on a modern workstation are roughly 1 minute/under 0.5 GB for smoke,
hours/1--4 GB for calibration, and several CPU-days/4--16 GB for publication.  Profile first;
do not launch publication until smoke, channel, cutoff, and calibration gates pass.

## Limitations

The inverse-dimension extrapolation uses small sectors; disorder explicitly replaces spatial
symmetry projection; only COE is implemented; Arnoldi values are approximate and gated by a
saved residual; local `q=4` task convergence is expensive; no hardware noise model or quantum
advantage claim is included.  Excessive mixing is identified from the complete channel and
echo curves, but publication inference requires a dedicated loss/collision control sweep.

## Verified primary references

- Martínez-Peña et al., *Dynamical phase transitions in quantum reservoir computing*,
  PRL 127, 100502 (2021), [arXiv:2103.05348](https://arxiv.org/abs/2103.05348),
  [DOI](https://doi.org/10.1103/PhysRevLett.127.100502).
- Martínez-Peña et al., *Information Processing Capacity of Spin-Based Quantum Reservoir
  Computing Systems*, Cognitive Computation 15, 1440 (2023),
  [DOI](https://doi.org/10.1007/s12559-020-09772-y).
- Kolovsky and Buchleitner, *Quantum Chaos in the Bose-Hubbard Model*, EPL 68, 632 (2004),
  [DOI](https://doi.org/10.1209/epl/i2004-10265-7).
- Oganesyan and Huse, adjacent-gap ratios, PRB 75, 155111 (2007),
  [DOI](https://doi.org/10.1103/PhysRevB.75.155111).
- Bertini, Kos, and Prosen, *Exact Spectral Form Factor in a Minimal Model of Many-Body
  Quantum Chaos*, PRL 121, 264101 (2018),
  [DOI](https://doi.org/10.1103/PhysRevLett.121.264101).
- Dambre et al., *Information processing capacity of dynamical systems*, Scientific Reports
  2, 514 (2012), [DOI](https://doi.org/10.1038/srep00514).
- Yusipov et al., *Lyapunov exponents of quantum trajectories beyond continuous
  measurements*, [arXiv:1806.09295](https://arxiv.org/abs/1806.09295).

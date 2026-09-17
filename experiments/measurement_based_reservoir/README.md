# Measurement-based reservoirs

Run from the `cv-mb-qrc` repository root after installing the audited PhotoGraphiQ checkout:

```sh
python -m pip install -e ".[dev,docs,experiments,graphix]"
python -m pip install -r tests/mentpy_reference/requirements.txt
python experiments/measurement_based_reservoir/main.py --config experiments/measurement_based_reservoir/config_small.json
python experiments/measurement_based_reservoir/reproduce.py
python -m pytest tests/reservoirs
```

On Windows the existing interpreter is `.venv/Scripts/python.exe`. No pushes or
remote writes are needed. `--output PATH` isolates a run. `--resume` accepts only
the identical saved config and resumes completed manifest entries. Exceptions
are saved with configuration and traceback and re-raised. Windows atomic writes
retry transient file-sharing errors for a bounded interval.

`config_small.json` uses ten reservoir seeds, 360 samples per locally generated
dataset, chronological 60/20/20 boundaries, a five-sample embargo and 20-sample
independent washout. This is a smoke study, not a publication-scale capacity
estimate: only 47 or 46 test examples remain per split. `config_full.json`
increases samples, seeds, delays and the physical transmission sweep. Full scale
execution is explicitly separate; do not label the small results publication-grade.

The dataset seed is shared across methods and reservoir seeds. Thus uncertainty
is across reservoir initialization conditional on these datasets, not uncertainty
over all possible time series. Mean, sample SD, median, individual seed values and
95% percentile bootstrap intervals (2,000 draws, seed zero) are saved. No
significance test or advantage claim is made.

## Models and input ownership

CV A/B use complete persistent Gaussian memory states and fixed seeded couplings.
Graphix uses a one-memory/one-ancilla collision graph with probability-weighted
outcome branches. WindowedMBQELM replays an explicit trailing window on a fresh
resource. It is not recurrent. Classical delay/RFF baselines receive the same
explicit window. ESN and input-only controls have 3 and 14 features, matching
Graphix and CV Tier B respectively. Tier A has 8. The delay model has `window`
features. These are disclosed feature comparisons, not equal physical budgets.
All readouts also receive the current input and an unpenalized intercept.

All quantum reservoir parameters stay fixed. Training fits the readout's means,
standard deviations and ridge weights on train only. Validation selects from the
declared ridge grid. Test is evaluated after selection. Every split resets state,
rebuilds lag targets inside its own boundaries and applies washout; no overlap
crosses a boundary. Changing seeds to improve test scores is prohibited.

Controls include zero CV coupling, no temporal state transfer, disabled CV
feed-forward, fixed measurement angles, Gaussian Tier A, windowed CV, a Graphix
no-entanglement control, linear delayed-input ridge, RFF, ESN, input-only tanh and
the Mackey–Glass persistence predictor. Removing feed-forward is not removing
the destructive measurement. A measurement-disabled reservoir is not asserted.

## Equations and metrics

For iid `u ~ Uniform[-1,1]`, linear targets are `sqrt(3) u[t-k]`, quadratic self
targets `sqrt(5)(3u[t-k]^2-1)/2` and cross targets `3u[t-a]u[t-b]` for distinct
positive delays. These are orthonormal under the input distribution. Raw test R²
is retained; reported total capacities sum `max(0,R²)`. This finite-sample clipping
has positive bias and no theoretical bound is invoked.

Temporal parity is the product of three delayed input signs. Regression outputs
are thresholded at zero for accuracy. NARMA10 uses `v=(u+1)/4` and
`y[t+1]=.3y[t]+.05y[t]sum(y[t-9:t+1])+1.5v[t-9]v[t]+.1`, with zero initial history.
The first ten steps are initialization and are covered by washout.

Mackey–Glass uses `dx/dt=.2x(t-17)/(1+x(t-17)^10)-.1x(t)`, constant history 1.2,
forward Euler step .1, unit-time sampling, and 1,000 unit-time burn-in samples.
The target is the next sample. This is a specified numerical dataset, not an exact
delay-equation solution; integration-step convergence is not claimed.

`NMSE=MSE/Var(test target)`, `NRMSE=sqrt(NMSE)`, `R²=1-NMSE`. Constant targets and
nonfinite values raise errors. Figures plot raw R² or explicitly clipped capacity.
Timing includes feature generation, model construction and readout selection;
per-step compilation metadata is separate. Process RSS is sampled between runs,
including interpreter/import memory and possibly missing short allocation peaks.

## Exact, conditional and sampled semantics

The CV exact path uses PhotoGraphiQ's unconditional Gaussian channel analyzer.
Selecting `backend='piquasso'` selects physical gates for **trajectory execution**;
it does not relabel the exact analyzer as Piquasso. Graphix exact execution
enumerates two branches and weights them by actual probability. Conditional
execution retains one sampled branch and carries that branch to the next step.

`shots=N` means N independently persistent measurement trajectories. Features are
exact conditional-state expectations within trajectories; the average estimates
the unconditional expectation. Gaussian covariance estimates additionally include
between-trajectory means. This is a Monte Carlo trajectory budget, **not** a full
experimental detector-shot budget for reading every output observable. Features
are simulated nondestructively; experimental tomography/readout replication costs
are not included. No exact results silently replace a requested trajectory run.

Gaussian number and quadrature-square features are Tier B and efficiently
Gaussian-simulable. The optional `reservoirs.fock` module uses an even-cat ancilla,
mixed-Fock state transfer, CZ, PNR and loss. `cutoff_study` selects the same zero-PNR
branch at each predeclared exclusive total-photon cutoff, includes its success
and complement probability, and checks selected observables. This is a numerical
control, not a non-Gaussian task comparison or whole-state convergence certificate.

MentPy is a corrected pure-wire reference only. Node/bit order, radians versus
Graphix units of pi, zero-branch convention and global phase are resolved. It
does not validate arbitrary mixed collision channels or CV numerical physics.

## Results and regeneration

`results/raw` holds datasets, split indices, environment, configuration, per-method
per-seed features/targets/predictions, validation-selected regularization and
resource counts. `manifest.json` defines completed task runs. The reproducer
reads only raw data, then writes CSV, JSON summaries, Markdown tables and SVG/PDF/PNG
figures. It never embeds scores. Do not delete failure records when retrying.

The design audit is in
`docs/research/measurement_based_quantum_reservoir_design.md`; the executable
tutorial notebook is `notebooks/06_measurement_based_quantum_reservoir.ipynb`.

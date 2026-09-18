# Methods

Inputs are IID Uniform[-1,1]. Perceval SLOS supplies exact probabilities and
Perceval sampling supplies finite-shot frequencies. Ordered PNR outcomes include
collisions and lower sectors when configured; nominal dimension,
nonzero/nonconstant dimension, and numerical rank are reported separately.

The principal readout is a centered SVD pseudoinverse and capacity is held-out
`C=max(0,R²)`. Ridge is selected on a separate validation stream with
train-only scaling. Null targets come from independent IID streams and repeat
the complete fit/validation procedure. P-values use the add-one formula, BH
within each model/run family, plus a max-statistic robustness threshold.

# EOC extension

The EOC workflow first scans the closed Bose–Hubbard core without reference to task scores, then
maps open-channel and performance quantities to signed distance from each candidate boundary.
Spacing ratios are never pooled across reflection sectors. Spectral form factors and OTOCs are
reported alongside, not replaced by, the ratio statistic. Temporal task utilities use chronological
train/validation/test partitions, an embargo, split-specific reset/washout, training-only scaling,
validation-only ridge selection, and an untouched test set. IPC definitions follow Dambre et al.
[@dambre_2012], while fading-memory interpretation follows the extended echo-state framework
[@kobayashi_tran_nakajima_2024].

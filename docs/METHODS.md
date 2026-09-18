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


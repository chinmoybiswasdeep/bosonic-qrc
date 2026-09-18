# Methods

Inputs are IID Uniform[-1,1]. Independent train, validation, and test streams
are selected by recorded seeds. The reservoir uses Piquasso's
`GaussianSimulator`; quantum transformations and homodyne sampling stay in
that backend.

The principal readout is a centered SVD pseudoinverse. Numerical rank uses
`max(absolute_tolerance, relative_tolerance*s_max)`. Stable rank,
participation rank, condition number, and the full singular spectrum are
recorded. Ridge is a robustness analysis selected only on validation data with
train-only scaling.

Null targets use independently regenerated IID input streams for each target
and surrogate. The complete fit and validation procedure is repeated. P-values
are `(1 + #null>=observed)/(B+1)`; BH correction is within each run's
preregistered target family. A max-statistic threshold is also reported.


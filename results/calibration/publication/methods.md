# Methods fragment

Inputs were IID Uniform[-1,1]. The principal readout was a centered SVD pseudoinverse and the principal held-out metric was C=max(0,R²). Ridge regularization was selected on a separate validation split using train-only scaling. Null targets were generated from independent IID streams and the complete fit/validation procedure was repeated. P-values used the add-one formula and Benjamini-Hochberg correction was applied within each declared target family. Uncertainty used a hierarchical bootstrap over reservoir seeds and data seeds.

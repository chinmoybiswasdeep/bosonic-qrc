# Statistics

Independent units are reservoir seeds, data seeds nested within reservoir
seeds, and measurement seeds when finite-shot sampling is enabled. Target rows
are not treated as independent replicates. Aggregate uncertainty uses a
hierarchical bootstrap over reservoir seeds and then data seeds.

Reports include mean, median, standard deviation, IQR, and bootstrap 95%
intervals. Capacity bounds are checked separately for raw and FDR-significant
totals. Confirmatory inference requires `1/(B+1) <= q/M`; production also
requires `B>=4999`. Smoke may pass pipeline checks while explicitly failing
confirmatory resolution.


# Edge-of-chaos theory

The closed reference model is the number-conserving Bose–Hubbard Hamiltonian

\[
H=-J\sum_{\langle i,j\rangle}(a_i^\dagger a_j+a_j^\dagger a_i)
 +\frac{U}{2}\sum_i n_i(n_i-1)+\sum_i\epsilon_i n_i.
\]

`exact` means \(e^{-iH\Delta t}\). `floquet` means the declared symmetric product
\(e^{-iH_{int}\Delta t/2}e^{-i(H_{hop}+H_{dis})\Delta t}e^{-iH_{int}\Delta t/2}\)
and is exact for that stroboscopic model. `trotter` is a Strang approximation to continuous
evolution and is always labelled with its substep count.

The physical scan precedes every task comparison. It uses fixed-particle-number spectra,
separate open-chain reflection sectors, a predeclared edge trim, bootstrap intervals for adjacent
spacing ratios, connected/unconnected spectral form factors, and infinite-temperature number or
parity OTOCs. Candidate crossings are descriptive interpolation points, not phase transitions.
This ordering follows the many-body QRC questions posed by Kobayashi and Motome
[@kobayashi_motome_2026] and the distinction between dynamical transitions and computational
performance emphasized by Martínez-Peña et al. [@martinez_pena_2021].

The recurrent reservoir is a collision channel
\(\rho'_M=\sum_yK_y(u)\rho_MK_y^\dagger(u)\). Every step prepares a fresh ancilla, couples it to
the retained memory, and traces or samples the ancilla. Completeness of the PNR Kraus family is
tested. Mixing gaps and equal-input trace-distance contraction are open-channel diagnostics; they
are not relabelled as closed-system chaos.


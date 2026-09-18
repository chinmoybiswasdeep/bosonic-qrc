# Edge-of-chaos theory

The common closed reference is the fixed-number Bose–Hubbard Hamiltonian. The implementation
separates exact \(e^{-iH\Delta t}\), the exact declared symmetric Floquet product, and approximate
Strang evolution. It locates finite-size candidate boundaries before task evaluation using separate
open-chain reflection sectors, fixed spectral trimming, bootstrapped spacing ratios, connected and
unconnected spectral form factors, and number/parity OTOCs.

The retained measurement-based memory evolves conditionally as
\(\rho_t^{(m)}=U_FK_m\rho_{t-1}K_m^\dagger U_F^\dagger/p(m)\), or unconditionally as the Born
sum without equal averaging. Direct interacting memory is the primary model; cat plus cubic-phase
resource injection is an ablation. Measurement-feedback instability, channel-gap closing, and
Gaussian instability remain logically distinct from many-body chaos. This design follows the
physical-before-performance question in Kobayashi and Motome [@kobayashi_motome_2026] and the
measurement/dissipation cautions in Sannia et al. [@sannia_2024].


# EOC-CVMB architecture

The existing Gaussian and one-mode reservoirs remain intact. `cv_mb_qrc.eoc.interacting` adds a
multimode mixed-Fock reservoir with persistent memory, fresh PhotoGraphiQ resources, fresh-mode-only
displacement encoding, memory–ancilla beam splitters, destructive PNR, causal feed-forward, direct
Bose–Hubbard/Floquet/Strang evolution, and loss. It carries the complete reduced memory state.

PhotoGraphiQ 0.3.1 supplies mixed Fock preparation/evolution, cat and number resources, PNR,
reduction, and loss. A local SciPy adapter supplies the explicitly defined multimode Bose–Hubbard
unitary and shared state metrics. The adapter is intentionally minimal and does not modify
PhotoGraphiQ. `core.py`, `chaos.py`, `tasks.py`, and `study.py` share the validated reference,
diagnostics, leakage-safe benchmarks, checksum provenance, deterministic chunking, and resume.


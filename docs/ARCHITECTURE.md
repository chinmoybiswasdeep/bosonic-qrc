# EOC-DV architecture

The existing Perceval static QRP remains unchanged and is the static/passive control. The new
`bosonic_qrc_dv.eoc` package has four layers:

1. `core.py`: independently indexed fixed-number and local-Fock bases, Bose–Hubbard matrices,
   exact/Floquet/Strang evolution, reductions, state distances, and channel utilities.
2. `collision.py`: a persistent mixed-state memory plus a fresh qubit-like optical ancilla at each
   input. Exact Born-weighted PNR Kraus evolution and seeded conditional trajectories are supported.
3. `chaos.py`: symmetry-resolved spectral, form-factor, OTOC, and boundary-distance analysis.
4. `tasks.py` and `study.py`: leakage-safe temporal datasets/readouts, smoke/calibration/publication
   profiles, convergence records, plots, provenance, chunking, and checksum-verified resume.

Perceval is used only for the preserved static linear-optics implementation. Persistent mixed-state
Kerr/Bose–Hubbard dynamics are implemented with SciPy dense/sparse Fock matrices because the
audited Perceval path does not expose that complete recurrent operation. The provenance field says
this explicitly.


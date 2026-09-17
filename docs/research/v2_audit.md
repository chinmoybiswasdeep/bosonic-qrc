# V2 audit

Audit started on 2026-09-17 before edits at commit
`91b33a029f2376329ab6af178dd7fc15313085fc`; `git status --short` was empty.
The tree contained a standalone distribution named `cv-mb-qrc` whose installed
top-level package was incorrectly `photographiqml`, plus committed experiment
results produced before this migration.

The baseline interpreter is Python 3.14. The baseline commands `python -m
pytest -q`, `python -m ruff check src tests experiments`, and `python -m mypy`
all failed because pytest, Ruff, and mypy are not installed. Import inspection
also failed because PhotoGraphiQ is not installed. Thus no claimed baseline or
post-change scientific execution exists in this environment.

Architecture inspected: `CVMBReservoir` is a persistent Gaussian PhotoGraphiQ
channel; its old features directly exposed simulated means/covariances.
`GraphixMBReservoir` is an optional one-memory-qubit collision model. Ridge,
ESN/RFF/delay controls, temporal split helpers, Fock numerical control, and a
small corrected Graphix--MentPy wire reference exist. Limitations include the
namespace collision, oracle/measurement ambiguity, redundant Tier-B features,
legacy result provenance, and unavailable runtime dependencies.

Changed files are the package relocation, imports/build configuration, CV
configuration/features/readout semantics, a NumPy Gaussian-twin module,
provenance verifier/configurations, CI, and documentation. These changes are
scientifically motivated to prevent a package collision, make oracle use
explicit, avoid exact feature duplication, and make source/result mismatch
detectable. The current worktree is necessarily dirty after these edits, so any
subsequent smoke outputs must be marked development-only and not publication
valid.

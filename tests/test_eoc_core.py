import numpy as np

from cv_mb_qrc.eoc.core import (
    bose_hubbard_parts,
    channel_spectrum,
    circular_phase_ratios,
    exact_action,
    exact_unitary,
    fixed_number_basis,
    floquet_unitary,
    infinite_temperature_otoc,
    kraus_completeness,
    number_operator,
    reduced_density_matrix,
    reflection_blocks,
    spacing_ratios,
    state_diagnostics,
    strang_unitary,
    symmetry_resolved_level_statistics,
    total_cutoff_basis,
    unitarity_error,
)


def test_fixed_number_basis_and_hermitian_number_conserving_core():
    basis = fixed_number_basis(4, 2)
    assert len(basis) == 10
    parts = bose_hubbard_parts(basis, 1.0, 0.7, boundary="open")
    assert np.allclose(parts.total, parts.total.conj().T)
    assert all(sum(state) == 2 for state in basis)


def test_exact_sparse_floquet_and_strang_convergence():
    basis = fixed_number_basis(3, 2)
    parts = bose_hubbard_parts(basis, 0.8, 0.5, np.asarray([0.1, -0.2, 0.3]))
    exact = exact_unitary(parts.total, 0.4)
    vector = np.arange(1, len(basis) + 1, dtype=complex)
    vector /= np.linalg.norm(vector)
    assert np.allclose(exact @ vector, exact_action(parts.total, vector, 0.4))
    errors = [np.linalg.norm(strang_unitary(parts, 0.4, steps) - exact) for steps in (1, 2, 4)]
    assert errors[2] < errors[1] < errors[0]
    assert unitarity_error(floquet_unitary(parts, 0.4)) < 1e-12
    linear = bose_hubbard_parts(basis, 0.8, 0.0)
    assert np.allclose(floquet_unitary(linear, 0.4), exact_unitary(linear.total, 0.4))


def test_exact_kerr_phase_and_reflection_sector_resolution():
    basis = fixed_number_basis(3, 2)
    parts = bose_hubbard_parts(basis, 0.0, 0.6)
    interval = 0.3
    unitary = exact_unitary(parts.total, interval)
    expected = np.diag(
        [np.exp(-0.5j * 0.6 * interval * sum(n * (n - 1) for n in state)) for state in basis]
    )
    assert np.allclose(unitary, expected)
    symmetric = bose_hubbard_parts(basis, 1.0, 0.6, np.asarray([0.2, 0.1, 0.2]))
    blocks = reflection_blocks(symmetric.total, basis)
    assert sum(len(block) for block in blocks.values()) == len(basis)
    report = symmetry_resolved_level_statistics(blocks, edge_fraction=0)
    assert set(report["by_sector"]) == {"reflection_even", "reflection_odd"}


def test_partial_trace_state_guards_and_kraus_channel():
    basis = total_cutoff_basis(2, 2)
    index = {state: i for i, state in enumerate(basis)}
    vector = np.zeros(len(basis), complex)
    vector[index[(1, 0)]] = vector[index[(0, 1)]] = 1 / np.sqrt(2)
    density = np.outer(vector, vector.conj())
    reduced, reduced_basis = reduced_density_matrix(density, basis, (0,))
    assert reduced_basis == ((0,), (1,))
    assert np.allclose(reduced, np.eye(2) / 2)
    diagnostics = state_diagnostics(density, basis, cutoff=2)
    assert np.isclose(diagnostics["trace"], 1)
    assert np.isclose(diagnostics["boundary_shell_population"], 1)

    probability = 0.2
    k0 = np.diag([1, np.sqrt(1 - probability)])
    k1 = np.asarray([[0, np.sqrt(probability)], [0, 0]])
    kraus = (k0, k1)
    assert kraus_completeness(kraus) < 1e-12
    assert channel_spectrum(kraus)["eigenvalue_mixing_gap"] > 0


def test_chaos_diagnostic_shapes_and_zero_time_otoc():
    basis = fixed_number_basis(3, 2)
    parts = bose_hubbard_parts(basis, 1.0, 0.7, np.asarray([0.1, -0.2, 0.3]))
    values = np.linalg.eigvalsh(parts.total)
    ratios = spacing_ratios(values, edge_fraction=0)
    assert np.all((ratios >= 0) & (ratios <= 1))
    unitary = exact_unitary(parts.total, 0.2)
    assert len(circular_phase_ratios(unitary)) == len(basis)
    number = number_operator(basis, 0)
    assert np.isclose(infinite_temperature_otoc(np.eye(len(basis)), number, number), 0)

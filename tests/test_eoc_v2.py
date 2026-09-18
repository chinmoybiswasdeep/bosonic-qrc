import hashlib
import json

import numpy as np

from bosonic_qrc_dv.eoc.channel import complete_channel_diagnostics
from bosonic_qrc_dv.eoc.core import fixed_number_basis
from bosonic_qrc_dv.eoc.floquet_reservoir import DVFloquetReservoir
from bosonic_qrc_dv.eoc.performance import TaskProtocol, run_task_suite
from bosonic_qrc_dv.eoc.science import (
    floquet_parts,
    floquet_unitary,
    quasienergy_statistics,
    sff_ensemble,
)
from bosonic_qrc_dv.eoc.workflow import _coordinate, _verify_artifacts


def test_fixed_number_floquet_operator_physics_and_degeneracy_policy():
    basis = fixed_number_basis(4, 2)
    assert {sum(state) for state in basis} == {2}
    disorder = np.array([0.11, -0.07, 0.03, -0.07])
    phases = np.array([0.05, -0.09, 0.12, -0.08])
    hopping, interaction = floquet_parts(basis, 1.0, 1.2, disorder, phases, "open")
    assert np.allclose(hopping, hopping.conj().T)
    assert np.allclose(interaction, interaction.conj().T)
    unitary = floquet_unitary(basis, 1.0, 1.2, disorder, phases, 0.37, 0.23)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(len(basis)))
    unresolved = quasienergy_statistics(np.eye(5), 1e-9)
    assert not unresolved["valid"]
    assert unresolved["degeneracy_fraction"] > 0
    assert unresolved["ratios"].size == 0


def test_sff_ensembles_stay_at_one_parameter_point():
    first = [np.diag(np.exp(1j * np.array([0.1, 1.2, 2.8]))) for _ in range(3)]
    second = [np.diag(np.exp(1j * np.array([0.2, 1.7, 5.1]))) for _ in range(2)]
    left = sff_ensemble(first, 4, np.random.default_rng(4), 20, "COE")
    right = sff_ensemble(second, 4, np.random.default_rng(4), 20, "COE")
    assert left["ensemble_size"] == 3
    assert right["ensemble_size"] == 2
    assert left["raw"] != right["raw"]
    assert left["thouless_time"] is None


def test_complete_four_mode_channel_includes_loss_and_preserves_states():
    model = DVFloquetReservoir()
    collision = model.collision_kraus(0.3)
    completeness = sum(operator.conj().T @ operator for operator in collision)
    assert np.allclose(completeness, np.eye(len(model.basis)), atol=1e-10)
    row = model.step(0.3)
    np.testing.assert_allclose(row["diagnostics"]["raw_output_trace"], 1.0, atol=1e-10)
    assert np.allclose(model.state, model.state.conj().T)
    assert np.linalg.eigvalsh(model.state).min() > -1e-10
    diagnostics = model.channel_diagnostics(0.3)
    assert diagnostics["kraus_count"] == 3 * 3**4
    assert diagnostics["validation_passed"]
    assert diagnostics["fixed_point_residual"] < 1e-8
    assert diagnostics["subleading_eigenvalue_ritz_residual"] < 1e-3
    assert diagnostics["traceless_hilbert_schmidt_contraction"] >= 0


def test_dense_fixed_point_and_traceless_analysis():
    eta = 0.8
    kraus = (
        np.array([[1, 0], [0, np.sqrt(eta)]], complex),
        np.array([[0, np.sqrt(1 - eta)], [0, 0]], complex),
    )
    diagnostics = complete_channel_diagnostics(kraus)
    assert diagnostics["validation_passed"]
    assert diagnostics["fixed_point_residual"] < 1e-10
    assert diagnostics["fixed_point_minimum_eigenvalue"] > -1e-10
    assert np.isclose(diagnostics["subleading_eigenvalue_modulus"], np.sqrt(eta))


def test_real_task_runner_exercises_every_advertised_family():
    protocol = TaskProtocol(
        train=5,
        validation=4,
        test=4,
        washout=1,
        embargo=1,
        maximum_delay=1,
        maximum_degree=2,
        alphas=(1e-4, 1e-2),
    )
    result = run_task_suite(DVFloquetReservoir, 211, protocol, "dv")
    assert len(result["advertised_tasks_exercised"]) == 7
    assert "narma10" in result["tasks"]
    assert "mackey_glass" in result["tasks"]
    assert "nonlinear_channel_equalization" in result["tasks"]


def test_coordinate_normalization_and_corruption_safe_resume(tmp_path):
    edge = {"estimate": 1.0, "ci95": [0.5, 1.5]}
    coordinate = _coordinate(edge, [0.0, 1.0, 2.0], [0.0, 0.25, 0.5])
    assert coordinate == {0.0: -2.0, 1.0: 0.0, 2.0: 2.0}
    artifact = tmp_path / "data.json"
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {"artifact_checksums": {"data.json": digest}}
    _verify_artifacts(tmp_path, manifest)
    artifact.write_text("corrupt", encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "corrupt"):
        _verify_artifacts(tmp_path, manifest)

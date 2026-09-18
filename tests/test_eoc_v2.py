import hashlib
import json

import numpy as np

from cv_mb_qrc.eoc.channel import complete_channel_diagnostics
from cv_mb_qrc.eoc.core import fixed_number_basis
from cv_mb_qrc.eoc.floquet_reservoir import (
    MBFloquetConfig,
    MBFloquetReservoir,
    _annihilation,
)
from cv_mb_qrc.eoc.performance import TaskProtocol, run_task_suite
from cv_mb_qrc.eoc.science import (
    floquet_parts,
    floquet_unitary,
    quasienergy_statistics,
    sff_ensemble,
)
from cv_mb_qrc.eoc.workflow import _coordinate, _verify_artifacts


def test_bosonic_ladder_and_fixed_number_floquet_physics():
    ladder = _annihilation(((0,), (1,), (2,)), 0)
    commutator = ladder @ ladder.conj().T - ladder.conj().T @ ladder
    assert np.allclose(np.diag(commutator), [1, 1, -2])
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


def test_complete_channel_and_manual_adapter_provenance():
    model = MBFloquetReservoir()
    collision = model.collision_kraus(0.3)
    completeness = sum(operator.conj().T @ operator for operator in collision)
    assert np.allclose(completeness, np.eye(len(model.basis)), atol=1e-10)
    row = model.step(0.3)
    np.testing.assert_allclose(row["diagnostics"]["raw_output_trace"], 1.0, atol=1e-10)
    assert "manual total-Fock" in row["provenance"]["backend"]
    assert np.linalg.eigvalsh(model.state).min() > -1e-10
    diagnostics = model.channel_diagnostics(0.3)
    assert diagnostics["kraus_count"] == 4 * 4**4
    assert diagnostics["validation_passed"]
    assert diagnostics["fixed_point_residual"] < 1e-8
    assert diagnostics["subleading_eigenvalue_ritz_residual"] < 1e-3
    assert model.non_gaussian_witness()["changed"]


def test_conditional_trajectory_is_seed_reproducible():
    config = MBFloquetConfig(evolution="conditional", seed=91)
    first = MBFloquetReservoir(config)
    second = MBFloquetReservoir(config)
    outcomes_a = [first.step(value)["outcome"] for value in (0.1, -0.2, 0.3)]
    outcomes_b = [second.step(value)["outcome"] for value in (0.1, -0.2, 0.3)]
    assert outcomes_a == outcomes_b
    assert np.allclose(first.state, second.state)


def test_dense_fixed_point_and_traceless_analysis():
    eta = 0.8
    kraus = (
        np.array([[1, 0], [0, np.sqrt(eta)]], complex),
        np.array([[0, np.sqrt(1 - eta)], [0, 0]], complex),
    )
    diagnostics = complete_channel_diagnostics(kraus)
    assert diagnostics["validation_passed"]
    assert diagnostics["fixed_point_residual"] < 1e-10
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
    result = run_task_suite(MBFloquetReservoir, 211, protocol, "cv")
    assert len(result["advertised_tasks_exercised"]) == 7
    assert {"narma10", "mackey_glass", "nonlinear_channel_equalization"} <= set(result["tasks"])


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

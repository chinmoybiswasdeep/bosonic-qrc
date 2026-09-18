import numpy as np
import pytest

from bosonic_qrc_cv.eoc.fock_loop import (
    FockLoopConfig,
    NonGaussianCVReservoir,
    piquasso_kerr_phases,
)


@pytest.mark.parametrize("resource", ["coherent", "squeezed", "cat", "number", "fock"])
def test_resource_families_are_physical(resource):
    config = FockLoopConfig(
        memory_modes=1,
        cutoff=4,
        disorder=(0.0, 0.0),
        resource=resource,
        number_state=1,
    )
    result = NonGaussianCVReservoir(config).step(0.3)
    diagnostics = result["diagnostics"]
    assert abs(diagnostics["trace"] - 1) < 1e-11
    assert diagnostics["minimum_eigenvalue"] > -1e-11
    assert 0 < diagnostics["retained_input_norm"] <= 1
    assert np.all(np.isfinite(result["features"]))


def test_persistent_state_causal_dependence_and_reset():
    config = FockLoopConfig(memory_modes=1, cutoff=4, disorder=(0.0, 0.0))
    left, right = NonGaussianCVReservoir(config), NonGaussianCVReservoir(config)
    left.step(-0.7)
    right.step(0.7)
    assert not np.allclose(left.step(0.2)["features"], right.step(0.2)["features"])
    left.reset()
    reference = NonGaussianCVReservoir(config)
    assert np.allclose(left.step(0.2)["features"], reference.step(0.2)["features"])


def test_piquasso_native_kerr_matches_bose_hubbard_onsite_phase():
    cutoff, interaction, interval = 5, 0.7, 0.3
    actual = piquasso_kerr_phases(cutoff, interaction, interval)
    photon = np.arange(cutoff)
    expected = np.exp(-0.5j * interaction * interval * photon * (photon - 1))
    assert np.allclose(actual, expected)


def test_cutoff_comparison_reports_full_state_distance():
    low = NonGaussianCVReservoir(
        FockLoopConfig(memory_modes=1, cutoff=3, disorder=(0.0, 0.0))
    )
    high = NonGaussianCVReservoir(
        FockLoopConfig(memory_modes=1, cutoff=4, disorder=(0.0, 0.0))
    )
    for value in (-0.2, 0.3):
        low.step(value)
        high.step(value)
    report = low.compare_state(high)
    assert 0 <= report["fidelity"] <= 1 + 1e-8
    assert 0 <= report["trace_distance"] <= 1 + 1e-8


def test_backend_and_dynamics_are_explicitly_separate():
    config = FockLoopConfig(
        memory_modes=1,
        cutoff=3,
        disorder=(0.0, 0.0),
        evolution_model="floquet",
    )
    model = NonGaussianCVReservoir(config)
    assert model.provenance()["gaussian_baseline_unchanged"]
    assert model.provenance()["cutoff_semantics"] == "exclusive total photon number"

import numpy as np
import piquasso as pq
import pytest

from bosonic_qrc_cv import CVConfig, GaussianLoopReservoir


def test_backend_execution_and_feature_dimension(monkeypatch):
    called = {"count": 0}
    original = pq.GaussianSimulator.execute

    def wrapped(self, program, *args, **kwargs):
        called["count"] += 1
        return original(self, program, *args, **kwargs)

    monkeypatch.setattr(pq.GaussianSimulator, "execute", wrapped)
    record = GaussianLoopReservoir(CVConfig(modes=3)).step(0.2)
    assert called["count"] == 1
    assert record.feature.shape == (6,)


def test_covariances_symmetric_and_physical():
    covariance = GaussianLoopReservoir(CVConfig(modes=2)).step(0.3).loop_covariance
    omega = 2 * np.kron(np.eye(2), [[0, 1], [-1, 0]])
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance + 1j * omega).min() > -1e-9


def test_beamsplitter_limits_identify_loop_and_detector_arms():
    common = {"modes": 1, "input_squeezing": 0, "active_squeezing": 0, "loss": 0}
    retained = GaussianLoopReservoir(CVConfig(loop_reflectivity=1, **common))
    retained.reset(covariance_scale=3)
    old_loop = retained.step(0)
    exchanged = GaussianLoopReservoir(CVConfig(loop_reflectivity=0, **common))
    exchanged.reset(covariance_scale=3)
    new_loop = exchanged.step(0)
    assert np.allclose(old_loop.loop_covariance, 6 * np.eye(2))
    assert np.allclose(old_loop.detector_covariance, [[2]])
    assert np.allclose(new_loop.loop_covariance, 2 * np.eye(2))
    assert np.allclose(new_loop.detector_covariance, [[6]])


def test_history_and_expected_fading_law():
    reflectivity = 0.6
    config = CVConfig(
        modes=1,
        loop_reflectivity=reflectivity,
        active_squeezing=0,
        loss=0,
        reservoir_seed=2,
    )
    impulse, control = GaussianLoopReservoir(config), GaussianLoopReservoir(config)
    impulse.loop_unitary[:] = 1
    impulse.detector_unitary[:] = 1
    control.loop_unitary[:] = 1
    control.detector_unitary[:] = 1
    observed = []
    for index in range(6):
        value = 1.0 if index == 0 else 0.0
        observed.append(abs(impulse.step(value).feature[0] - control.step(0.0).feature[0]))
    observed = np.asarray(observed[1:])
    assert np.allclose(observed[1:] / observed[:-1], reflectivity, rtol=1e-5, atol=1e-8)


def test_echo_state_convergence_and_bounded_energy():
    config = CVConfig(modes=2, loop_reflectivity=0.4, active_squeezing=0, loss=0.02)
    first, second = GaussianLoopReservoir(config), GaussianLoopReservoir(config)
    first.reset(1.0)
    second.reset(3.0)
    distances = []
    for value in np.linspace(-0.8, 0.8, 12):
        a, b = first.step(value), second.step(value)
        distances.append(np.linalg.norm(a.loop_covariance - b.loop_covariance))
        assert a.maximum_covariance_eigenvalue < config.stability_covariance_limit
    assert distances[-1] < 0.01 * distances[0]


@pytest.mark.parametrize("modes", [1, 2, 3])
def test_finite_shot_columns_and_convergence(modes):
    base = {"modes": modes, "active_squeezing": 0, "loss": 0, "reservoir_seed": 11}
    exact = GaussianLoopReservoir(CVConfig(measurement="exact", **base)).step(0.2).feature
    sampled = GaussianLoopReservoir(
        CVConfig(measurement="finite_shot", shots=3000, **base)
    ).step(0.2)
    assert sampled.measurement_sample_shape == (3000, 2 * modes)
    assert sampled.detector_covariance.shape == (modes, modes)
    assert sampled.feature.shape == (modes * (modes + 1) // 2,)
    assert np.allclose(sampled.feature, exact, rtol=0.25, atol=0.25)

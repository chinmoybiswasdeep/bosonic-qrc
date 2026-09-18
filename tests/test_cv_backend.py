import numpy as np
import pytest

from bosonic_qrc_cv import CVConfig, GaussianLoopReservoir


def test_backend_execution_is_called(monkeypatch):
    pq = pytest.importorskip("piquasso")
    called = {"value": False}
    original = pq.GaussianSimulator.execute

    def wrapped(self, program, *args, **kwargs):
        called["value"] = True
        return original(self, program, *args, **kwargs)

    monkeypatch.setattr(pq.GaussianSimulator, "execute", wrapped)
    reservoir = GaussianLoopReservoir(CVConfig(modes=2, active_squeezing=0.0))
    feature = reservoir.step(0.2).feature
    assert called["value"] and reservoir.execution_count == 1
    assert feature.shape == (3,)


def test_history_changes_feature():
    pytest.importorskip("piquasso")
    config = CVConfig(modes=2, active_squeezing=0.0, seed=3)
    first = GaussianLoopReservoir(config)
    first.step(-0.8)
    a = first.step(0.1).feature
    second = GaussianLoopReservoir(config)
    second.step(0.8)
    b = second.step(0.1).feature
    assert not np.allclose(a, b)


def test_covariance_is_symmetric():
    pytest.importorskip("piquasso")
    record = GaussianLoopReservoir(CVConfig(modes=2)).step(0.3)
    assert np.allclose(record.covariance, record.covariance.T)

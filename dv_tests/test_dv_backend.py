import numpy as np
import pytest

from bosonic_qrc_dv import LinearOpticalReservoir


def test_perceval_execution_is_called(monkeypatch):
    pcvl = pytest.importorskip("perceval")
    called = {"value": False}
    original = pcvl.algorithm.Sampler.probs

    def wrapped(self, *args, **kwargs):
        called["value"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pcvl.algorithm.Sampler, "probs", wrapped)
    features = LinearOpticalReservoir().probabilities()
    assert called["value"]
    assert np.isclose(features.values.sum(), 1.0)


def test_hom_interference_at_balanced_beamsplitter():
    features = LinearOpticalReservoir(theta=np.pi / 4).probabilities((1, 1))
    result = dict(zip(features.outcomes, features.values))
    assert result.get("|1,1>", 0.0) < 1e-10
    assert np.isclose(result["|2,0>"], 0.5)
    assert np.isclose(result["|0,2>"], 0.5)

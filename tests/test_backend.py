import numpy as np
import perceval as pcvl

from bosonic_qrc_dv import DVConfig, LinearOpticalReservoir
from bosonic_qrc_dv.perturbation import perturb_unitary


def test_backend_execution_is_called(monkeypatch):
    called = {"value": False}
    original = pcvl.algorithm.Sampler._create_job

    def wrapped(self, *args, **kwargs):
        called["value"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pcvl.algorithm.Sampler, "_create_job", wrapped)
    features = LinearOpticalReservoir().probabilities()
    assert called["value"]
    assert np.isclose(features.values.sum(), 1)


def test_four_mode_two_photon_index_has_fifteen_sectors():
    features = LinearOpticalReservoir(DVConfig(modes=4, photons=2)).probabilities()
    assert features.values.shape == (15,)
    assert features.sector_weights[0] == 0
    assert features.sector_weights[1] == 0
    assert np.isclose(features.sector_weights[2], 1)
    assert "|0,0,0,2>" in features.outcomes


def test_dual_rail_encoding_is_sample_dependent():
    reservoir = LinearOpticalReservoir(DVConfig(modes=4, photons=2, seed=4))
    first = reservoir.probabilities(state=(1, 0, 1, 0), coordinates=(0.1, -0.2)).values
    second = reservoir.probabilities(state=(1, 0, 1, 0), coordinates=(1.2, 0.7)).values
    assert not np.allclose(first, second)


def test_unitary_perturbation_remains_unitary():
    reservoir = LinearOpticalReservoir()
    perturbed = perturb_unitary(reservoir.unitary, strength=0.2, seed=9)
    assert np.allclose(perturbed.conj().T @ perturbed, np.eye(4), atol=1e-10)

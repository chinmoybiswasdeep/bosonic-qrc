from pathlib import Path

import numpy as np
import perceval as pcvl
import pytest
import yaml

from bosonic_qrc_dv import DVConfig, LinearOpticalReservoir
from bosonic_qrc_dv.perturbation import perturb_unitary
from bosonic_qrc_dv.temporal_qrc.cli import main as temporal_main


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
    reservoir = LinearOpticalReservoir(DVConfig(modes=4, photons=2, reservoir_seed=4))
    first = reservoir.probabilities(state=(1, 0, 1, 0), coordinates=(0.1, -0.2)).values
    second = reservoir.probabilities(state=(1, 0, 1, 0), coordinates=(1.2, 0.7)).values
    assert not np.allclose(first, second)


def test_unitary_perturbation_remains_unitary():
    reservoir = LinearOpticalReservoir()
    perturbed = perturb_unitary(reservoir.unitary, strength=0.2, seed=9)
    assert np.allclose(perturbed.conj().T @ perturbed, np.eye(4), atol=1e-10)


def test_finite_shot_features_are_perceval_frequencies():
    config = DVConfig(modes=4, photons=2, measurement="finite_shot", shots=2000)
    features = LinearOpticalReservoir(config).probabilities()
    assert features.values.shape == (15,)
    assert np.isclose(features.values.sum(), 1)
    assert np.allclose(features.values * config.shots, np.round(features.values * config.shots))


def test_identity_direct_pnr_control():
    config = DVConfig(modes=4, photons=2)
    features = LinearOpticalReservoir(config, unitary=np.eye(4)).probabilities(state=(1, 1, 0, 0))
    result = dict(zip(features.outcomes, features.values))
    assert np.isclose(result["|1,1,0,0>"], 1)


@pytest.mark.parametrize("profile", ["smoke", "calibration", "full"])
def test_every_committed_configuration_executes(profile):
    raw = yaml.safe_load(Path(f"configs/{profile}/xor.yaml").read_text(encoding="utf-8"))
    reservoir = LinearOpticalReservoir(DVConfig(**raw["reservoir"]))
    features = reservoir.probabilities(state=reservoir.dual_rail_input(), coordinates=(0.1, -0.2))
    modes = reservoir.config.modes
    assert len(features.values) == (modes + 1) * (modes + 2) // 2
    assert np.count_nonzero(features.values > 1e-14) <= modes * (modes + 1) // 2
    assert features.sector_weights[0] == 0
    assert features.sector_weights[1] == 0


def test_temporal_ipc_refuses_without_recurrent_channel():
    with pytest.raises(RuntimeError, match="no validated recurrent"):
        temporal_main()

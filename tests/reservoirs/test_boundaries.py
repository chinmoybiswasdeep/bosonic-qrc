import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from photographiqml.reservoirs import CVConfig, CVMBReservoir, QubitConfig, RidgeReadout
from photographiqml.reservoirs.diagnostics import contraction, trace_distance
from photographiqml.reservoirs.results import atomic_json


def test_atomic_retry_preserves_json(tmp_path, monkeypatch):
    import photographiqml.reservoirs.results as results

    original = results.os.replace
    attempts = []

    def replace_file(source, destination):
        attempts.append(source)
        if len(attempts) == 1:
            raise PermissionError("transient reader")
        original(source, destination)

    monkeypatch.setattr(results.os, "replace", replace_file)
    monkeypatch.setattr(results.time, "sleep", lambda _: None)
    path = tmp_path / "record.json"
    atomic_json(path, {"score": 1})
    assert json.loads(path.read_text()) == {"score": 1}
    assert len(attempts) == 2 and not list(tmp_path.glob("*.tmp"))


def test_readout_invalid_and_optional_clean_import():
    with pytest.raises(ValueError):
        RidgeReadout(-1)
    model = RidgeReadout()
    with pytest.raises(ValueError):
        model.predict([0], [[1]])
    with pytest.raises(ValueError):
        model.fit([0, 1], [[0], [1]], [np.nan, 1])
    model.fit([0, 1], [[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([0], [[1, 2]])
    code = "import sys; import photographiqml.reservoirs; assert 'graphix' not in sys.modules; assert 'mentpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_phase_squeeze_encoding_and_trajectory_limits():
    c = CVConfig(memory_modes=1, phase_scale=0.5, squeeze_scale=0.1)
    x = CVMBReservoir(c).step(0.3).features
    assert np.isfinite(x).all()
    assert not np.allclose(
        x, CVMBReservoir(replace(c, phase_scale=0, squeeze_scale=0)).step(0.3).features
    )
    with pytest.raises(MemoryError):
        CVMBReservoir(CVConfig(max_trajectories=2)).step(0, shots=3)
    with pytest.raises(ValueError):
        QubitConfig(input_channels=True)


def test_zero_probability_branch_and_trace_distance():
    pytest.importorskip("graphix")
    from photographiqml.reservoirs.graphix_backend import GraphixMBReservoir, graphix_wire

    c = QubitConfig(entangle=False, angle=0)
    model = GraphixMBReservoir(c)
    u = (np.pi / 2 - model.bias) / model.mask
    result = model.step(u)
    assert min(result.diagnostics["branch_probabilities"]) < 1e-12
    assert np.isfinite(result.features).all()
    a, b = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    assert trace_distance(a, b) == 1
    curves = contraction(GraphixMBReservoir, [0.1, 0.2, 0.3], [a, b])
    assert len(curves["trace_distances"]) == 3
    with pytest.raises(MemoryError):
        graphix_wire([1, 0], np.zeros(17), adaptive=True)

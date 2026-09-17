import numpy as np
import pytest

from cv_mb_qrc.reservoirs.benchmarks import ClassicalFeatures, mackey_glass, metrics, narma10
from cv_mb_qrc.reservoirs.diagnostics import bootstrap_summary
from cv_mb_qrc.reservoirs.fock import FockConfig, FockMBReservoir, cutoff_study
from cv_mb_qrc.reservoirs.temporal import chronological_splits


def test_tasks_and_exact_delay_control():
    u = np.random.default_rng(12).uniform(0, 0.5, 200)
    y = narma10(u)
    assert y.shape == u.shape and np.isfinite(y).all()
    assert y[9] == pytest.approx(1.5 * u[0] * u[9] + 0.1)
    mg = mackey_glass(100)
    assert mg.shape == (100,) and np.std(mg) > 0.01
    for name in ("delay", "esn", "rff", "input_only", "persistence"):
        model = ClassicalFeatures(name, dimension=4, window=4)
        x = model.transform(u)
        np.testing.assert_array_equal(x, model.transform(u))
        changed = u.copy()
        changed[50:] += 1
        np.testing.assert_array_equal(x[:50], model.transform(changed)[:50])
    x = ClassicalFeatures("delay", window=4).transform(u)
    assert metrics(u[:-3], x[3:, 3])["r2"] == 1
    summary = bootstrap_summary([1, 2, 3])
    assert summary["mean"] == 2 and summary["values"] == [1, 2, 3]
    for value in ([-1], [np.nan]):
        with pytest.raises(ValueError):
            narma10(value)
    with pytest.raises(ValueError):
        chronological_splits(4, gap=4)
    with pytest.raises(ValueError):
        metrics([1, 1], [1, 1])


def test_fock_invariants_and_convergence():
    result = FockMBReservoir().run_sequence([0.02, 0.04])
    assert result.features.shape == (2, 4)
    for d in result.diagnostics["step_diagnostics"]:
        assert d["trace"] == pytest.approx(1, abs=1e-12)
        assert d["minimum_eigenvalue"] >= -1e-12
        assert d["hermiticity_error"] < 1e-12
        assert 0 <= d["herald_probability"] <= 1
    study = cutoff_study([0.02], cutoffs=(8, 12))
    assert study["observable_convergence"]
    assert study["rows"][-1]["failure_probability"] > 0
    with pytest.raises(MemoryError):
        FockConfig(cutoff=1000)

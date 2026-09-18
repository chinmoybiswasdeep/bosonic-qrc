import numpy as np

from bosonic_qrc_cv.benchmarks import (
    classical_gaussian_features,
    esn_features,
    mackey_glass,
    narma10,
    nonlinear_channel,
    nvar_features,
    random_nonlinear_features,
    regression_metrics,
    tapped_delay_features,
)


def test_benchmarks_are_deterministic_and_finite():
    u1, y1 = narma10(100, 7)
    u2, y2 = narma10(100, 7)
    assert np.array_equal(u1, u2)
    assert np.array_equal(y1, y2)
    channel, symbols = nonlinear_channel(100, 8)
    assert set(np.unique(symbols)) <= {-3, -1, 1, 3}
    assert np.all(np.isfinite(channel))
    assert np.array_equal(mackey_glass(100, 9), mackey_glass(100, 9))


def test_control_feature_dimensions_and_metrics():
    values = np.linspace(-1, 1, 40)
    assert tapped_delay_features(values, 4).shape == (40, 4)
    assert random_nonlinear_features(values, 6, 1).shape == (40, 6)
    assert esn_features(values, 6, 1).shape == (40, 6)
    assert nvar_features(values, 3, 2).shape == (40, 10)
    assert classical_gaussian_features(values, 6, 1).shape == (40, 6)
    assert regression_metrics(values, values)["nmse"] == 0


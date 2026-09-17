import numpy as np
import pytest

from cv_mb_qrc.reservoirs.benchmarks import null_corrected_capacity
from cv_mb_qrc.reservoirs.diagnostics import feature_diagnostics
from cv_mb_qrc.reservoirs.readout import RidgeReadout


def test_train_only_collapsed_filter_and_rank_diagnostics():
    inputs = np.arange(4.0)
    features = np.c_[inputs, np.ones(4)]
    readout = RidgeReadout(variance_floor=1e-10).fit(inputs, features, inputs)
    assert readout.removed_columns == [2]  # duplicate is valid; constant column is removed
    report = feature_diagnostics(features)
    assert report["exact_rank"] == 1 and report["variance_collapsed_columns"] == 1


def test_null_capacity_reports_bias_correction():
    report = null_corrected_capacity([0.2, -0.1], [[0.1, 0.1], [0.1, 0.1]])
    assert report["raw_total"] == pytest.approx(0.1)
    assert report["null_corrected_total"] == pytest.approx(-0.1)

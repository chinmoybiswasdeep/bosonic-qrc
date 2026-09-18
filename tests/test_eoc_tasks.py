import numpy as np

from cv_mb_qrc.eoc.tasks import (
    SplitPolicy,
    chronological_indices,
    delayed_parity,
    effective_feature_rank,
    esn_features,
    fit_evaluate,
    ipc_targets,
    mackey_glass,
    narma10,
    nonlinear_channel,
    nvar_features,
    rff_features,
    tapped_delay_features,
)


def test_tasks_and_controls_are_deterministic():
    for function in (narma10, delayed_parity, nonlinear_channel, mackey_glass):
        first = function(200, 8)
        second = function(200, 8)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
    values = np.linspace(-1, 1, 50)
    assert tapped_delay_features(values, 3).shape == (50, 3)
    assert rff_features(values, 5, 2).shape == (50, 5)
    assert esn_features(values, 5, 2).shape == (50, 5)
    assert nvar_features(values, 3).shape == (50, 10)


def test_chronological_split_embargo_and_train_only_preprocessing():
    policy = SplitPolicy(train=40, validation=20, test=30, embargo=5)
    train, validation, test = chronological_indices(100, policy)
    assert train[-1] + policy.embargo < validation[0]
    assert validation[-1] + policy.embargo < test[0]
    rng = np.random.default_rng(3)
    features = rng.normal(size=(100, 5))
    target = features[:, 0] + 0.01 * rng.normal(size=100)
    report = fit_evaluate(features, target, policy)
    assert report["preprocessing_fit_indices"] == report["train_indices"]
    assert not set(report["train_indices"]) & set(report["test_indices"])
    assert report["test_r2"] > 0.99


def test_ipc_targets_and_effective_rank_are_labelled_and_deterministic():
    values = np.linspace(0, 1, 40)
    targets = ipc_targets(values, maximum_delay=3, maximum_degree=2)
    assert "degree=1;delays=1" in targets
    assert "degree=2;delays=1,2" in targets
    assert all(len(target) == len(values) for target in targets.values())
    assert effective_feature_rank(np.column_stack([values, 2 * values])) == 1

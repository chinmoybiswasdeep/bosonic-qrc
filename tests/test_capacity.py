import numpy as np

from bosonic_qrc_dv.capacity import (
    benjamini_hochberg,
    capacity_bound,
    conventional_capacity,
    enumerate_targets,
    evaluate_target,
    fit_pseudoinverse,
    independent_null_capacities,
    max_statistic_threshold,
    minimum_null_surrogates,
    normalized_legendre,
    rank_diagnostics,
    squared_correlation,
    target_bank_diagnostics,
)
from bosonic_qrc_dv.capacity import test_r2 as coefficient_of_determination


def test_normalized_legendre_orthogonality():
    rng = np.random.default_rng(4)
    values = rng.uniform(-1, 1, 100_000)
    gram = np.asarray(
        [
            [
                np.mean(normalized_legendre(i, values) * normalized_legendre(j, values))
                for j in range(4)
            ]
            for i in range(4)
        ]
    )
    assert np.allclose(gram, np.eye(4), atol=0.015)


def test_multi_indices_are_deterministic_and_nonconstant():
    first = enumerate_targets(3, 4, 2)
    assert first == enumerate_targets(3, 4, 2)
    assert all(item.degree == sum(item.multi_index) > 0 for item in first)
    assert all(item.interaction_order <= 2 for item in first)


def test_perfect_linear_delay_and_shuffled_control():
    rng = np.random.default_rng(8)
    inputs = rng.uniform(-1, 1, 600)
    definition = next(item for item in enumerate_targets(1, 2, 1) if item.maximum_delay == 2)
    target = evaluate_target(inputs, definition)
    features = target[:, None]
    fit = fit_pseudoinverse(features[:300], target[:300])
    prediction = fit.predict(features[300:])
    assert squared_correlation(target[300:], prediction) > 0.999999
    assert coefficient_of_determination(target[300:], prediction) > 0.999999
    assert squared_correlation(rng.permutation(target[300:]), prediction) < 0.05


def test_rank_bound_and_fdr():
    assert capacity_bound(1.9, 2)[0]
    assert not capacity_bound(2.1, 2)[0]
    assert benjamini_hochberg(np.asarray([0.001, 0.01, 0.5])).tolist() == [True, True, False]
    near_constant = np.ones((30, 3)) + 1e-14 * np.arange(30)[:, None]
    assert fit_pseudoinverse(near_constant, np.arange(30)).numerical_rank == 0


def test_complete_basis_and_capacity_definition():
    targets = enumerate_targets(2, 4, 2)
    assert len(targets) == 20
    truth = np.asarray([0.0, 1.0, 2.0])
    assert conventional_capacity(truth, np.zeros(3)) == 0.0
    assert conventional_capacity(truth, truth) == 1.0
    assert minimum_null_surrogates(len(targets), 0.05) == 399


def test_rank_target_and_independent_null_diagnostics():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(80, 4))
    diagnostics = rank_diagnostics(features)
    assert diagnostics.numerical_rank == 4
    assert 1 <= diagnostics.stable_rank <= 4
    definition = enumerate_targets(1, 0, 1)[0]
    targets = np.asarray([evaluate_target(rng.uniform(-1, 1, 80), definition)])
    assert target_bank_diagnostics(targets)["effective_independent_target_count"] > 0.99
    null, ridge = independent_null_capacities(
        features[:40], features[40:60], features[60:], definition, 7, 12, 1e-10, (1e-6, 1e-3)
    )
    assert null.shape == ridge.shape == (7,)
    assert np.all(null >= 0)
    assert max_statistic_threshold(np.vstack([null, ridge])) >= 0

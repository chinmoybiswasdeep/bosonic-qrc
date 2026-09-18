import numpy as np

from bosonic_qrc_dv.capacity import (
    benjamini_hochberg,
    capacity_bound,
    enumerate_targets,
    evaluate_target,
    fit_pseudoinverse,
    normalized_legendre,
    squared_correlation,
)
from bosonic_qrc_dv.capacity import test_r2 as coefficient_of_determination


def test_normalized_legendre_orthogonality():
    rng = np.random.default_rng(4)
    values = rng.uniform(-1, 1, 100_000)
    gram = np.asarray([[np.mean(normalized_legendre(i, values) * normalized_legendre(j, values)) for j in range(4)] for i in range(4)])
    assert np.allclose(gram, np.eye(4), atol=0.015)


def test_multi_indices_are_deterministic_and_nonconstant():
    first = enumerate_targets(3, 4, 2, maximum_targets=20)
    assert first == enumerate_targets(3, 4, 2, maximum_targets=20)
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

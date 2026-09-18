import numpy as np

from bosonic_qrc_dv.benchmarks import (
    boundary_fragmentation,
    concentric_rings,
    intertwined_spirals,
    stratified_split,
    xor_dataset,
)


def test_boundary_datasets_are_deterministic():
    for generator in (concentric_rings, intertwined_spirals):
        first = generator(100, 7)
        second = generator(100, 7)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
    points, labels = xor_dataset(10, 8)
    assert points.shape == (40, 2)
    assert np.bincount(labels).tolist() == [20, 20]


def test_stratified_split_and_fragmentation():
    _, labels = concentric_rings(100, 9)
    splits = stratified_split(labels, 10)
    assert sum(map(len, splits)) == 100
    assert not set(splits[0]) & set(splits[1])
    grid = np.asarray([[0, 0, 1], [0, 1, 1]])
    assert boundary_fragmentation(grid) == 3


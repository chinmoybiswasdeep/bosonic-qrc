import numpy as np

from bosonic_qrc_dv.eoc.collision import CollisionConfig, RecurrentDVReservoir


def test_kraus_completeness_cptp_and_physical_state():
    model = RecurrentDVReservoir(
        CollisionConfig(memory_modes=1, disorder=(0.0, 0.0), local_cutoff=3)
    )
    operators = model.collision_kraus(0.37)
    assert model.step(0.37)["kraus_completeness_error"] < 1e-11
    assert np.allclose(sum(k.conj().T @ k for k in operators), np.eye(3))
    diagnostics = model.step(0.62)["diagnostics"]
    assert abs(diagnostics["trace"] - 1) < 1e-12
    assert diagnostics["minimum_eigenvalue"] > -1e-12


def test_surviving_memory_causal_dependence_and_reset():
    config = CollisionConfig(
        memory_modes=1,
        disorder=(0.0, 0.0),
        local_cutoff=3,
        interaction=0.8,
        memory_transmissivity=0.95,
    )
    first = RecurrentDVReservoir(config)
    second = RecurrentDVReservoir(config)
    first.step(0.9)
    second.step(0.1)
    response_first = first.step(0.4)["features"]
    response_second = second.step(0.4)["features"]
    assert not np.allclose(response_first, response_second)
    first.reset()
    assert np.allclose(
        first.step(0.4)["features"], RecurrentDVReservoir(config).step(0.4)["features"]
    )


def test_conditional_trajectory_uses_born_weight_and_is_seeded():
    config = CollisionConfig(
        memory_modes=1,
        disorder=(0.0, 0.0),
        evolution="conditional",
        seed=91,
    )
    left, right = RecurrentDVReservoir(config), RecurrentDVReservoir(config)
    rows_left = [left.step(value) for value in (0.2, 0.8, 0.4)]
    rows_right = [right.step(value) for value in (0.2, 0.8, 0.4)]
    assert [row["outcome"] for row in rows_left] == [row["outcome"] for row in rows_right]
    assert all(np.isclose(row["outcome_probabilities"].sum(), 1) for row in rows_left)


def test_noninteracting_reduction_and_open_channel_gap():
    model = RecurrentDVReservoir(
        CollisionConfig(
            memory_modes=1,
            disorder=(0.0, 0.0),
            interaction=0.0,
            memory_transmissivity=0.8,
        )
    )
    report = model.channel_diagnostics(0.5)
    assert report["eigenvalue_mixing_gap"] >= -1e-10

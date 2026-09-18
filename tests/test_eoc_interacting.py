import numpy as np

from cv_mb_qrc.eoc.interacting import InteractingMBReservoir, MBFockConfig


def test_unconditional_branches_are_born_weighted_and_physical():
    config = MBFockConfig(memory_modes=1, cutoff=4, disorder=(0.0,))
    result = InteractingMBReservoir(config).step(0.2)
    probabilities = np.asarray(result["outcome_probabilities"])
    assert probabilities.sum() > 0.999
    diagnostics = result["diagnostics"]
    assert abs(diagnostics["trace"] - 1) < 1e-10
    assert diagnostics["minimum_eigenvalue"] > -1e-10
    assert result["outcome"] is None


def test_conditional_branch_and_recurrence_are_causal():
    config = MBFockConfig(
        memory_modes=1,
        cutoff=4,
        disorder=(0.0,),
        evolution="conditional",
        seed=21,
    )
    left = InteractingMBReservoir(config)
    right = InteractingMBReservoir(config)
    left.step(-0.5, outcome=0)
    right.step(0.5, outcome=0)
    left_features = left.step(0.1, outcome=0)["features"]
    right_features = right.step(0.1, outcome=0)["features"]
    assert not np.allclose(left_features, right_features)


def test_direct_interacting_and_resource_ablation_are_distinct():
    direct = InteractingMBReservoir(
        MBFockConfig(memory_modes=1, cutoff=4, disorder=(0.0,), resource_ablation="cat")
    )
    injected = InteractingMBReservoir(
        MBFockConfig(
            memory_modes=1,
            cutoff=4,
            disorder=(0.0,),
            resource_ablation="cat_cubic",
        )
    )
    assert not np.allclose(direct.step(0.3)["features"], injected.step(0.3)["features"])
    provenance = direct.provenance()
    assert "PhotoGraphiQ" in provenance["backend"]
    assert provenance["unconditional_policy"].startswith("Born-probability")

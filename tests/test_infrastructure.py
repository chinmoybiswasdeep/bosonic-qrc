import json

import pytest
import yaml

from bosonic_qrc_dv.aggregate import aggregate
from bosonic_qrc_dv.infrastructure import ResultJournal, RunKey, canonical_hash


def test_canonical_hash_and_exact_resume(tmp_path):
    config = {"b": 2, "a": 1}
    assert canonical_hash(config) == canonical_hash({"a": 1, "b": 2})
    journal = ResultJournal(tmp_path, config)
    key = RunKey(1, 2, 3, "qrc")
    journal.append(key, {"metric": 0.5})
    assert ResultJournal(tmp_path, config).has(key)
    with pytest.raises(ValueError, match="duplicate"):
        journal.append(key, {})
    with pytest.raises(ValueError, match="config hash"):
        ResultJournal(tmp_path, {"a": 2})
    assert json.loads((tmp_path / "runs.jsonl").read_text())["payload"]["metric"] == 0.5


def test_shard_merge_is_complete_and_deterministic(tmp_path):
    config = {
        "reservoir": {"measurement_seed": 3},
        "models": ["reservoir", "identity"],
        "seeds": {"reservoir": [1], "data": [4], "measurement": [3]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    shards = [tmp_path / "s0", tmp_path / "s1"]
    for model, shard in zip(config["models"], shards):
        journal = ResultJournal(shard, config)
        journal.append(RunKey(1, 4, 3, model), {"rows": []})
    assert aggregate(config_path, tmp_path / "merged", shards) == 2
    lines = (tmp_path / "merged" / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 2

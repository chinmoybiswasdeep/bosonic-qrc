import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from cv_mb_qrc.eoc.study import _resume_manifest


def test_schema_declares_reproducibility_fields():
    schema_path = Path(__file__).parents[1] / "docs" / "eoc_result_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["branch"]["const"] == "EOC-CVMB"
    assert {"config_hash", "chunk", "artifact_checksums"} <= set(schema["required"])


def test_resume_requires_matching_identity_and_checksums(tmp_path):
    artifact = tmp_path / "raw.csv"
    artifact.write_text("x\\n1\\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "config_hash": "a" * 64,
        "chunk": {"index": 0, "count": 1},
        "artifact_checksums": {"raw.csv": digest},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    chunk = cast(dict[str, Any], manifest["chunk"])
    assert _resume_manifest(tmp_path, "a" * 64, chunk) == manifest

    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _resume_manifest(tmp_path, "a" * 64, chunk)

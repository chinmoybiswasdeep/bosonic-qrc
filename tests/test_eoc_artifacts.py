import hashlib
import json
from pathlib import Path

import pytest

from bosonic_qrc_cv.eoc.study import _resume_manifest


def test_schema_declares_reproducibility_fields():
    schema_path = Path(__file__).parents[1] / "docs" / "eoc_result_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["branch"]["const"] == "EOC-CV"
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
    assert _resume_manifest(tmp_path, "a" * 64, manifest["chunk"]) == manifest

    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _resume_manifest(tmp_path, "a" * 64, manifest["chunk"])

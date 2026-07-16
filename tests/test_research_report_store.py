from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from analysis.promotion import ArtifactIntegrityError
from analysis.research_report_store import (
    ResearchReportArtifact,
    ResearchReportStore,
)


def _report():
    return {
        "portfolio_history": [
            {"timestamp": pd.Timestamp("2025-01-02"), "portfolio_value": 100.0},
            {"timestamp": pd.Timestamp("2025-01-03"), "portfolio_value": 101.0},
        ],
        "trades": [{"date": date(2025, 1, 3), "quantity": np.int64(2),
                    "price": np.float64(10.5)}],
        "pending_signals": [],
    }


def test_report_is_deterministic_and_round_trips(tmp_path):
    first = ResearchReportArtifact.create(_report())
    second = ResearchReportArtifact.create(dict(reversed(list(_report().items()))))
    assert first.report_hash == second.report_hash
    store = ResearchReportStore(tmp_path)
    path = store.persist(first)
    assert path == store.path_for(first.report_hash)
    loaded = store.load(first.report_hash)
    assert loaded == first
    assert loaded.report["trades"][0]["quantity"] == 2


def test_existing_identical_report_is_idempotent(tmp_path):
    artifact = ResearchReportArtifact.create(_report())
    store = ResearchReportStore(tmp_path)
    assert store.persist(artifact) == store.persist(artifact)


def test_tampered_document_is_rejected():
    artifact = ResearchReportArtifact.create(_report())
    document = json.loads(artifact.to_json())
    document["report"]["trades"][0]["price"] = 999.0
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        ResearchReportArtifact.from_json(json.dumps(document))


def test_ambiguous_or_extra_document_fields_are_rejected():
    artifact = ResearchReportArtifact.create(_report())
    document = artifact.to_json().rstrip()
    duplicate = document.replace(
        '"report_hash":', '"report_hash":"' + '0' * 64 + '","report_hash":',
        1)
    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        ResearchReportArtifact.from_json(duplicate)
    parsed = json.loads(document)
    parsed["unhashed_note"] = "not allowed"
    with pytest.raises(ArtifactIntegrityError, match="invalid fields"):
        ResearchReportArtifact.from_json(json.dumps(parsed))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_evidence_is_rejected(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        ResearchReportArtifact.create({"returns": [bad]})


def test_unknown_objects_are_not_stringified():
    with pytest.raises(TypeError, match="unsupported"):
        ResearchReportArtifact.create({"opaque": object()})

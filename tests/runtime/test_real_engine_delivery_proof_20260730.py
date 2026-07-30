"""Deterministic real-engine proof for the four formerly stubbed all-17 rows.

The natural attachment matrix proves that all 17 registered features reach their
production delivery seams, but historically supplied controlled outputs for the
localizer, syntax checker, covering runner, and recovery governor.  These tests
close that proof gap: each engine consumes real typed state (and, where relevant,
real repository files / SQLite graph / pytest execution) before its output is
bridged into canonical GT evidence.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from groundtruth.runtime import gateway
from groundtruth.runtime import reasoning_runtime as rr
from groundtruth.runtime.canonical_producers import (
    ProducerContext,
    produce_covering_red,
    produce_recovery,
    produce_syntax_result,
)
from groundtruth.runtime.covering_runner import (
    attribute_covering_red,
    run_covering_tests,
    select_covering_tests,
)
from groundtruth.runtime.edit_check import check_edit_syntax
from groundtruth.runtime.episode_state import EpisodeState
from groundtruth.runtime.hypothesis_ledger import (
    D_HYPOTHESIS_FALSIFIED,
    LedgerEvent,
    classify_edit_contradicted_contract,
)


REVISION = rr.RevisionVector(
    repository_content="real-engine-repo-20260730",
    graph="real-engine-graph-20260730",
    lsp="real-engine-lsp-20260730",
    runtime_evidence="real-engine-runtime-20260730",
)

_NODES_SCHEMA = (
    "CREATE TABLE nodes (id INTEGER PRIMARY KEY, label TEXT, name TEXT, "
    "qualified_name TEXT, file_path TEXT, start_line INT, end_line INT, "
    "signature TEXT, return_type TEXT, is_exported INT, is_test INT, "
    "language TEXT, parent_id INT)"
)
_EDGES_SCHEMA = (
    "CREATE TABLE edges (id INTEGER PRIMARY KEY, source_id INT, target_id INT, "
    "type TEXT, source_line INT, source_file TEXT, resolution_method TEXT, "
    "confidence REAL, metadata TEXT)"
)


def _graph(
    path: Path,
    *,
    include_test: bool = False,
    source_path: str = "src/pricing.py",
) -> Path:
    db = path / "graph.db"
    connection = sqlite3.connect(db)
    connection.execute(_NODES_SCHEMA)
    connection.execute(_EDGES_SCHEMA)
    connection.execute(
        "INSERT INTO nodes "
        "(id,label,name,file_path,start_line,end_line,is_test,language) "
        "VALUES (1,'Function','calculate_total',?,1,2,0,'python')",
        (source_path,),
    )
    if include_test:
        connection.execute(
            "INSERT INTO nodes "
            "(id,label,name,file_path,start_line,end_line,is_test,language) "
            "VALUES (2,'Function','test_calculate_total',"
            "'test_pricing.py',1,3,1,'python')"
        )
        connection.execute(
            "INSERT INTO edges "
            "(id,source_id,target_id,type,resolution_method,confidence) "
            "VALUES (1,2,1,'CALLS','import',0.99)"
        )
    connection.commit()
    connection.close()
    return db


def _context(subject: str, path: str, line: int) -> ProducerContext:
    return ProducerContext(
        subject=subject,
        provenance=((path, line),),
        revision=REVISION,
        decision_id=f"real-engine:{subject}",
        causal_neighborhood=(f"path:{path}",),
    )


def test_real_localizer_produces_ranked_repository_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "pricing.py"
    source.parent.mkdir()
    source.write_text(
        "def calculate_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    db = _graph(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert gateway._localize is not None

    rows = gateway._compute_ranked_localization_rows(
        gateway.GatewayState(
            graph_db=str(db),
            repo_root=str(tmp_path),
            graph_revision=REVISION.graph,
            issue_text="calculate_total returns the wrong price",
        )
    )

    assert rows
    assert rows[0] == ("src/pricing.py", 1, "calculate_total")
    assert all("test" not in path.lower() for path, _line, _symbol in rows)


def test_real_syntax_checker_bridges_parser_fact_and_physical_owner(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "pricing.py"
    source.parent.mkdir()
    source.write_text(
        "def calculate_total(items:\n    return sum(items)\n",
        encoding="utf-8",
    )

    result = check_edit_syntax(str(source), str(tmp_path))
    envelope = produce_syntax_result(
        context=_context(
            "src/pricing.py::calculate_total",
            "src/pricing.py",
            1,
        ),
        result=result,
    )

    assert result["verdict"] == "syntax_error"
    assert envelope is not None
    record = rr.canonical_evidence_from_envelope(envelope)
    assert record is not None
    assert record.feature_id == "syntax_result"
    assert record.owner_feature_ids == ("GT_EDIT_CHECK",)


@pytest.mark.skipif(
    shutil.which("pytest") is None,
    reason="pytest is required for real covering-runner proof",
)
def test_real_graph_and_pytest_red_bridge_attributed_covering_fact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pricing.py"
    test_file = tmp_path / "test_pricing.py"
    source.write_text(
        "def calculate_total(items):\n"
        "    raise RuntimeError('broken calculation')\n",
        encoding="utf-8",
    )
    test_file.write_text(
        "from pricing import calculate_total\n\n"
        "def test_calculate_total():\n"
        "    assert calculate_total([1, 2]) == 3\n",
        encoding="utf-8",
    )
    db = _graph(tmp_path, include_test=True, source_path="pricing.py")

    selected = select_covering_tests(
        str(db),
        {"pricing.py::calculate_total"},
        repo_root=str(tmp_path),
    )
    assert selected == [
        {"file": "test_pricing.py", "confidence": 0.99}
    ]
    result = run_covering_tests(
        str(tmp_path),
        [row["file"] for row in selected],
        per_file_timeout=30,
        total_budget_seconds=45,
    )
    attribution = attribute_covering_red(
        result,
        {"pricing.py"},
        test_files={"test_pricing.py"},
        covering_files=["test_pricing.py"],
    )
    envelope = produce_covering_red(
        context=_context(
            "pricing.py::calculate_total",
            "pricing.py",
            2,
        ),
        result=result,
        attribution=attribution,
    )

    assert result["verdict"] == "fail", (
        result.get("reason"),
        result.get("stdout_tail"),
        result.get("stderr_tail"),
    )
    assert attribution.attributed is True
    assert attribution.method == "trace_frame"
    assert envelope is not None
    record = rr.canonical_evidence_from_envelope(envelope)
    assert record is not None
    assert record.feature_id == "covering_red"
    assert record.owner_feature_ids == ()


def test_real_recovery_classifier_bridges_only_typed_contradiction() -> None:
    state = EpisodeState(episode_id="real-engine-recovery")
    state.failure_fingerprints.add("fp-price")
    state.last_failure_record = {
        "failure_fingerprint": "fp-price",
        "action_index": 2,
    }
    state.edit_events.append(
        {"index": 3, "blob": "src/pricing.py calculate_total"}
    )

    advisory = classify_edit_contradicted_contract(
        state,
        LedgerEvent(
            action_index=4,
            failure_fingerprint="fp-price",
            observation="assertion failed",
        ),
    )
    assert advisory is not None
    assert advisory.disposition == D_HYPOTHESIS_FALSIFIED
    envelope = produce_recovery(
        context=_context(
            "src/pricing.py::calculate_total",
            "src/pricing.py",
            2,
        ),
        advisory=advisory,
    )

    assert envelope is not None
    record = rr.canonical_evidence_from_envelope(envelope)
    assert record is not None
    assert record.feature_id == "recovery"
    assert record.owner_feature_ids == ("GT_HYPOTHESIS",)

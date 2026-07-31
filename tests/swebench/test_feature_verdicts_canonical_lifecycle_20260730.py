from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "artifact_deepswe"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.swebench import gt_feature_verdicts as verdicts  # noqa: E402


def _evaluate(tmp_path: Path, *rows: dict):
    ledger = tmp_path / "gt_runtime_ledger_task-1.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return verdicts.evaluate(tmp_path)


def _features(result) -> dict[str, verdicts.FeatureRow]:
    return {feature.feature_id: feature for feature in result["features"]}


def _strict_canonical_row() -> dict:
    claim = "Rank src/auth/session.py as the active implementation target."
    action = "View src/auth/session.py before editing."
    provenance = ["graph:def:refreshSession"]
    revision = {
        "repository_content": "repo-1",
        "graph": "graph-1",
        "lsp": "lsp-1",
        "runtime_evidence": "runtime-1",
    }
    state_vector = {
        "claim": claim,
        "actionable_consequence": action,
        "revision": revision,
        "fresh": True,
        "superseded": False,
        "lifecycle": "DELIVERED",
    }
    candidate_id = "candidate-localization"
    return {
        "schema": "gt.canonical_delivery.v1",
        "layer": "canonical.provider_delivery",
        "outcome": "delivered",
        "chars_delivered": 81,
        "content_sha256_16": "0123456789abcdef",
        "event_type": "canonical_provider_delivery",
        "evidence_ids": ["GT-E-localization"],
        "evidence_lineage": [
            {
                "candidate_id": candidate_id,
                "fact_class": "localization",
                "cap_owners": ["GT_LOC_RESLOT"],
            }
        ],
        "task_anchor": {
            "configured": True,
            "task_sha256": hashlib.sha256(b"task").hexdigest(),
            "task_chars": 4,
            "verbatim_text_present": True,
            "json_paths": ["$.messages[1].content"],
        },
        "semantic_receipts_complete": True,
        "semantic_receipts": [
            {
                "evidence_id": "GT-E-localization",
                "feature_id": "localization",
                "producer_id": "localization",
                "candidate_id": candidate_id,
                "fact_class": "localization",
                "cap_owners": ["GT_LOC_RESLOT"],
                "authorized_cap_owners": ["GT_LOC_RESLOT"],
                "subject": "src/auth/session.py",
                "claim": claim,
                "claim_sha256": hashlib.sha256(
                    claim.encode("utf-8")
                ).hexdigest(),
                "actionable_consequence": action,
                "intended_action": action,
                "actionable_consequence_sha256": hashlib.sha256(
                    action.encode("utf-8")
                ).hexdigest(),
                "provenance": provenance,
                "provenance_hash": verdicts._canonical_json_hash(provenance),
                "authority": "RESULT_DERIVED",
                "grade": "VERIFIED",
                "revision": revision,
                "repository_revision": "repo-1",
                "graph_revision": "graph-1",
                "revision_dependencies": ["nodes", "edges", "props_rev"],
                "observed_substrates": ["graph"],
                "fresh": True,
                "superseded": False,
                "lifecycle": "DELIVERED",
                "lifecycle_stage": "SOURCE_TARGET_SELECTION",
                "state_vector_hash": verdicts._canonical_json_hash(
                    state_vector
                ),
            }
        ],
    }


def test_current_canonical_delivery_requires_valid_semantic_receipts(
    tmp_path,
) -> None:
    valid = _evaluate(tmp_path, _strict_canonical_row())
    valid_features = _features(valid)
    assert valid_features["localization"].verdict == verdicts._VERDICT_FIRED
    assert valid_features["GT_LOC_RESLOT"].verdict == verdicts._VERDICT_FIRED
    assert valid["semantic_receipt_integrity"] == {"valid": 1}

    forged = _strict_canonical_row()
    forged["semantic_receipts"][0]["claim"] = "forged semantic claim"
    forged["fact_class"] = "localization"
    invalid = _evaluate(tmp_path, forged)
    invalid_features = _features(invalid)
    assert invalid_features["localization"].delivered == 0
    assert invalid_features["GT_LOC_RESLOT"].delivered == 0
    assert invalid["semantic_receipt_integrity"] == {"invalid": 1}


def test_normalized_delivery_requires_exact_fire_candidate_join(
    tmp_path,
) -> None:
    from groundtruth.runtime.trigger_opportunity import lifecycle_opportunity_id

    observation_id = "attempt-1:observation:localize"
    fire_id = lifecycle_opportunity_id(
        observation_id,
        "task_start",
        "localization",
    )
    delivery = _strict_canonical_row()
    delivery["observation_id"] = observation_id
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.lifecycle_opportunity.v1",
            "layer": "feature.lifecycle_opportunity",
            "outcome": "evaluated",
            "feature_id": "localization",
            "fact_class": "localization",
            "lifecycle_boundary": "task_start",
            "observation_id": observation_id,
            "feature_fire_id": fire_id,
        },
        {
            "schema": "gt.feature_fire_disposition.v1",
            "layer": "canonical_runtime.produce_funnel",
            "outcome": "suppressed_internal_only",
            "feature_fire_ids": [fire_id],
            "feature_dispositions": [
                {
                    "feature_fire_id": fire_id,
                    "feature_id": "localization",
                    "fact_class": "localization",
                    "lifecycle_boundary": "task_start",
                    "disposition": "produced",
                    "produced_candidate_ids": ["different-candidate"],
                    "available_candidate_ids": ["different-candidate"],
                }
            ],
        },
        delivery,
    )
    feature = _features(result)["localization"]
    assert feature.verdict == verdicts._VERDICT_FIRED
    assert feature.normalized_terminal_states == {"DELIVERY_FAILURE": 1}


def test_canonical_nested_lineage_credits_fact_and_authorized_cap(tmp_path) -> None:
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.canonical_delivery.v1",
            "layer": "canonical.provider_delivery",
            "outcome": "delivered",
            "chars_delivered": 81,
            "content_sha256_16": "0123456789abcdef",
            "event_type": "file_view",
            "evidence_lineage": [
                {
                    "candidate_id": "candidate-1",
                    "fact_class": "localization",
                    "cap_owners": ["GT_LOC_RESLOT"],
                }
            ],
        },
    )
    features = _features(result)
    assert features["localization"].delivered == 1
    assert features["localization"].delivered_chars == 81
    assert features["GT_LOC_RESLOT"].delivered == 1
    assert features["GT_LOC_RESLOT"].inherited is False
    assert "evidence_lineage.cap_owners" in features["GT_LOC_RESLOT"].attribution


def test_canonical_cap_claim_is_rejected_when_fact_binding_is_wrong(tmp_path) -> None:
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.canonical_delivery.v1",
            "layer": "canonical.provider_delivery",
            "outcome": "delivered",
            "chars_delivered": 81,
            "content_sha256_16": "0123456789abcdef",
            "event_type": "task_start",
            "evidence_lineage": [
                {
                    "candidate_id": "candidate-1",
                    "fact_class": "obligations",
                    "cap_owners": ["GT_LOC_RESLOT"],
                }
            ],
        },
    )
    feature = _features(result)["GT_LOC_RESLOT"]
    assert feature.inherited is True
    assert feature.delivered == 0


def test_unsealed_canonical_claim_credits_neither_fact_nor_cap(tmp_path) -> None:
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.canonical_delivery.v1",
            "layer": "canonical.provider_delivery",
            "outcome": "delivered",
            "chars_delivered": 81,
            "event_type": "file_view",
            "evidence_lineage": [
                {
                    "candidate_id": "candidate-1",
                    "fact_class": "localization",
                    "cap_owners": ["GT_LOC_RESLOT"],
                }
            ],
        },
    )
    features = _features(result)
    assert features["localization"].delivered == 0
    assert features["GT_LOC_RESLOT"].delivered == 0


def test_lifecycle_opportunity_is_a_measured_abstention_for_direct_feature(
    tmp_path,
) -> None:
    from groundtruth.runtime.trigger_opportunity import lifecycle_opportunity_id

    observation_id = "model-1:proposal:edit-1"
    fire_id = lifecycle_opportunity_id(
        observation_id,
        "edit_proposed",
        "syntax_result",
    )
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.lifecycle_opportunity.v1",
            "layer": "feature.lifecycle_opportunity",
            "outcome": "evaluated",
            "chars_delivered": 0,
            "event_type": "lifecycle_evaluation",
            "feature_id": "syntax_result",
            "fact_class": "syntax_result",
            "lifecycle_boundary": "edit_proposed",
            "observation_id": observation_id,
            "feature_fire_id": fire_id,
            "lifecycle_opportunity_id": fire_id,
        },
        {
            "schema": "gt.feature_fire_disposition.v1",
            "layer": "canonical_runtime.produce_funnel",
            "outcome": "suppressed_internal_only",
            "chars_delivered": 0,
            "feature_fire_ids": [fire_id],
            "feature_dispositions": [
                {
                    "feature_fire_id": fire_id,
                    "feature_id": "syntax_result",
                    "fact_class": "syntax_result",
                    "lifecycle_boundary": "edit_proposed",
                    "disposition": "abstained",
                    "produced_candidate_ids": [],
                    "available_candidate_ids": [],
                }
            ],
            "disposition": "abstained",
        },
    )
    feature = _features(result)["syntax_result"]
    assert feature.verdict == verdicts._VERDICT_ABSENT
    assert feature.verdict_detail == "correct_quiet"
    assert "correct-quiet" in feature.evidence
    assert result["lifecycle_opportunities"]["syntax_result"] == 1
    assert feature.terminal_dispositions == 1
    assert feature.normalized_terminal_states == {"INELIGIBLE": 1}
    assert result["lifecycle_integrity"]["unterminated_ids"] == []


def test_legacy_row_level_disposition_never_claims_per_feature_outcome(
    tmp_path,
) -> None:
    """A generic old-row status terminates integrity but cannot credit one feature."""
    from groundtruth.runtime.trigger_opportunity import lifecycle_opportunity_id

    observation_id = "attempt-legacy:observation:2"
    fire_id = lifecycle_opportunity_id(
        observation_id,
        "file_view",
        "caller_contract",
    )
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.lifecycle_opportunity.v1",
            "layer": "feature.lifecycle_opportunity",
            "outcome": "evaluated",
            "chars_delivered": 0,
            "feature_id": "caller_contract",
            "fact_class": "caller_contract",
            "lifecycle_boundary": "file_view",
            "observation_id": observation_id,
            "feature_fire_id": fire_id,
        },
        {
            "schema": "gt.feature_fire_disposition.v1",
            "layer": "canonical_runtime.produce_funnel",
            "outcome": "suppressed_internal_only",
            "chars_delivered": 0,
            "feature_fire_ids": [fire_id],
            "disposition": "produced",
        },
    )
    feature = _features(result)["caller_contract"]
    assert feature.terminal_dispositions == 1
    assert feature.verdict == verdicts._VERDICT_UNINSTRUMENTED
    assert feature.verdict_detail == "abstained_after_1_opportunities"
    assert feature.normalized_terminal_states == {"FAULT": 1}
    assert result["lifecycle_integrity"]["unterminated_ids"] == []
    assert result["lifecycle_integrity"]["legacy_generic_terminal_ids"] == [
        fire_id
    ]
    assert result["lifecycle_integrity"]["invalid_rows"] == {}


def test_canonical_delivery_timing_joins_by_observation_identity(tmp_path) -> None:
    """Provider delivery has its own event vocabulary; observation identity is authority."""
    from groundtruth.runtime.trigger_opportunity import lifecycle_opportunity_id

    observation_id = "attempt-1:observation:2"
    fire_id = lifecycle_opportunity_id(
        observation_id,
        "file_view",
        "caller_contract",
    )
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.lifecycle_opportunity.v1",
            "layer": "feature.lifecycle_opportunity",
            "outcome": "evaluated",
            "chars_delivered": 0,
            "feature_id": "caller_contract",
            "fact_class": "caller_contract",
            "lifecycle_boundary": "file_view",
            "observation_id": observation_id,
            "feature_fire_id": fire_id,
        },
        {
            "schema": "gt.feature_fire_disposition.v1",
            "layer": "canonical_runtime.produce_funnel",
            "outcome": "suppressed_internal_only",
            "chars_delivered": 0,
            "feature_fire_ids": [fire_id],
            "feature_dispositions": [
                {
                    "feature_fire_id": fire_id,
                    "feature_id": "caller_contract",
                    "fact_class": "caller_contract",
                    "lifecycle_boundary": "file_view",
                    "disposition": "produced",
                    "produced_candidate_ids": ["candidate-caller"],
                    "available_candidate_ids": ["candidate-caller"],
                }
            ],
            "disposition": "produced",
        },
        {
            "schema": "gt.canonical_delivery.v1",
            "layer": "canonical.provider_delivery",
            "outcome": "delivered",
            "event_type": "canonical_provider_delivery",
            "chars_delivered": 73,
            "content_sha256_16": "0123456789abcdef",
            "observation_id": observation_id,
            "evidence_lineage": [
                {
                    "candidate_id": "candidate-caller",
                    "fact_class": "caller_contract",
                    "cap_owners": [],
                }
            ],
        },
    )
    feature = _features(result)["caller_contract"]
    assert feature.verdict == verdicts._VERDICT_FIRED
    assert feature.on_time == "ON-TIME 1/1"
    assert feature.terminal_dispositions == 1
    assert feature.normalized_terminal_states == {"DELIVERED": 1}
    assert result["boundary_stamped"] == 1


def test_forged_lifecycle_identity_is_not_counted(tmp_path) -> None:
    result = _evaluate(
        tmp_path,
        {
            "schema": "gt.lifecycle_opportunity.v1",
            "layer": "feature.lifecycle_opportunity",
            "outcome": "evaluated",
            "chars_delivered": 0,
            "feature_id": "syntax_result",
            "fact_class": "syntax_result",
            "lifecycle_boundary": "edit_proposed",
            "observation_id": "model-1:proposal:edit-1",
            "feature_fire_id": "0" * 64,
        },
    )
    assert result["lifecycle_opportunities"]["syntax_result"] == 0
    assert result["lifecycle_integrity"]["invalid_rows"] == {
        "invalid_opportunity_identity": 1
    }


def test_synthesized_submit_red_uses_the_registered_byte_owner() -> None:
    source = (REPO / "artifact_deepswe" / "gt_mini_patch.py").read_text(
        encoding="utf-8"
    )
    assert 'cap_feature_ids=("GT_SS_SUBMIT_RED",)' in source
    assert 'cap_feature_ids=(("GT_REPRO_SYNTH",)' not in source

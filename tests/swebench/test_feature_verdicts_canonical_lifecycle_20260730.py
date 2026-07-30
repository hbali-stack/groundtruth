from __future__ import annotations

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
            "disposition": "abstained",
        },
    )
    feature = _features(result)["syntax_result"]
    assert feature.verdict == verdicts._VERDICT_UNINSTRUMENTED
    assert feature.verdict_detail == "abstained_after_1_opportunities"
    assert "lifecycle census" in feature.evidence
    assert result["lifecycle_opportunities"]["syntax_result"] == 1
    assert feature.terminal_dispositions == 1
    assert result["lifecycle_integrity"]["unterminated_ids"] == []


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

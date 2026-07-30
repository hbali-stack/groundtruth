from __future__ import annotations

from groundtruth.runtime import fact_registry
from groundtruth.runtime import reasoning_runtime as rr


EXPECTED_FACT_WINDOWS = {
    "obligations": ("task_start", "edit_proposed", "submit_proposed"),
    "localization": ("task_start", "edit_proposed", "edit_proposed"),
    "def_partition": ("search_result", "edit_proposed", "edit_result"),
    "caller_contract": ("search_result", "file_view", "edit_result"),
    "syntax_result": ("edit_proposed", "edit_result", "submit_proposed"),
    "signature_delta": ("edit_proposed", "edit_result", "submit_proposed"),
    "covering_red": ("edit_result", "test_result", "submit_proposed"),
    "submit_refusal": ("submit_proposed", "submit_proposed", "submit_proposed"),
    "newfile_precedent": (
        "failed_search",
        "file_create_proposed",
        "edit_result",
    ),
    "recovery": ("failure_obs", "test_result", "submit_proposed"),
}


def _window_tuple(window: rr.FeatureWindow) -> tuple[str, str, str]:
    return (
        window.earliest_event,
        window.deliver_by,
        window.corrective_boundary,
    )


def test_registry_exposes_proposal_boundaries_for_the_full_sdlc() -> None:
    assert {
        "edit_proposed",
        "file_create_proposed",
        "test_proposed",
        "compile_proposed",
        "verification_horizon",
        "submit_proposed",
    } <= fact_registry.EVENTS


def test_all_ten_delivery_facts_have_registry_owned_lifecycle_windows() -> None:
    delivery_facts = {
        feature_id
        for feature_id, registration in fact_registry.REGISTRY.items()
        if registration.fact_role == fact_registry.FACT_ROLE_DELIVERY
    }
    assert delivery_facts == set(EXPECTED_FACT_WINDOWS)
    assert {
        feature_id: fact_registry.lifecycle_window_for(feature_id)
        for feature_id in sorted(delivery_facts)
    } == EXPECTED_FACT_WINDOWS


def test_all_17_contracts_have_windows_and_caps_inherit_the_bound_fact() -> None:
    assert len(rr.FEATURE_CONTRACTS) == 17
    for feature_id, expected in EXPECTED_FACT_WINDOWS.items():
        contract = rr.feature_contract_for(feature_id)
        assert contract is not None
        assert contract.window is not None
        assert _window_tuple(contract.window) == expected

    for owner, bound_fact in rr._CAP_FACT_BINDING.items():
        owner_contract = rr.feature_contract_for(owner)
        fact_contract = rr.feature_contract_for(bound_fact)
        assert owner_contract is not None
        assert fact_contract is not None
        assert owner_contract.window == fact_contract.window


def test_every_edit_is_evaluated_but_only_evidence_matched_edits_interrupt() -> None:
    from groundtruth.runtime.commitment_control import (
        BatchPhase,
        CommitmentControlContext,
        CommitmentDecision,
        CommitmentEvidence,
        CommitmentIntent,
        decide_commitment_control,
    )

    action = rr.CanonicalAction(
        action_id="edit-1",
        operation=rr.ActionOperation.EDIT,
        tool_family="structured",
        tool_name="mini",
        structured_operation="edit",
        subject="src/a.py",
        targets=("src/a.py",),
    )
    intent = CommitmentIntent(action=action, sandboxed=True)
    base = dict(
        intents=(intent,),
        phase=BatchPhase.BEFORE_BATCH,
        active_decision_id="PATCH_CONSTRUCTION",
        proposing_model_call_id="call-1",
        failure_state=rr.FailurePolicyState.initial(attempt_id="attempt-1"),
        epistemic_prefix_may_change_decision=False,
        certificate_requirements_met=False,
        repository_revision="rev-1",
    )

    unmatched = decide_commitment_control(
        CommitmentControlContext(evidence=(), **base)
    )
    assert unmatched.decision is CommitmentDecision.ALLOW
    assert unmatched.execute_now == (intent,)

    matched = decide_commitment_control(
        CommitmentControlContext(
            evidence=(
                CommitmentEvidence(
                    evidence_id="GT-E-caller",
                    decision_id="PATCH_CONSTRUCTION",
                    grade=rr.EvidenceGrade.VERIFIED,
                    lifecycle=rr.EvidenceLifecycle.RELEASED,
                    fresh=True,
                    superseded=False,
                    release_allowed=True,
                    visible_to_model_call_ids=(),
                    material_action_ids=("edit-1",),
                    staged_for_next_inference=True,
                ),
            ),
            **base,
        )
    )
    assert matched.decision is CommitmentDecision.FRESH_INFERENCE
    assert matched.deferred == (intent,)


def test_proposal_census_emits_stable_feature_fire_identity(monkeypatch) -> None:
    from artifact_deepswe import gt_mini_patch as seam
    from groundtruth.runtime.trigger_opportunity import (
        lifecycle_opportunity_id,
    )

    rows = []
    monkeypatch.setattr(seam, "_inseam_metrics_on", lambda: True)
    monkeypatch.setattr(seam, "_ledger_line_direct", rows.append)
    seam._emitted_trigger_ids.clear()

    observation_id = "model-1:proposal:edit-1"
    fire_ids = seam._record_trigger_opportunities(
        ("edit_proposed",),
        observation_id=observation_id,
        action_ids=("edit-1",),
        subjects=("src/a.py",),
    )

    lifecycle = [
        row
        for row in rows
        if row["layer"] == "feature.lifecycle_opportunity"
    ]
    assert set(fire_ids) == {row["feature_fire_id"] for row in lifecycle}
    assert {row["feature_id"] for row in lifecycle} == {
        "obligations",
        "localization",
        "def_partition",
        "syntax_result",
        "signature_delta",
        "GT_EDIT_CHECK",
        "GT_PATCH_DELTA",
        "GT_LOC_RESLOT",
    }
    for row in lifecycle:
        expected = lifecycle_opportunity_id(
            observation_id,
            "edit_proposed",
            row["feature_id"],
        )
        assert row["feature_fire_id"] == expected
        assert row["lifecycle_opportunity_id"] == expected
        assert row["action_ids"] == ["edit-1"]
        assert row["subjects"] == ["src/a.py"]

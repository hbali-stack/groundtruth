"""Stable task-contract anchoring and abandoned-delivery recovery.

Provider-committed immutable obligations satisfy later decision completeness
without repeated bytes. A merely RELEASED/DELIVERED generation can still
rematerialize after an abandoned provider path because it is not yet a durable
task anchor.
"""

from __future__ import annotations

import json

from artifact_deepswe import gt_mini_patch as seam
from groundtruth.runtime import reasoning_runtime as rr


REVISION = rr.RevisionVector(
    repository_content="repo-obligation",
    graph="graph-obligation",
    lsp="lsp-obligation",
    runtime_evidence="runtime-obligation",
)

SATISFIED = frozenset(
    {
        rr.TemporalPredicate.PRODUCER_COMPUTATION_COMPLETE,
        rr.TemporalPredicate.REVISION_DEPENDENCIES_CAPTURED,
        rr.TemporalPredicate.ACTIVE_DECISION_CONTEXT_MATCHES,
        rr.TemporalPredicate.ACTIVE_DECISION_ID_MATCHES,
        rr.TemporalPredicate.REASONING_GRAPH_CONNECTED,
        rr.TemporalPredicate.COMMITMENT_WINDOW_OPEN,
        rr.TemporalPredicate.AUTHORIZED_BYTE_OWNER_LINEAGE_PRESENT,
    }
)


def _obligation() -> rr.EvidenceRecord:
    contract = rr.feature_contract_for("obligations")
    assert contract is not None
    return rr.EvidenceRecord(
        evidence_id="GT-E-task-obligation-source",
        feature_id="obligations",
        decision_context=contract.decision_context,
        roles=contract.roles,
        subject="issue",
        claim="Task requirements: preserve the returned Session.",
        actionable_consequence=(
            "Keep every listed task requirement open until validated."
        ),
        provenance=("issue:1",),
        grade=rr.EvidenceGrade.VERIFIED,
        revision=REVISION,
        causal_neighborhood=(
            f"decision:{contract.decision_context.value}",
            "subject:issue",
        ),
        lifecycle=rr.EvidenceLifecycle.PENDING,
        fresh=True,
        already_visible=False,
        superseded=False,
        mandatory_reason=rr.MandatoryReason.TASK_OBLIGATION,
        token_cost=28,
        failure_prevention=5,
        causal_value=4,
        contradiction_resolution=0,
        anchoring_risk=0,
        revision_dependencies=contract.revision_dependencies,
        authority=rr.Authority.RESULT_DERIVED,
        observed_substrates=("issue_text", "obligation_parser"),
    )


def _caller_contract(runtime: rr.AttemptReasoningRuntime) -> rr.EvidenceRecord:
    """A later, useful patch fact that must not be starved by a stable contract."""
    contract = rr.feature_contract_for("caller_contract")
    assert contract is not None
    return rr.EvidenceRecord(
        evidence_id="GT-E-caller-contract-delta",
        feature_id="caller_contract",
        decision_context=contract.decision_context,
        roles=contract.roles,
        subject="src/auth/session.py",
        claim="Caller expects the returned Session identity to remain stable.",
        actionable_consequence=(
            "Preserve the existing Session object while applying the patch."
        ),
        provenance=("callsite:tests/test_session.py:17",),
        grade=rr.EvidenceGrade.VERIFIED,
        revision=runtime.work_state.revision,
        causal_neighborhood=(
            f"decision:{rr.DecisionContext.PATCH_CONSTRUCTION.value}",
            "obligation:task",
            "subject:src/auth/session.py",
        ),
        lifecycle=rr.EvidenceLifecycle.PENDING,
        fresh=True,
        already_visible=False,
        superseded=False,
        mandatory_reason=None,
        token_cost=24,
        failure_prevention=4,
        causal_value=4,
        contradiction_resolution=0,
        anchoring_risk=0,
        revision_dependencies=contract.revision_dependencies,
        authority=rr.Authority.RESULT_DERIVED,
        observed_substrates=("graph", "lsp"),
    )


def _append_boundary(
    runtime: rr.AttemptReasoningRuntime,
    event_id: str,
    outcome: rr.SemanticOutcome,
) -> None:
    prior = runtime.journal.events(runtime.attempt_id)
    event = rr.CanonicalEvent(
        event_id=event_id,
        attempt_id=runtime.attempt_id,
        sequence=runtime.work_state.sequence + 1,
        kind=rr.EventKind.OBSERVATION_COMMITTED,
        authority=rr.Authority.STRUCTURED,
        outcomes=(outcome,),
        revision_before=runtime.work_state.revision,
        revision_after=runtime.work_state.revision,
        previous_event_hash=prior[-1].content_hash if prior else "",
        observation_id=f"observation:{event_id}",
        carrier="native_tool_result",
    )
    runtime.append_event(event)
    assert runtime.work_state.decision_window_key == event_id


def _open_patch_window(runtime: rr.AttemptReasoningRuntime, event_id: str) -> None:
    _append_boundary(
        runtime,
        event_id,
        rr.SemanticOutcome(
            kind=rr.SemanticKind.EDIT_PROPOSED,
            subject="src/auth/session.py",
            authority=rr.Authority.STRUCTURED,
        ),
    )


def _decision(runtime: rr.AttemptReasoningRuntime) -> rr.ActiveDecision:
    decision = seam.CanonicalRuntimeAttachment._active_decision(
        tuple(runtime._evidence.values()),
        runtime.work_state,
        runtime.work_state.revision,
        (),
    )
    assert decision.context is rr.DecisionContext.PATCH_CONSTRUCTION
    return decision


def _prepare(
    runtime: rr.AttemptReasoningRuntime,
    *,
    observation_id: str,
    model_call_id: str,
    substrates: tuple[str, ...] = ("issue_text", "obligation_parser"),
) -> rr.InferencePlan:
    return runtime.prepare_next_inference(
        decisions=(_decision(runtime),),
        satisfied_predicates=SATISFIED,
        commitment_window=rr.CommitmentWindowState.OPEN,
        available_substrates=substrates,
        native_observation="native observation",
        observation_id=observation_id,
        source_model_call_id=f"source:{observation_id}",
        model_call_id=model_call_id,
    )


def _deliver_and_commit(
    runtime: rr.AttemptReasoningRuntime,
    plan: rr.InferencePlan,
    *,
    response_id: str,
) -> None:
    assert plan.delivery_attempt_id
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": plan.compilation.capsule_text}
                ],
            }
        ]
    }
    runtime.bind_provider_payload(plan.delivery_attempt_id, payload)
    runtime.mark_dispatched(plan.delivery_attempt_id, payload)
    accepted = runtime.mark_provider_accepted(
        plan.delivery_attempt_id,
        provider_response_id=response_id,
    )
    runtime.record_provider_terminal(
        plan.delivery_attempt_id,
        rr.ModelCallAttempt(
            model_call_id=accepted.model_call_id,
            joined_capsule_hash=accepted.joined_capsule_hash,
            provider_payload_hash=accepted.provider_payload_hash,
            provider_response_id=accepted.provider_response_id,
            terminal_kind=rr.ProviderTerminalKind.COMPLETED,
        ),
    )
    runtime.commit_provider_response(
        plan.delivery_attempt_id,
        response_hash="a" * 64,
    )


def test_reopened_patch_window_reuses_provider_proven_contract_without_bytes(
    tmp_path,
) -> None:
    journal = rr.RuntimeJournal(tmp_path / "standing-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-standing-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "patch-window-1")
        first = _prepare(
            runtime,
            observation_id="obs-window-1",
            model_call_id="model-window-1",
        )
        _deliver_and_commit(runtime, first, response_id="response-window-1")

        source_after_delivery = runtime.evidence_record(source.evidence_id)
        assert source_after_delivery.lifecycle is rr.EvidenceLifecycle.ACTIVE

        # A second inference in the same window is not a new standing dose.
        same_window = _prepare(
            runtime,
            observation_id="obs-window-1-later",
            model_call_id="model-window-1-later",
        )
        assert not same_window.delivery_attempt_id
        assert tuple(runtime._evidence) == (source.evidence_id,)

        # EDIT_PROPOSED is a decision boundary even though the phase remains
        # IMPLEMENTATION. The new window must get a new identity, not ACTIVE ->
        # PENDING on the provider-proven source.
        _open_patch_window(runtime, "patch-window-2")
        blocked = _prepare(
            runtime,
            observation_id="obs-window-2-blocked",
            model_call_id="model-window-2-blocked",
            substrates=(),
        )
        assert not blocked.delivery_attempt_id
        assert tuple(runtime._evidence) == (source.evidence_id,)
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.ACTIVE
        )
        # With no semantic delta, satisfying the stable prerequisite alone is
        # intentionally quiet even when its original substrates are present.
        reopened = _prepare(
            runtime,
            observation_id="obs-window-2",
            model_call_id="model-window-2",
        )
        assert not reopened.delivery_attempt_id
        assert reopened.oracle_decision.decision_complete
        assert reopened.oracle_decision.coalition == ()
        assert tuple(runtime._evidence) == (source.evidence_id,)

        restarted = rr.AttemptReasoningRuntime(
            attempt_id=runtime.attempt_id,
            journal=journal,
            initial_revision=REVISION,
            role_driven_coalition=True,
        )
        assert (
            restarted.evidence_record(source.evidence_id)
            == runtime.evidence_record(source.evidence_id)
        )
    finally:
        journal.close()


def test_provider_proven_task_contract_is_a_stable_anchor_for_sparse_deltas(
    tmp_path,
) -> None:
    """A provider-proven immutable contract satisfies later decision completeness
    without repeating its bytes or starving a genuinely new caller fact."""
    journal = rr.RuntimeJournal(tmp_path / "stable-task-anchor.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-stable-task-anchor",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "stable-anchor-window-1")
        first = _prepare(
            runtime,
            observation_id="obs-stable-anchor-1",
            model_call_id="model-stable-anchor-1",
        )
        _deliver_and_commit(
            runtime,
            first,
            response_id="response-stable-anchor-1",
        )
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.ACTIVE
        )

        runtime.ingest_evidence(_caller_contract(runtime))
        _open_patch_window(runtime, "stable-anchor-window-2")
        decision = _decision(runtime)
        assert decision.established_roles == (
            rr.EvidenceRole.BEHAVIORAL_CONTRACT,
        )
        assert decision.established_evidence_ids == (source.evidence_id,)

        delta = _prepare(
            runtime,
            observation_id="obs-stable-anchor-2",
            model_call_id="model-stable-anchor-2",
            substrates=(
                "issue_text",
                "obligation_parser",
                "graph",
                "lsp",
            ),
        )
        assert delta.delivery_attempt_id, (
            delta.oracle_decision,
            delta.compilation.failure_code,
            runtime.evidence_record("GT-E-caller-contract-delta"),
        )
        assert delta.oracle_decision.decision_complete
        assert tuple(
            item.evidence_id for item in delta.oracle_decision.coalition
        ) == ("GT-E-caller-contract-delta",)
        assert source.claim not in delta.compilation.capsule_text
        assert not tuple(
            item
            for item in runtime._evidence.values()
            if item.standing_source_evidence_id == source.evidence_id
        )
    finally:
        journal.close()


def test_non_patch_decision_does_not_rematerialize_delivered_obligation(
    tmp_path,
) -> None:
    journal = rr.RuntimeJournal(tmp_path / "non-patch-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-non-patch-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "non-patch-source-window")
        source_plan = _prepare(
            runtime,
            observation_id="obs-non-patch-source",
            model_call_id="model-non-patch-source",
        )
        _deliver_and_commit(
            runtime,
            source_plan,
            response_id="response-non-patch-source",
        )
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.ACTIVE
        )

        _append_boundary(
            runtime,
            "validation-window",
            rr.SemanticOutcome(
                kind=rr.SemanticKind.TEST_PASS,
                subject="tests/test_session.py",
                authority=rr.Authority.STRUCTURED,
            ),
        )
        decision = seam.CanonicalRuntimeAttachment._active_decision(
            tuple(runtime._evidence.values()),
            runtime.work_state,
            runtime.work_state.revision,
            (),
        )
        assert decision.context is rr.DecisionContext.PATCH_PROPAGATION
        plan = runtime.prepare_next_inference(
            decisions=(decision,),
            satisfied_predicates=SATISFIED,
            commitment_window=rr.CommitmentWindowState.OPEN,
            available_substrates=("issue_text", "obligation_parser"),
            native_observation="native observation",
            observation_id="obs-non-patch",
            source_model_call_id="source-non-patch",
            model_call_id="model-non-patch",
        )
        assert not plan.delivery_attempt_id
        assert tuple(runtime._evidence) == (source.evidence_id,)
    finally:
        journal.close()


def test_released_generation_can_rematerialize_after_provider_path_abandons_it(
    tmp_path,
) -> None:
    journal = rr.RuntimeJournal(tmp_path / "released-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-released-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "released-window-1")
        first = _prepare(
            runtime,
            observation_id="obs-released-1",
            model_call_id="model-released-1",
        )
        assert first.delivery_attempt_id
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.RELEASED
        )

        _open_patch_window(runtime, "released-window-2")
        reopened = _prepare(
            runtime,
            observation_id="obs-released-2",
            model_call_id="model-released-2",
        )

        clones = tuple(
            item
            for item in runtime._evidence.values()
            if item.evidence_id != source.evidence_id
        )
        assert len(clones) == 1
        assert reopened.delivery_attempt_id
        assert tuple(
            item.evidence_id for item in reopened.oracle_decision.coalition
        ) == (clones[0].evidence_id,)
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.RELEASED
        )
    finally:
        journal.close()


def test_fallback_patch_candidate_does_not_repeat_stable_obligation(
    tmp_path,
) -> None:
    journal = rr.RuntimeJournal(tmp_path / "fallback-patch-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-fallback-patch-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "fallback-patch-window-1")
        first = _prepare(
            runtime,
            observation_id="obs-fallback-patch-1",
            model_call_id="model-fallback-patch-1",
        )
        _deliver_and_commit(
            runtime,
            first,
            response_id="response-fallback-patch-1",
        )
        _open_patch_window(runtime, "fallback-patch-window-2")

        patch = _decision(runtime)
        completion = rr.ActiveDecision(
            decision_id="completion-before-fallback-patch",
            context=rr.DecisionContext.COMPLETION,
            primary_claim="Determine whether the repair is complete.",
            required_roles=(rr.EvidenceRole.TERMINAL_ASSURANCE,),
            useful_roles=(rr.EvidenceRole.BEHAVIORAL_CONTRACT,),
            causal_neighborhood=patch.causal_neighborhood,
            token_budget=200,
            current_revision=runtime.work_state.revision,
        )
        plan = runtime.prepare_next_inference(
            decisions=(completion, patch),
            satisfied_predicates=SATISFIED,
            commitment_window=rr.CommitmentWindowState.OPEN,
            available_substrates=("issue_text", "obligation_parser"),
            native_observation="native observation",
            observation_id="obs-fallback-patch-2",
            source_model_call_id="source-fallback-patch-2",
            model_call_id="model-fallback-patch-2",
        )

        assert not plan.delivery_attempt_id
        clones = tuple(
            item
            for item in runtime._evidence.values()
            if item.standing_source_evidence_id == source.evidence_id
        )
        assert clones == ()
        assert runtime.evidence_record(source.evidence_id).lifecycle is (
            rr.EvidenceLifecycle.ACTIVE
        )
    finally:
        journal.close()


def test_role_driven_off_preserves_single_dose_behavior(tmp_path) -> None:
    journal = rr.RuntimeJournal(tmp_path / "default-off-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-default-off-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=False,
    )
    source = _obligation()
    runtime.ingest_evidence(source)

    try:
        _open_patch_window(runtime, "default-off-window-1")
        first = _prepare(
            runtime,
            observation_id="obs-default-off-1",
            model_call_id="model-default-off-1",
        )
        _deliver_and_commit(runtime, first, response_id="response-default-off-1")
        _open_patch_window(runtime, "default-off-window-2")

        second = _prepare(
            runtime,
            observation_id="obs-default-off-2",
            model_call_id="model-default-off-2",
        )

        assert not second.delivery_attempt_id
        assert tuple(runtime._evidence) == (source.evidence_id,)
        assert not runtime.evidence_record(
            source.evidence_id
        ).decision_window_generation
    finally:
        journal.close()


def test_legacy_evidence_json_defaults_new_generation_fields_to_empty() -> None:
    raw = json.loads(rr._canonical_json(_obligation()))
    raw.pop("standing_source_evidence_id")
    raw.pop("decision_window_generation")

    restored = rr._evidence_record_from_json(
        json.dumps(raw, sort_keys=True, separators=(",", ":"))
    )

    assert restored.standing_source_evidence_id == ""
    assert restored.decision_window_generation == ""

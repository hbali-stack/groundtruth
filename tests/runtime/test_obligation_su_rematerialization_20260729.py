"""SOURCE_UNDERSTANDING reuses a provider-proven stable task contract.

The original P2-1 repair rematerialized the full task obligation in every SU
window to avoid starving its required BEHAVIORAL_CONTRACT role. The stable
anchor model keeps the completeness guarantee while sending no repeated bytes:
only a later semantic delta can create another capsule.
"""

from __future__ import annotations

from artifact_deepswe import gt_mini_patch as seam  # noqa: F401 — decision helper parity
from groundtruth.runtime import reasoning_runtime as rr

REVISION = rr.RevisionVector(
    repository_content="repo-su-obligation",
    graph="graph-su-obligation",
    lsp="lsp-su-obligation",
    runtime_evidence="runtime-su-obligation",
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


def _edit_boundary(runtime: rr.AttemptReasoningRuntime, event_id: str) -> None:
    _append_boundary(
        runtime,
        event_id,
        rr.SemanticOutcome(
            kind=rr.SemanticKind.EDIT_PROPOSED,
            subject="src/auth/session.py",
            authority=rr.Authority.STRUCTURED,
        ),
    )


def _patch_decision(runtime: rr.AttemptReasoningRuntime) -> rr.ActiveDecision:
    decision = seam.CanonicalRuntimeAttachment._active_decision(
        tuple(runtime._evidence.values()),
        runtime.work_state,
        runtime.work_state.revision,
        (),
    )
    assert decision.context is rr.DecisionContext.PATCH_CONSTRUCTION
    return decision


def _su_decision(runtime: rr.AttemptReasoningRuntime) -> rr.ActiveDecision:
    """A SOURCE_UNDERSTANDING decision requiring BEHAVIORAL_CONTRACT — the exact
    role shape the seam's role_requirements table declares."""
    patch = _patch_decision(runtime)
    return rr.ActiveDecision(
        decision_id="su-window-decision",
        context=rr.DecisionContext.SOURCE_UNDERSTANDING,
        primary_claim="Understand the active source contract before mutation.",
        required_roles=(rr.EvidenceRole.BEHAVIORAL_CONTRACT,),
        useful_roles=(),
        causal_neighborhood=patch.causal_neighborhood,
        token_budget=200,
        current_revision=runtime.work_state.revision,
        established_roles=patch.established_roles,
        established_evidence_ids=patch.established_evidence_ids,
        established_grade=patch.established_grade,
    )


def _prepare(
    runtime: rr.AttemptReasoningRuntime,
    decision: rr.ActiveDecision,
    *,
    observation_id: str,
    model_call_id: str,
) -> rr.InferencePlan:
    return runtime.prepare_next_inference(
        decisions=(decision,),
        satisfied_predicates=SATISFIED,
        commitment_window=rr.CommitmentWindowState.OPEN,
        available_substrates=("issue_text", "obligation_parser"),
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


def test_source_understanding_reuses_the_provider_proven_task_contract(
    tmp_path,
) -> None:
    """A provider-proven immutable contract completes SU without repeat bytes."""
    journal = rr.RuntimeJournal(tmp_path / "su-obligation.sqlite3")
    journal.open()
    runtime = rr.AttemptReasoningRuntime(
        attempt_id="attempt-su-obligation",
        journal=journal,
        initial_revision=REVISION,
        role_driven_coalition=True,
    )
    source = _obligation()
    runtime.ingest_evidence(source)
    try:
        _edit_boundary(runtime, "su-source-window")
        first = _prepare(
            runtime,
            _patch_decision(runtime),
            observation_id="obs-su-source",
            model_call_id="model-su-source",
        )
        _deliver_and_commit(runtime, first, response_id="response-su-source")
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.ACTIVE
        )

        # a NEW window whose open decision is SOURCE_UNDERSTANDING
        _edit_boundary(runtime, "su-window-2")
        plan = _prepare(
            runtime,
            _su_decision(runtime),
            observation_id="obs-su-2",
            model_call_id="model-su-2",
        )

        clones = tuple(
            item
            for item in runtime._evidence.values()
            if item.standing_source_evidence_id == source.evidence_id
        )
        assert clones == ()
        assert not plan.delivery_attempt_id
        assert plan.oracle_decision.decision_complete
        assert (
            runtime.evidence_record(source.evidence_id).lifecycle
            is rr.EvidenceLifecycle.ACTIVE
        ), "rematerialization must never rewind the provider-proven source"
    finally:
        journal.close()

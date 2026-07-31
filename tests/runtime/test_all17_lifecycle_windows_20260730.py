from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_result_dispositions_are_per_feature_not_row_level(monkeypatch) -> None:
    """One produced FACT must not falsely credit every feature at that boundary."""
    from artifact_deepswe import gt_mini_patch as seam

    monkeypatch.setattr(seam, "_inseam_metrics_on", lambda: True)
    monkeypatch.setattr(seam, "_ledger_line_direct", lambda row: None)
    seam._emitted_trigger_ids.clear()
    observation_id = "attempt-1:observation:2"
    fire_ids = seam._record_trigger_opportunities(
        ("file_view",),
        observation_id=observation_id,
    )
    caller = SimpleNamespace(
        feature_id="caller_contract",
        owner_feature_ids=(),
        evidence_id="GT-E-caller",
    )
    dispositions = seam._feature_fire_dispositions(
        observation_id=observation_id,
        observed_events=("file_view",),
        feature_fire_ids=fire_ids,
        produced_records=(caller,),
        available_records=(caller,),
    )
    by_feature = {row["feature_id"]: row for row in dispositions}
    assert by_feature["caller_contract"]["disposition"] == "produced"
    assert by_feature["caller_contract"]["produced_candidate_ids"] == [
        "GT-E-caller"
    ]
    # Other features whose lifecycle includes file_view receive their own
    # truthful terminal result; they are not credited by caller_contract bytes.
    for feature_id, row in by_feature.items():
        if feature_id != "caller_contract":
            assert row["disposition"] == "abstained"
            assert row["produced_candidate_ids"] == []


def test_commitment_plan_terminates_each_proposal_feature_independently(
    monkeypatch,
) -> None:
    """Executed and deferred proposal IDs must receive different truthful outcomes."""
    from artifact_deepswe import gt_mini_patch as seam
    from groundtruth.runtime.commitment_control import CommitmentIntent

    rows: list[dict] = []
    monkeypatch.setattr(seam, "_ledger_line_direct", rows.append)
    monkeypatch.setattr(seam, "_inseam_metrics_on", lambda: True)
    seam._emitted_trigger_ids.clear()
    permitted_action = rr.CanonicalAction(
        action_id="edit-permitted",
        operation=rr.ActionOperation.EDIT,
        tool_family="structured",
        tool_name="mini",
        structured_operation="edit",
        subject="src/a.py",
        targets=("src/a.py",),
    )
    deferred_action = rr.CanonicalAction(
        action_id="edit-deferred",
        operation=rr.ActionOperation.EDIT,
        tool_family="structured",
        tool_name="mini",
        structured_operation="edit",
        subject="src/b.py",
        targets=("src/b.py",),
    )
    permitted = CommitmentIntent(action=permitted_action, sandboxed=True)
    deferred = CommitmentIntent(action=deferred_action, sandboxed=True)
    context = SimpleNamespace(
        intents=(permitted, deferred),
        proposing_model_call_id="provider-response-1",
        repository_revision="repo-1",
    )
    for intent in context.intents:
        seam._record_trigger_opportunities(
            ("edit_proposed",),
            observation_id=(
                f"{context.proposing_model_call_id}:proposal:"
                f"{intent.action.action_id}"
            ),
            action_ids=(intent.action.action_id,),
            subjects=(intent.action.subject,),
        )
    plan = SimpleNamespace(
        execute_now=(permitted,),
        deferred=(deferred,),
        decision=SimpleNamespace(name="FRESH_INFERENCE"),
        reason_code="qualifying_evidence",
        interruption_key="interrupt-1",
        qualifying_evidence_ids=("GT-E-caller",),
        fresh_inference_required=True,
    )
    attachment = seam.CanonicalRuntimeAttachment(
        attached=True,
        attempt_runtime=SimpleNamespace(),
        provider_boundary=None,
        gateway_state=None,
        graph_revision="graph-1",
    )
    attachment._observe_commitment_plan(context, plan, ({}, {}))

    opportunity_ids = {
        row["feature_fire_id"]
        for row in rows
        if row.get("schema") == "gt.lifecycle_opportunity.v1"
    }
    terminal_rows = [
        row
        for row in rows
        if row.get("layer") == "commitment_boundary.plan"
        and row.get("schema") == "gt.feature_fire_disposition.v1"
    ]
    assert len(terminal_rows) == 1
    terminal = terminal_rows[0]
    assert set(terminal["feature_fire_ids"]) == opportunity_ids
    assert len(terminal["feature_dispositions"]) == len(opportunity_ids)
    dispositions = {
        item["feature_fire_id"]: item["disposition"]
        for item in terminal["feature_dispositions"]
    }
    for row in rows:
        if row.get("schema") != "gt.lifecycle_opportunity.v1":
            continue
        expected = (
            "permitted"
            if row["action_ids"] == ["edit-permitted"]
            else "deferred"
        )
        assert dispositions[row["feature_fire_id"]] == expected


def test_real_result_path_emits_then_terminates_each_lifecycle_id_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the production observer seam that was broken in live run 30581286663."""
    from artifact_deepswe import gt_mini_patch as seam

    class Model:
        def _prepare_messages_for_api(self, messages):
            return messages

        def _query(self, messages, **kwargs):
            return SimpleNamespace(id="", status="failed", choices=[])

    class Agent:
        def add_messages(self, *messages):
            return list(messages)

        def execute_actions(self, message):
            return []

    source = tmp_path / "src" / "api.py"
    source.parent.mkdir(parents=True)
    source.write_text("def get_user(uid):\n    return uid\n", encoding="utf-8")
    rows: list[dict] = []
    monkeypatch.setattr(seam, "_CANONICAL_RUNTIME_ATTACHMENT", None)
    monkeypatch.setattr(seam, "_root", lambda: str(tmp_path))
    monkeypatch.setattr(seam, "_db_path", lambda: str(tmp_path / "graph.db"))
    monkeypatch.setattr(seam, "_inseam_metrics_on", lambda: True)
    monkeypatch.setattr(seam, "_ledger_line_direct", rows.append)
    seam._emitted_trigger_ids.clear()
    attachment = seam.install_canonical_runtime(
        model=Model(),
        agent=Agent(),
        env={
            "GT_ATTEMPT_ID": "attempt-result-lifecycle",
            "GT_RUNTIME_LEDGER": str(tmp_path / "runtime.jsonl"),
            "GT_BRIEF_FILE": str(tmp_path / "missing-brief.txt"),
        },
        task="inspect get_user",
    )
    action = {"command": "view", "path": "src/api.py"}
    attachment.observe_action_proposal(action)
    attachment.observe_action_result(
        action,
        {"output": source.read_text(encoding="utf-8"), "returncode": 0},
    )

    result_observation_id = "attempt-result-lifecycle:observation:2"
    opportunities = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("schema") == "gt.lifecycle_opportunity.v1"
        and row.get("observation_id") == result_observation_id
    ]
    funnels = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("schema") == "gt.feature_fire_disposition.v1"
        and row.get("layer") == "canonical_runtime.produce_funnel"
        and row.get("result_observation_id") == result_observation_id
    ]
    assert opportunities
    assert len(funnels) == 1
    funnel_index, funnel = funnels[0]
    opportunity_ids = [row["feature_fire_id"] for _, row in opportunities]
    assert len(opportunity_ids) == len(set(opportunity_ids))
    assert set(funnel["feature_fire_ids"]) == set(opportunity_ids)
    assert max(index for index, _ in opportunities) < funnel_index

    dispositions = funnel["feature_dispositions"]
    assert len(dispositions) == len(opportunity_ids)
    assert {item["feature_fire_id"] for item in dispositions} == set(
        opportunity_ids
    )
    assert len(
        [row for row in rows if set(row.get("feature_fire_ids") or ()) & set(
            opportunity_ids
        )]
    ) == 1
    opportunity_by_id = {
        row["feature_fire_id"]: row
        for _, row in opportunities
    }
    for item in dispositions:
        opportunity = opportunity_by_id[item["feature_fire_id"]]
        assert item["feature_id"] == opportunity["feature_id"]
        assert item["fact_class"] == opportunity["fact_class"]
        assert item["lifecycle_boundary"] == opportunity["lifecycle_boundary"]
        assert item["disposition"] in {"produced", "available", "abstained"}
    attachment.attempt_runtime.journal.close()

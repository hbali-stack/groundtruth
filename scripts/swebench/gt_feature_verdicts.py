#!/usr/bin/env python3
"""Per-feature verdict reporter for the 17 DIRECT GroundTruth features.

Answers ONE question from a completed run's artifacts: for each of the 17 DIRECT
features (10 FACT + 7 CAP byte owners), is it working, and did it deliver at the
boundary its contract names?

THE THREE VERDICTS -- the distinction is the whole point:

  FIRED             the trigger occurred AND evidence reached the model
                    (``outcome == "delivered"`` AND ``chars_delivered > 0``).
  TRIGGER-ABSENT    the trigger never occurred in this trajectory.  CORRECT-QUIET.
                    A legitimate outcome, NEVER a feature failure.  Six of the 17
                    are MISTAKE-GATED: they need the agent to write a syntax error,
                    submit dirty, break a signature, create a new file, stall, or
                    leave a covering test failing.  A competent agent produces few
                    of those, so a dark mistake-gated feature is evidence the run
                    went well -- not evidence the feature broke.
  DELIVERY-FAILURE  the trigger DID occur and evidence was produced, but the bytes
                    never reached the model.  THIS IS THE ONLY REAL FAILURE.

Everything the tool asserts is derived from executable authorities, never re-typed:

  * the 17 features + each one's contracted boundary
        ``groundtruth.runtime.reasoning_runtime``:
        ``_FACT_DECISION_CONTRACTS`` (10 FACT), ``_CAP_FACT_BINDING`` (7 CAP),
        ``feature_contract_for(f).commitment_boundary``
  * the registry row (``deliver_by`` / ``surface``)
        ``groundtruth.runtime.fact_registry.REGISTRY``
  * producer layer -> fact class
        ``gt_mini_patch._LAYER_TO_FACT_CLASS`` / ``_fact_identity_for_layer``
  * CAP byte-owner attribution
        ``gt_mini_patch._LANE_PROFILE_MEMBER_OWNERS`` + the row's own
        ``profile_member`` / ``feature_ids`` lineage.

THREE VOCABULARY TRAP (read before touching the on-time logic).  Three different
fields each LOOK like "the boundary" and none of them is interchangeable:

  ``contracted_boundary``  EVENT vocabulary   (task_start, search_result, file_view,
                                               edit_result, test_result, submit,
                                               failed_search, failure_obs, ...)
  ``surface``              SURFACE vocabulary (brief, post_search, post_view,
                                               post_edit, post_test, submit, steer)
  ``event_type``           the ledger's own free-form channel string, which mixes
                           surface names (post_view/post_edit), event names
                           (test_result), producer names (l3.contract) and "".

Comparing the wrong pair manufactures a false "0 on-time".  So on-time is scored
ONLY from ``contracted_boundary`` against an observed value that is itself in
``fact_registry.EVENTS``.  Missing either side => NOT-EVALUABLE.  Never guessed.
(Run 30225435976 proves the trap is live: its lineage rows carry
``required_event="file_view"`` vs ``actual_event="post_edit"`` -- an EVENT compared
against a SURFACE, which would read as "late" and be meaningless.)

Usage:
    python scripts/swebench/gt_feature_verdicts.py <artifacts_dir> [--json]
    python scripts/swebench/gt_feature_verdicts.py --run <run_id> [--json]

``--run`` reuses the download cache at ``D:/tmp/gt_run_check/<run_id>/`` when it
exists, then falls back to ``D:/tmp/gtrun4/``.  Either way the directory is walked
recursively for ``gt_runtime_ledger_*.jsonl``, so per-task subdirectories at any
depth work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------- bootstrap ---

def _bootstrap_import_path() -> Path:
    """Put the repo's ``src`` and ``artifact_deepswe`` on ``sys.path``."""
    repo = Path(__file__).resolve().parents[2]
    for sub in ("src", "artifact_deepswe"):
        candidate = repo / sub
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return repo


REPO_ROOT = _bootstrap_import_path()


# --------------------------------------------------------------- authorities ---

# Mistake-gated: the trigger is an agent MISTAKE or a rare task shape, so
# TRIGGER-ABSENT is the expected reading on a clean trajectory.  Exactly the six
# FACT rows whose decision context is an error/recovery/completion boundary or a
# net-new-file shape; their CAP byte owners inherit the gate through
# ``_CAP_FACT_BINDING`` (computed, not listed).
_MISTAKE_GATED_FACTS: dict[str, str] = {
    "covering_red": "needs an observed failure with a covering repository test",
    "newfile_precedent": "needs the task to require a net-new file",
    "recovery": "needs the agent to stall / loop / collapse coherence",
    "signature_delta": "needs the agent to change a signature that has callers",
    "submit_refusal": "needs the agent to submit with an unresolved observed RED",
    "syntax_result": "needs the agent to write a syntax error",
}

# Reason classes.  Reason-first, then outcome: the OUTCOME alone cannot decide,
# because e.g. ``suppressed_hidden_only`` carries both "the trigger evaluated
# false" (correct-quiet) and "a referee withheld a real payload" (arbitration).

# The producer ran and there was NOTHING to deliver -> correct-quiet, not a failure.
_NO_EVIDENCE_REASONS = frozenset(
    {
        # CORRECTED 2026-07-28.  This string does NOT mean "no covering test exists".
        # `_verification_plan_emission` (gt_mini_patch.py:11954) reaches its
        # `plan_none_produced` tail (:12080) only after the progressive plan RAN, and on
        # the independent producer path it is entered only when
        # `_covering_tests_for_symbols` already returned a NON-empty selection
        # (:11880 guard -> :11924 call).  So it conflates: every rung was GREEN, or the
        # unit rung was RED but UNATTRIBUTED (invariant (2) forbids delivering it), or a
        # syntax rung was RED but carried no renderable first error, or the only RED rung
        # was a non-deliverable kind.  "No covering test exists in GT's graph" is a
        # DIFFERENT string -- `no_covering_test_selected`, immediately below.
        "plan_none_produced",
        # The genuine CAPABILITY GAP: the empty-selection branch (:11880-11909).  GT's
        # graph knows no FACT-tier test->impl edge for the symbols the agent just edited,
        # so no covering test could even be attempted.  Correct-quiet as a VERDICT (there
        # was nothing to deliver) but it is the single most informative dark state
        # covering_red has, which is why it is carried as a sub-detail (see `verdict_detail`).
        "no_covering_test_selected",
        # Selection was NON-empty but no selected file exists on disk (:11678-11685).
        "covering_no_covering",
        # The covering test EXECUTED and was GREEN (:11721-11728, verdict "pass").
        "covering_pass",
        # A real RED the edit did not plausibly cause; the attribution gate withheld it
        # on purpose (:11755-11762).  Nothing legitimately deliverable.
        "covering_unattributable",
        "edit_opportunity",     # edit.syntax denominator marker (an edit happened)
        "clean",                # submit/completion gate ran, nothing to refuse
    }
)
_NO_EVIDENCE_PREFIXES = ("trigger_false",)  # e.g. trigger_false:clean_exit|checkers=...

# The producer could not run / could not render -> a real defect, candidate failure.
# `covering_none_produced` (:11767-11775) is the covering twin of `render_failed`: an
# ATTRIBUTED RED existed and the native renderer could not surface it, i.e. evidence was
# produced and the bytes never reached the model -- this tool's own definition of a
# DELIVERY-FAILURE.  It is graded like `render_failed` for exactly that parity.
_DEFECT_PREFIXES = (
    "engine_import_unavailable",
    "checker_raised",
    "render_failed",
    "provider_failed",
    "covering_none_produced",
)

# Referee arbitration keyed on the REASON rather than the outcome.  `suppressed_ack_failure`
# is emitted (gt_mini_patch.py:11789 / :12062 / :12242) at the exact call site where
# `_ss_ack_failure_suppresses` returned True -- i.e. the block WAS built and SS-ACK dropped it
# as a duplicate of an already-acknowledged failure identity.  That referee logs its OWN row
# with outcome `suppressed_duplicate` / reason `acknowledged_failure_identity`, which this
# reader already classes `arbitration`; grading the producer-side twin as a DEFECT made the
# SAME event read as both "a referee did its job" and "the only real failure".  Reason-first,
# and checked BEFORE `_DEFECT_PREFIXES` so re-adding the prefix there cannot silently
# resurrect the contradiction.
_ARBITRATION_REASONS = frozenset({"suppressed_ack_failure"})

# Referee arbitration: evidence EXISTED and a self-governing referee withheld it on
# purpose (novelty / dedup / dose / step-behind / provenance / gate).  Legitimate --
# explicitly NOT a delivery failure.
_ARBITRATION_OUTCOMES = frozenset({"suppressed_duplicate", "suppressed_budget"})

# Chars<=0 bookkeeping rows (ACK receipts, internal-only telemetry).  Neither
# evidence-withheld nor a failure.
_TELEMETRY_REASONS = frozenset({"ss_ack"})

_WRONG_PHASE_OUTCOMES = frozenset({"suppressed_wrong_phase"})
_PRODUCED_OUTCOMES = frozenset({"produced", "eligible"})

# A GATE DECISION outcome.  If a layer the map cannot resolve emits one of these,
# some gate-owned feature among the 17 acted and its verdict is UNDERSTATED here.
# Mechanical, not a hand-list of layers: the tool refuses to guess the owner and
# instead reports that its own attribution is incomplete.
_GATE_DECISION_OUTCOMES = frozenset(
    {"bounce_once", "allow", "submit_clean", "refused", "blocked", "provider_failed"}
)

_VERDICT_FIRED = "FIRED"
_VERDICT_ABSENT = "TRIGGER-ABSENT"
_VERDICT_FAILURE = "DELIVERY-FAILURE"
# TRIGGER-ABSENT used to absorb these two as well, which let a SUPPRESSION and a BLIND
# SPOT both read as correct quiet. `classify_reason` already distinguished them; only the
# published verdict did not.
_VERDICT_ARBITRATED = "ARBITRATED"          # evidence produced, a referee withheld it
_VERDICT_UNINSTRUMENTED = "NO-INSTRUMENTATION"  # no ledger row at all -> not evaluable

_RUN_CACHE_ROOTS = ("D:/tmp/gt_run_check/{run}", "D:/tmp/gtrun4")


@dataclass
class FeatureRow:
    feature_id: str
    kind: str                       # FACT | CAP
    bound_fact: str                 # CAP -> its FACT; FACT -> itself
    contracted_boundary: str        # FeatureContract.commitment_boundary (EVENT vocab)
    registry_surface: str
    mistake_gated: bool
    gate_note: str
    verdict: str = ""
    # The dominant REASON behind `verdict`.  Six distinct engineering states publish as one
    # TRIGGER-ABSENT; this is what makes TRIGGER-ABSENT(no_covering_test_selected) -- GT's
    # graph knows no covering test, a capability gap -- readable apart from
    # TRIGGER-ABSENT(covering_pass) -- the test ran and was green.  Deliberately a
    # SUB-DETAIL and not a new top-level verdict: `verdict` stays inside the existing set so
    # the summary counters and the split pinned by
    # tests/swebench/test_verdict_funnel_splits_trigger_absent_20260727.py keep their meaning.
    verdict_detail: str = ""
    delivered: int = 0
    opportunities: int = 0
    terminal_dispositions: int = 0
    normalized_terminal_states: Counter = field(default_factory=Counter)
    delivered_chars: int = 0
    tasks_fired: set[str] = field(default_factory=set)
    attribution: str = "none"
    on_time: str = ""
    seen_event_types: Counter = field(default_factory=Counter)
    reason_classes: Counter = field(default_factory=Counter)
    reason_detail: Counter = field(default_factory=Counter)
    downgraded: int = 0             # outcome=delivered but chars<=0 (NOT a delivery)
    evidence: str = ""
    inherited: bool = False         # CAP counts copied from its bound FACT


def load_feature_universe() -> list[FeatureRow]:
    """The 17 DIRECT rows, straight from the runtime's canonical contracts."""
    from groundtruth.runtime import fact_registry
    from groundtruth.runtime.reasoning_runtime import (
        _CAP_FACT_BINDING,
        _FACT_DECISION_CONTRACTS,
        feature_contract_for,
    )

    rows: list[FeatureRow] = []
    for feature_id in sorted(_FACT_DECISION_CONTRACTS):
        rows.append(_build_row(feature_id, "FACT", feature_id, feature_contract_for,
                               fact_registry))
    for feature_id in sorted(_CAP_FACT_BINDING):
        rows.append(_build_row(feature_id, "CAP", _CAP_FACT_BINDING[feature_id],
                               feature_contract_for, fact_registry))
    if len(rows) != 17:
        raise SystemExit(
            f"feature universe drifted: expected 17 DIRECT rows, got {len(rows)}"
        )
    return rows


def _build_row(feature_id: str, kind: str, bound_fact: str, contract_for,
               fact_registry) -> FeatureRow:
    contract = contract_for(feature_id)
    if contract is None:
        raise SystemExit(f"no feature contract for {feature_id!r}")
    registration = fact_registry.REGISTRY.get(bound_fact)
    return FeatureRow(
        feature_id=feature_id,
        kind=kind,
        bound_fact=bound_fact,
        contracted_boundary=contract.commitment_boundary,
        registry_surface=getattr(registration, "surface", "?"),
        mistake_gated=bound_fact in _MISTAKE_GATED_FACTS,
        gate_note=_MISTAKE_GATED_FACTS.get(bound_fact, ""),
    )


# ------------------------------------------------------------------- ledgers ---

def resolve_artifacts_dir(args: argparse.Namespace) -> Path:
    if args.artifacts_dir:
        path = Path(args.artifacts_dir)
        if not path.is_dir():
            raise SystemExit(f"not a directory: {path}")
        return path
    candidates = [Path(t.format(run=args.run)) for t in _RUN_CACHE_ROOTS]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("gt_runtime_ledger_*.jsonl")):
            # A cache root that does not carry the run id cannot prove it holds THIS
            # run.  Serve it (the caller asked for the fallback) but never let it be
            # mistaken for run-keyed artifacts.
            if str(args.run) not in str(candidate):
                sys.stderr.write(
                    f"WARNING: {candidate} is not keyed by run {args.run}; its "
                    "artifacts may belong to a DIFFERENT run. Verify before citing.\n"
                )
            return candidate
    raise SystemExit(
        "no ledgers found for run "
        f"{args.run} in: {', '.join(str(c) for c in candidates)}"
    )


def iter_ledger_rows(root: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(task_id, row)`` for every ledger line under ``root``."""
    for path in sorted(root.rglob("gt_runtime_ledger_*.jsonl")):
        task_id = path.stem[len("gt_runtime_ledger_"):] or path.parent.name
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(row, dict):
                    yield task_id, row


# --------------------------------------------------------------- attribution ---

def attribute_row(row: dict[str, Any], layer_to_fact, fact_identity_for_layer,
                  delivery_facts: frozenset[str]) -> tuple[str | None, str]:
    """Return ``(fact_class, how)`` for a ledger row, or ``(None, reason)``.

    Correct-or-quiet: an unmapped layer resolves to nothing rather than a guess.
    """
    stamped = row.get("fact_class")
    if isinstance(stamped, str) and stamped in delivery_facts:
        return stamped, "row.fact_class"
    layer = str(row.get("layer") or "").strip()
    mapped, _boundary = fact_identity_for_layer(layer)
    if mapped and mapped in delivery_facts:
        return mapped, "layer_map"
    if mapped:
        return None, f"layer_map->{mapped}(not a delivery FACT)"
    return None, "unmapped_layer"


def cap_owners_in_row(row: dict[str, Any], lane_owners: dict[str, str],
                      cap_ids: frozenset[str]) -> set[str]:
    """CAP byte owners this row NAMES (never inferred from the bound FACT)."""
    owners: set[str] = set()
    member = row.get("profile_member")
    if isinstance(member, str) and member in cap_ids:
        owners.add(member)
    lane = lane_owners.get(str(row.get("layer") or "").strip())
    if lane in cap_ids:
        owners.add(lane)
    for entry in row.get("feature_ids") or ():
        if isinstance(entry, dict):
            fid = entry.get("feature_id")
            if isinstance(fid, str) and fid in cap_ids:
                owners.add(fid)
    # Canonical provider deliveries carry ownership inside each physical
    # evidence-lineage entry. Re-check owner->FACT authorization here; a row merely
    # naming an owner is not enough to mint attribution.
    try:
        from groundtruth.runtime.feature_lineage import CAP_BYTE_OWNER_MECHANISMS

        for entry in (
            row.get("evidence_lineage") or ()
            if _canonical_row_has_seal(row)
            else ()
        ):
            if not isinstance(entry, dict):
                continue
            fact_class = entry.get("fact_class")
            claimed = entry.get("cap_owners")
            if not isinstance(fact_class, str) or not isinstance(claimed, list):
                continue
            for owner in claimed:
                if not isinstance(owner, str) or owner not in cap_ids:
                    continue
                mechanism = CAP_BYTE_OWNER_MECHANISMS.get(owner)
                authorized_facts = {
                    binding.fact_class
                    for binding in mechanism.bindings
                    if binding.fact_class is not None
                } if mechanism is not None else set()
                if fact_class in authorized_facts:
                    owners.add(owner)
    except Exception:
        pass
    return owners


def _canonical_row_has_seal(row: dict[str, Any]) -> bool:
    seal = row.get("content_sha256_16")
    return (
        str(row.get("layer") or "") == "canonical.provider_delivery"
        and str(row.get("outcome") or "") == "delivered"
        and isinstance(seal, str)
        and len(seal) == 16
        and all(char in "0123456789abcdef" for char in seal.lower())
    )


def canonical_fact_classes_in_row(
    row: dict[str, Any],
    delivery_facts: frozenset[str],
) -> tuple[str, ...]:
    """Registered FACTs physically carried by one canonical provider delivery."""
    if not _canonical_row_has_seal(row):
        return ()
    lineage = row.get("evidence_lineage")
    if not isinstance(lineage, list):
        return ()
    return tuple(sorted({
        str(entry.get("fact_class"))
        for entry in lineage
        if isinstance(entry, dict)
        and isinstance(entry.get("fact_class"), str)
        and entry.get("fact_class") in delivery_facts
    }))


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def semantic_receipt_status(
    row: dict[str, Any],
    delivery_facts: frozenset[str],
) -> tuple[str, tuple[str, ...]]:
    """Validate v1 semantic proof when a canonical row claims to carry it.

    Older saved rows remain readable as ``legacy-unmeasured``. A current row
    that advertises receipts but fails any join is invalid and cannot earn
    semantic FIRED credit.
    """
    if (
        "semantic_receipts" not in row
        and "semantic_receipts_complete" not in row
        and "task_anchor" not in row
    ):
        return "legacy-unmeasured", ()
    anchor = row.get("task_anchor")
    if (
        not isinstance(anchor, dict)
        or anchor.get("configured") is not True
        or anchor.get("verbatim_text_present") is not True
        or not isinstance(anchor.get("task_sha256"), str)
        or len(anchor["task_sha256"]) != 64
    ):
        return "invalid", ("task_anchor",)
    receipts = row.get("semantic_receipts")
    evidence_ids = row.get("evidence_ids")
    lineage = row.get("evidence_lineage")
    if (
        row.get("semantic_receipts_complete") is not True
        or not isinstance(receipts, list)
        or not isinstance(evidence_ids, list)
        or not isinstance(lineage, list)
        or len(receipts) != len(evidence_ids)
        or len(lineage) != len(evidence_ids)
    ):
        return "invalid", ("receipt_cardinality",)
    reasons: list[str] = []
    fact_classes: list[str] = []
    for index, receipt in enumerate(receipts):
        lineage_item = lineage[index]
        if not isinstance(receipt, dict) or not isinstance(lineage_item, dict):
            reasons.append("receipt_shape")
            continue
        fact_class = str(receipt.get("fact_class") or "")
        if (
            receipt.get("evidence_id") != evidence_ids[index]
            or fact_class not in delivery_facts
            or receipt.get("feature_id") != fact_class
            or not isinstance(receipt.get("producer_id"), str)
            or not receipt.get("producer_id")
            or lineage_item.get("fact_class") != fact_class
            or receipt.get("candidate_id")
            != str(lineage_item.get("candidate_id") or "")
            or receipt.get("authorized_cap_owners")
            != lineage_item.get("cap_owners")
        ):
            reasons.append("identity_join")
            continue
        claim = receipt.get("claim")
        action = receipt.get("actionable_consequence")
        provenance = receipt.get("provenance")
        revision = receipt.get("revision")
        if (
            not isinstance(claim, str)
            or not isinstance(action, str)
            or not isinstance(provenance, list)
            or not isinstance(revision, dict)
            or receipt.get("claim_sha256")
            != hashlib.sha256(claim.encode("utf-8")).hexdigest()
            or receipt.get("actionable_consequence_sha256")
            != hashlib.sha256(action.encode("utf-8")).hexdigest()
            or receipt.get("provenance_hash")
            != _canonical_json_hash(provenance)
            or receipt.get("repository_revision")
            != revision.get("repository_content")
            or receipt.get("graph_revision") != revision.get("graph")
            or receipt.get("intended_action") != action
        ):
            reasons.append("semantic_hash")
            continue
        state_vector = {
            "claim": claim,
            "actionable_consequence": action,
            "revision": revision,
            "fresh": receipt.get("fresh"),
            "superseded": receipt.get("superseded"),
            "lifecycle": receipt.get("lifecycle"),
        }
        if (
            receipt.get("state_vector_hash")
            != _canonical_json_hash(state_vector)
            or receipt.get("fresh") is not True
            or receipt.get("superseded") is not False
            or not isinstance(receipt.get("authority"), str)
            or not isinstance(receipt.get("grade"), str)
            or not isinstance(receipt.get("observed_substrates"), list)
            or not isinstance(receipt.get("revision_dependencies"), list)
            or not isinstance(receipt.get("lifecycle_stage"), str)
        ):
            reasons.append("state_vector")
            continue
        fact_classes.append(fact_class)
    if reasons or len(fact_classes) != len(receipts):
        return "invalid", tuple(sorted(set(reasons)))
    return "valid", tuple(sorted(set(fact_classes)))


def classify_reason(row: dict[str, Any]) -> tuple[str, str]:
    """Return ``(class, detail)`` for a NON-delivered row."""
    outcome = str(row.get("outcome") or "")
    reason = str(row.get("reason") or "")
    head = reason.split(":", 1)[0].split("|", 1)[0]

    if reason in _NO_EVIDENCE_REASONS or head in _NO_EVIDENCE_REASONS:
        return "no_evidence", reason or outcome
    if any(reason.startswith(p) for p in _NO_EVIDENCE_PREFIXES):
        return "no_evidence", head
    if reason in _ARBITRATION_REASONS or head in _ARBITRATION_REASONS:
        return "arbitration", reason
    if any(reason.startswith(p) for p in _DEFECT_PREFIXES):
        return "defect", reason
    if outcome in _WRONG_PHASE_OUTCOMES:
        return "wrong_phase", reason or outcome
    if reason in _TELEMETRY_REASONS:
        return "telemetry", reason
    if outcome in _ARBITRATION_OUTCOMES or reason.startswith("ss_"):
        return "arbitration", reason or outcome
    if outcome == "suppressed_internal_only":
        return "telemetry", reason or outcome
    if outcome in _PRODUCED_OUTCOMES:
        return "produced", reason or outcome
    if outcome == "delivered":
        return "downgraded", f"{outcome}(chars<=0)"
    if outcome in {"submit_clean", "allow"}:
        return "no_evidence", reason or outcome
    return "other", reason or outcome


# ------------------------------------------------------------------ on-time ---

def observed_event(row: dict[str, Any], events: frozenset[str]) -> str | None:
    """An observed boundary in EVENT vocabulary, or ``None``.

    ``event_type`` and ``surface`` are deliberately NOT consulted: they are
    different vocabularies and comparing them to ``contracted_boundary``
    manufactures a false "off-boundary" verdict.
    """
    for key in ("observed_boundary", "actual_event"):
        value = row.get(key)
        if isinstance(value, str) and value in events:
            return value
    return None


# ---------------------------------------------------------------- evaluation ---

def evaluate(root: Path) -> dict[str, Any]:
    from groundtruth.runtime import fact_registry
    from groundtruth.runtime import feature_lineage
    from gt_mini_patch import (  # type: ignore[import-not-found]
        _LANE_PROFILE_MEMBER_OWNERS,
        _LAYER_TO_FACT_CLASS,
        _fact_identity_for_layer,
    )

    features = load_feature_universe()
    facts = {f.feature_id: f for f in features if f.kind == "FACT"}
    caps = [f for f in features if f.kind == "CAP"]
    cap_ids = frozenset(feature_lineage.CAP_BYTE_OWNER_IDS)
    delivery_facts = frozenset(facts)
    events = fact_registry.EVENTS

    tasks: set[str] = set()
    total_rows = 0
    unattributed: Counter = Counter()
    unattributed_delivered: Counter = Counter()
    #: Delivered rows this reader cannot attribute BY LAYER but which ARE attributed by
    #: their nested `evidence_lineage`. Reported separately so they stop reading as a bug.
    lineage_attributed_delivered: Counter = Counter()
    #: Trigger-census rows per FACT class: the boundary OCCURRED, independent of
    #: whether any producer then spoke. This is the denominator that separates
    #: "trigger never happened" from "producer abstained".
    trigger_opportunities: Counter = Counter()
    lifecycle_opportunities: Counter = Counter()
    lifecycle_fire_features: dict[str, str] = {}
    lifecycle_fire_observations: dict[str, str] = {}
    lifecycle_fire_candidates: dict[str, set[str]] = defaultdict(set)
    lifecycle_boundaries: dict[tuple[str, str], set[str]] = defaultdict(set)
    lifecycle_disposition_ids: set[str] = set()
    lifecycle_dispositions: dict[str, str] = {}
    delivered_fire_ids: set[str] = set()
    delivered_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    canonical_delivery_feature_instances: list[
        tuple[str, str, frozenset[str]]
    ] = []
    legacy_generic_disposition_ids: set[str] = set()
    invalid_lifecycle_rows: Counter = Counter()
    semantic_receipt_integrity: Counter = Counter()
    unresolved_gate: Counter = Counter()
    cap_direct: dict[str, dict[str, Any]] = {
        cap.feature_id: {"delivered": 0, "tasks": set(), "how": set(),
                         "event_types": Counter()}
        for cap in caps
    }
    boundary_stamped = 0
    on_time_hits: Counter = Counter()

    for task_id, row in iter_ledger_rows(root):
        tasks.add(task_id)
        total_rows += 1
        outcome = str(row.get("outcome") or "")
        try:
            chars = int(row.get("chars_delivered") or 0)
        except (TypeError, ValueError):
            chars = 0
        is_delivery = outcome == "delivered" and chars > 0
        event_type = str(row.get("event_type") or "") or "(none)"
        semantic_status = "not-applicable"
        semantic_facts: tuple[str, ...] = ()
        if _canonical_row_has_seal(row):
            semantic_status, semantic_facts = semantic_receipt_status(
                row,
                delivery_facts,
            )
            semantic_receipt_integrity[semantic_status] += 1
        attribution_row = row
        if semantic_status == "invalid":
            attribution_row = dict(row)
            attribution_row["evidence_lineage"] = []

        # CAP byte owners are credited ONLY when the row names them.
        row_cap_owners = (
            set()
            if semantic_status == "invalid"
            else cap_owners_in_row(
                attribution_row,
                _LANE_PROFILE_MEMBER_OWNERS,
                cap_ids,
            )
        )
        for owner in row_cap_owners:
            if owner not in cap_direct:
                continue
            how = []
            if row.get("profile_member") == owner:
                how.append("profile_member")
            if _LANE_PROFILE_MEMBER_OWNERS.get(str(row.get("layer") or "")) == owner:
                how.append("lane_owner")
            if any(isinstance(e, dict) and e.get("feature_id") == owner
                   for e in row.get("feature_ids") or ()):
                how.append("feature_ids")
            if any(
                isinstance(e, dict)
                and e.get("fact_class")
                and owner in (e.get("cap_owners") or ())
                for e in row.get("evidence_lineage") or ()
            ):
                how.append("evidence_lineage.cap_owners")
            if is_delivery:
                cap_direct[owner]["delivered"] += 1
                cap_direct[owner]["tasks"].add(task_id)
                cap_direct[owner]["event_types"][event_type] += 1
                cap_direct[owner]["how"].update(how)

        # TRIGGER CENSUS: a host-side row (chars=0, outcome="evaluated") stating that a
        # fact's boundary OCCURRED, emitted independently of whether any producer then
        # spoke. Counted BEFORE attribution because it is not a delivery and must never
        # land in the delivered/unattributed accounting.
        if row.get("layer") == "feature.trigger_opportunity":
            _census_fact = str(row.get("fact_class") or "").strip()
            if _census_fact:
                trigger_opportunities[_census_fact] += 1
            continue
        if row.get("layer") == "feature.lifecycle_opportunity":
            feature_id = str(row.get("feature_id") or "").strip()
            observation_id = str(row.get("observation_id") or "").strip()
            boundary = str(
                row.get("lifecycle_boundary")
                or row.get("required_event")
                or ""
            ).strip()
            fire_id = str(
                row.get("feature_fire_id")
                or row.get("lifecycle_opportunity_id")
                or ""
            ).strip()
            try:
                from groundtruth.runtime.trigger_opportunity import (
                    lifecycle_opportunity_id,
                )

                expected_fire_id = lifecycle_opportunity_id(
                    observation_id,
                    boundary,
                    feature_id,
                )
            except (TypeError, ValueError):
                expected_fire_id = ""
            if (
                feature_id not in {feature.feature_id for feature in features}
                or not expected_fire_id
                or fire_id != expected_fire_id
            ):
                invalid_lifecycle_rows["invalid_opportunity_identity"] += 1
                continue
            lifecycle_opportunities[feature_id] += 1
            lifecycle_fire_features[fire_id] = feature_id
            lifecycle_fire_observations[fire_id] = observation_id
            lifecycle_boundaries[(observation_id, feature_id)].add(boundary)
            continue
        if row.get("schema") == "gt.feature_fire_disposition.v1":
            fire_ids = row.get("feature_fire_ids")
            if not isinstance(fire_ids, list):
                invalid_lifecycle_rows["invalid_disposition_ids"] += 1
                continue
            normalized_ids = {
                str(fire_id)
                for fire_id in fire_ids
                if isinstance(fire_id, str) and fire_id
            }
            per_feature = row.get("feature_dispositions")
            if isinstance(per_feature, list):
                seen_here: set[str] = set()
                for item in per_feature:
                    if not isinstance(item, dict):
                        invalid_lifecycle_rows[
                            "invalid_feature_disposition"
                        ] += 1
                        continue
                    fire_id = str(item.get("feature_fire_id") or "")
                    disposition = str(item.get("disposition") or "")
                    if (
                        not fire_id
                        or fire_id not in normalized_ids
                        or fire_id in seen_here
                        or disposition not in {
                            "abstained",
                            "available",
                            "produced",
                            "permitted",
                            "deferred",
                            "staged",
                            "withheld",
                            "blocked",
                            "delivered",
                        }
                    ):
                        invalid_lifecycle_rows[
                            "invalid_feature_disposition"
                        ] += 1
                        continue
                    seen_here.add(fire_id)
                    prior = lifecycle_dispositions.get(fire_id)
                    if prior is not None and prior != disposition:
                        invalid_lifecycle_rows[
                            "conflicting_feature_disposition"
                        ] += 1
                        continue
                    lifecycle_dispositions[fire_id] = disposition
                    for candidate_key in (
                        "produced_candidate_ids",
                        "available_candidate_ids",
                    ):
                        candidates = item.get(candidate_key)
                        if isinstance(candidates, list):
                            lifecycle_fire_candidates[fire_id].update(
                                str(candidate)
                                for candidate in candidates
                                if isinstance(candidate, str) and candidate
                            )
                missing = normalized_ids - seen_here
                if missing:
                    invalid_lifecycle_rows[
                        "missing_feature_disposition"
                    ] += len(missing)
            else:
                # Backward-compatible reader for older saved artifacts.  The
                # row-level disposition proves termination but cannot support a
                # per-feature trigger verdict.  In particular, one old
                # ``produced`` row can cover several fire ids even when only one
                # FACT produced bytes.  Preserve denominator integrity without
                # resurrecting that false cross-feature credit.
                legacy_generic_disposition_ids.update(normalized_ids)
            lifecycle_disposition_ids.update(normalized_ids)
            continue

        # A canonical capsule is one physical row that can carry several independently
        # typed FACT records. Its constant layer cannot identify any one of them, so
        # credit the validated nested lineage rather than dropping the entire capsule as
        # "lineage-attributed but uncounted".
        canonical_facts = canonical_fact_classes_in_row(row, delivery_facts)
        if semantic_status == "invalid":
            canonical_facts = ()
        elif semantic_status == "valid" and set(canonical_facts) != set(
            semantic_facts
        ):
            semantic_receipt_integrity["valid"] -= 1
            semantic_receipt_integrity["invalid"] += 1
            semantic_receipt_integrity["identity_set_mismatch"] += 1
            semantic_status = "invalid"
            canonical_facts = ()
        if is_delivery and canonical_facts:
            delivery_observation_id = str(
                row.get("observation_id") or ""
            ).strip()
            for canonical_fact in canonical_facts:
                target = facts[canonical_fact]
                target.delivered += 1
                target.delivered_chars += chars
                target.tasks_fired.add(task_id)
                target.seen_event_types[event_type] += 1
                delivery_evidence_ids = row.get("evidence_ids") or ()
                for index, entry in enumerate(
                    row.get("evidence_lineage") or ()
                ):
                    if (
                        not isinstance(entry, dict)
                        or entry.get("fact_class") != canonical_fact
                    ):
                        continue
                    candidate_id = str(entry.get("candidate_id") or "")
                    evidence_id = (
                        str(delivery_evidence_ids[index])
                        if (
                            isinstance(delivery_evidence_ids, list)
                            and index < len(delivery_evidence_ids)
                            and isinstance(
                                delivery_evidence_ids[index],
                                str,
                            )
                        )
                        else ""
                    )
                    sealed_identities = {
                        identity
                        for identity in (candidate_id, evidence_id)
                        if identity
                    }
                    if not sealed_identities:
                        continue
                    delivered_candidates[
                        (canonical_fact, delivery_observation_id)
                    ].update(sealed_identities)
                    canonical_delivery_feature_instances.append(
                        (
                            canonical_fact,
                            delivery_observation_id,
                            frozenset(sealed_identities),
                        )
                    )
                    for owner in entry.get("cap_owners") or ():
                        if owner in row_cap_owners:
                            delivered_candidates[
                                (str(owner), delivery_observation_id)
                            ].update(sealed_identities)
                            canonical_delivery_feature_instances.append(
                                (
                                    str(owner),
                                    delivery_observation_id,
                                    frozenset(sealed_identities),
                                )
                            )
                # A canonical capsule's event_type is intentionally the constant
                # provider-delivery vocabulary. Its timing authority is the shared
                # canonical observation id: a matching lifecycle row proves this
                # feature was released inside one of its registered windows.
                if lifecycle_boundaries.get(
                    (delivery_observation_id, canonical_fact)
                ):
                    boundary_stamped += 1
                    on_time_hits[(canonical_fact, "on_time")] += 1
                contracted = row.get("contracted_boundary") or row.get(
                    "gt_audit_contracted_boundary"
                )
                if isinstance(contracted, str) and contracted:
                    boundary_stamped += 1
                    seen = observed_event(row, events)
                    if seen is None:
                        on_time_hits[
                            (canonical_fact, "no_event_vocab_observation")
                        ] += 1
                    elif seen == contracted:
                        on_time_hits[(canonical_fact, "on_time")] += 1
                    else:
                        on_time_hits[
                            (canonical_fact, f"off_boundary:{seen}")
                        ] += 1
            lineage_attributed_delivered[
                f"{row.get('layer')}|{event_type}"
            ] += 1
            continue
        if is_delivery and semantic_status == "invalid":
            unattributed_delivered["canonical.semantic_receipt_invalid"] += 1
            continue

        fact_class, how = attribute_row(row, _LAYER_TO_FACT_CLASS,
                                        _fact_identity_for_layer, delivery_facts)
        if fact_class is None:
            key = f"{row.get('layer')}|{outcome}|{how}"
            unattributed[key] += 1
            if is_delivery:
                # DO NOT REPORT A CANONICAL CAPSULE ROW AS UNATTRIBUTED. The canonical
                # plane writes ONE delivery row per model call whose `layer` is the
                # constant "canonical.provider_delivery"; its attribution lives in the
                # NESTED `evidence_lineage` entries, each carrying its own `fact_class`
                # and `cap_owners`. `gt_feature_metrics` attributes it exactly that way
                # and documents why the layer cannot be used: matching a lane layer on a
                # canonical row would mean writing a layer the record does not have.
                #
                # Measured on run 30390877219: 10 of the 14 remaining "UNATTRIBUTED
                # DELIVERIES" were canonical rows that ARE attributed, just by a
                # mechanism this reader does not consult. Listing them here sends the
                # next reader chasing a non-bug -- the same class of harm as a zero that
                # means "not measured" sharing a rendering with "measured, found none".
                _lineage = row.get("evidence_lineage")
                _lineage_attributed = isinstance(_lineage, list) and any(
                    isinstance(e, dict) and e.get("fact_class") for e in _lineage
                )
                if _lineage_attributed:
                    lineage_attributed_delivered[
                        f"{row.get('layer')}|{event_type}"] += 1
                else:
                    unattributed_delivered[f"{row.get('layer')}|{event_type}"] += 1
            if outcome in _GATE_DECISION_OUTCOMES:
                unresolved_gate[f"{row.get('layer')}|{outcome}"] += 1
            continue

        target = facts[fact_class]
        if is_delivery:
            target.delivered += 1
            target.delivered_chars += chars
            target.tasks_fired.add(task_id)
            target.seen_event_types[event_type] += 1
            # The GRADED key wins when a row carries a genuinely registered identity;
            # `gt_audit_*` is the diagnostic namespace the seam derives from the layer
            # name. They are kept separate on purpose: `fact_class` seats the
            # attestation truth join that feeds the promotion authority, so deriving it
            # would inflate a proof number to make this table easier to build.
            contracted = row.get("contracted_boundary") or row.get(
                "gt_audit_contracted_boundary"
            )
            if isinstance(contracted, str) and contracted:
                boundary_stamped += 1
                seen = observed_event(row, events)
                if seen is None:
                    on_time_hits[(fact_class, "no_event_vocab_observation")] += 1
                elif seen == contracted:
                    on_time_hits[(fact_class, "on_time")] += 1
                else:
                    on_time_hits[(fact_class, f"off_boundary:{seen}")] += 1
        else:
            klass, detail = classify_reason(row)
            target.reason_classes[klass] += 1
            target.reason_detail[f"{klass}:{detail}"] += 1
            if klass == "downgraded":
                target.downgraded += 1

    delivered_fire_ids.update(
        fire_id
        for fire_id, feature_id in lifecycle_fire_features.items()
        if (
            lifecycle_fire_candidates.get(fire_id)
            and lifecycle_fire_candidates[fire_id].intersection(
                delivered_candidates.get(
                    (
                        feature_id,
                        lifecycle_fire_observations.get(fire_id, ""),
                    ),
                    set(),
                )
            )
        )
    )
    normalized_by_fire: dict[str, str] = {}
    terminal_map = {
        "abstained": "INELIGIBLE",
        "permitted": "APPLIED_QUIET",
        "deferred": "SUPPRESSED",
        "withheld": "SUPPRESSED",
        # Produced/available evidence entered the one-dose arbiter but was not
        # selected into this provider request. That is arbitration, not a
        # provider failure. Once a candidate is STAGED, absence of exact
        # provider-terminal proof is a delivery failure.
        "produced": "SUPPRESSED",
        "available": "SUPPRESSED",
        "staged": "DELIVERY_FAILURE",
        "blocked": "DELIVERY_FAILURE",
        "delivered": "DELIVERY_FAILURE",
    }
    for fire_id in lifecycle_fire_features:
        if fire_id in delivered_fire_ids:
            normalized_by_fire[fire_id] = "DELIVERED"
        else:
            normalized_by_fire[fire_id] = terminal_map.get(
                lifecycle_dispositions.get(fire_id, ""),
                "FAULT",
            )
    joined_delivery_by_feature: Counter = Counter()
    unjoined_delivery_by_feature: Counter = Counter()
    for feature_id, observation_id, identities in (
        canonical_delivery_feature_instances
    ):
        joined = any(
            lifecycle_fire_features[fire_id] == feature_id
            and lifecycle_fire_observations.get(fire_id) == observation_id
            and bool(
                identities.intersection(
                    lifecycle_fire_candidates.get(fire_id, set())
                )
            )
            for fire_id in lifecycle_fire_features
        )
        (
            joined_delivery_by_feature
            if joined
            else unjoined_delivery_by_feature
        )[feature_id] += 1
    joined_delivery_instances = sum(joined_delivery_by_feature.values())

    # ---- FACT verdicts -------------------------------------------------------
    for fact in facts.values():
        fact.opportunities = (
            lifecycle_opportunities.get(fact.feature_id, 0)
            or trigger_opportunities.get(fact.bound_fact, 0)
        )
        fact.terminal_dispositions = sum(
            1
            for fire_id, feature_id in lifecycle_fire_features.items()
            if feature_id == fact.feature_id
            and fire_id in lifecycle_disposition_ids
        )
        fact.normalized_terminal_states = Counter(
            normalized_by_fire[fire_id]
            for fire_id, feature_id in lifecycle_fire_features.items()
            if feature_id == fact.feature_id
        )
        _decide(
            fact,
            on_time_hits,
            boundary_stamped,
            trigger_opportunities,
            lifecycle_opportunities,
            Counter(
                lifecycle_dispositions.get(fire_id, "")
                for fire_id, feature_id in lifecycle_fire_features.items()
                if feature_id == fact.feature_id
                and fire_id in lifecycle_dispositions
            ),
        )

    # ---- CAP verdicts --------------------------------------------------------
    for cap in caps:
        cap.opportunities = lifecycle_opportunities.get(cap.feature_id, 0)
        cap.terminal_dispositions = sum(
            1
            for fire_id, feature_id in lifecycle_fire_features.items()
            if feature_id == cap.feature_id
            and fire_id in lifecycle_disposition_ids
        )
        cap.normalized_terminal_states = Counter(
            normalized_by_fire[fire_id]
            for fire_id, feature_id in lifecycle_fire_features.items()
            if feature_id == cap.feature_id
        )
        direct = cap_direct[cap.feature_id]
        bound = facts[cap.bound_fact]
        cap_dispositions = Counter(
            lifecycle_dispositions.get(fire_id, "")
            for fire_id, feature_id in lifecycle_fire_features.items()
            if feature_id == cap.feature_id
            and fire_id in lifecycle_dispositions
        )
        if direct["delivered"] > 0:
            cap.verdict = _VERDICT_FIRED
            cap.delivered = direct["delivered"]
            cap.tasks_fired = set(direct["tasks"])
            cap.seen_event_types = Counter(direct["event_types"])
            how = "+".join(sorted(direct["how"])) or "row"
            cap.attribution = f"byte-owner lineage: {how}"
            cap.verdict_detail = "delivered"
            cap.on_time = bound.on_time
            cap.evidence = ""
            continue
        if cap_dispositions:
            cap.attribution = "per-feature CAP lifecycle disposition"
            if cap_dispositions.get("deferred") or cap_dispositions.get("withheld"):
                cap.verdict = _VERDICT_ARBITRATED
                cap.verdict_detail = "lifecycle_arbitrated"
                cap.evidence = "CAP opportunity was explicitly deferred/withheld"
            elif any(
                cap_dispositions.get(name)
                for name in ("produced", "available", "staged", "blocked")
            ):
                cap.verdict = _VERDICT_FAILURE
                cap.verdict_detail = "cap_evidence_without_delivery"
                cap.evidence = (
                    "CAP-owned evidence existed, but no provider-bound delivery "
                    "row named this byte owner"
                )
            elif set(cap_dispositions) <= {"abstained", "permitted", ""}:
                cap.verdict = _VERDICT_ABSENT
                cap.verdict_detail = "correct_quiet"
                cap.evidence = (
                    "correct-quiet: CAP lifecycle evaluated and its authorized "
                    "byte-owner computation abstained"
                )
            else:
                cap.verdict = _VERDICT_UNINSTRUMENTED
                cap.verdict_detail = "unknown_terminal_disposition"
                cap.evidence = "CAP lifecycle termination was not classifiable"
            continue
        # No row names this CAP.  Fall back to its bound FACT, and SAY SO.  Every
        # inherited cell is marked '^' in the table so a bound-FACT count is never
        # read as proof that THIS capability owned the bytes.
        cap.verdict = bound.verdict
        cap.verdict_detail = bound.verdict_detail
        cap.inherited = True
        cap.delivered = bound.delivered
        cap.delivered_chars = bound.delivered_chars
        cap.tasks_fired = set(bound.tasks_fired)
        cap.seen_event_types = Counter(bound.seen_event_types)
        cap.on_time = bound.on_time
        cap.reason_classes = Counter(bound.reason_classes)
        cap.reason_detail = Counter(bound.reason_detail)
        cap.attribution = f"bound-FACT {cap.bound_fact} (no byte-owner lineage row)"
        if lifecycle_opportunities.get(cap.feature_id):
            cap.attribution += (
                f"; lifecycle evaluated "
                f"{lifecycle_opportunities[cap.feature_id]}x"
            )
        if bound.verdict == _VERDICT_FIRED:
            cap.evidence = (
                f"bytes reached the model as {cap.bound_fact} "
                f"({bound.delivered} delivery/ies) but no ledger row names this CAP "
                "-- CAP-specific attribution UNPROVEN, not a failure"
            )
        else:
            cap.evidence = bound.evidence

    return {
        "root": str(root),
        "tasks": sorted(tasks),
        "total_rows": total_rows,
        "features": features,
        "unattributed": unattributed,
        "unattributed_delivered": unattributed_delivered,
        "lineage_attributed_delivered": lineage_attributed_delivered,
        "unresolved_gate": unresolved_gate,
        "boundary_stamped": boundary_stamped,
        "on_time_hits": on_time_hits,
        "trigger_opportunities": trigger_opportunities,
        "lifecycle_opportunities": lifecycle_opportunities,
        "lifecycle_integrity": {
            "opportunity_ids": len(lifecycle_fire_features),
            "terminal_ids": len(
                set(lifecycle_fire_features) & lifecycle_disposition_ids
            ),
            "unterminated_ids": sorted(
                set(lifecycle_fire_features) - lifecycle_disposition_ids
            ),
            "orphan_terminal_ids": sorted(
                lifecycle_disposition_ids - set(lifecycle_fire_features)
            ),
            "legacy_generic_terminal_ids": sorted(
                legacy_generic_disposition_ids
            ),
            "invalid_rows": dict(invalid_lifecycle_rows),
            "normalized_terminal_counts": dict(
                Counter(normalized_by_fire.values())
            ),
            "missing_normalized_ids": sorted(
                set(lifecycle_fire_features) - set(normalized_by_fire)
            ),
        },
        "semantic_receipt_integrity": dict(semantic_receipt_integrity),
        "delivery_fire_join_integrity": {
            "canonical_feature_instances": len(
                canonical_delivery_feature_instances
            ),
            "exact_fire_candidate_joins": joined_delivery_instances,
            "unjoined_feature_instances": (
                len(canonical_delivery_feature_instances)
                - joined_delivery_instances
            ),
            "exact_joins_by_feature": dict(joined_delivery_by_feature),
            "unjoined_by_feature": dict(unjoined_delivery_by_feature),
        },
    }


def _decide(fact: FeatureRow, on_time_hits: Counter, boundary_stamped: int,
            trigger_opportunities: Counter | None = None,
            lifecycle_opportunities: Counter | None = None,
            lifecycle_dispositions: Counter | None = None) -> None:
    classes = fact.reason_classes
    if fact.delivered > 0:
        fact.verdict_detail = "delivered"
        fact.verdict = _VERDICT_FIRED
        fact.attribution = "layer/fact_class rows"
        bits = []
        if classes.get("arbitration"):
            bits.append(f"+{classes['arbitration']} arbitrated")
        if classes.get("produced"):
            bits.append(f"+{classes['produced']} produced")
        if fact.downgraded:
            bits.append(f"+{fact.downgraded} delivered-but-chars<=0 (NOT a delivery)")
        fact.evidence = "; ".join(bits)
    elif classes.get("wrong_phase") or classes.get("defect"):
        fact.verdict = _VERDICT_FAILURE
        fact.attribution = "layer/fact_class rows"
        fact.evidence = _top_detail(fact, ("wrong_phase", "defect"))
        fact.verdict_detail = _dominant_reason(fact, ("wrong_phase", "defect"))
    elif ((classes.get("produced") or classes.get("downgraded"))
            and not classes.get("arbitration")):
        fact.verdict = _VERDICT_FAILURE
        fact.attribution = "layer/fact_class rows"
        fact.evidence = (
            "produced/eligible (or delivered with chars<=0) but no real delivery and "
            "no arbitration explaining it: "
            + _top_detail(fact, ("produced", "downgraded"))
        )
        fact.verdict_detail = _dominant_reason(fact, ("produced", "downgraded"))
    elif classes.get("arbitration"):
        # Evidence EXISTED and a referee withheld it.  Explicitly NOT a delivery
        # failure (the referee is doing its job), and not literally "trigger absent"
        # either -- the three-verdict taxonomy has no fourth bucket, so it is
        # published as its OWN verdict: the trigger DID occur and evidence existed, so
        # calling it trigger-absent asserts something false about the trajectory.
        fact.verdict = _VERDICT_ARBITRATED
        fact.attribution = "layer/fact_class rows"
        fact.evidence = "ARBITRATED (evidence produced, referee withheld -- NOT a " \
            "delivery failure): " + _top_detail(fact, ("arbitration",))
        fact.verdict_detail = _dominant_reason(fact, ("arbitration",))
    else:
        # No rows at all is BLINDNESS, not quiet: the trigger may well have occurred and
        # nothing recorded it either way. Absence of evidence is not evidence of absence.
        fact.verdict = _VERDICT_ABSENT if classes else _VERDICT_UNINSTRUMENTED
        fact.attribution = "layer/fact_class rows" if classes else "no rows"
        if classes:
            fact.evidence = _top_detail(
                fact, ("no_evidence", "telemetry", "downgraded", "other")
            )
            fact.verdict_detail = _dominant_reason(
                fact, ("no_evidence", "telemetry", "downgraded", "other")
            )
        else:
            # SPLIT BLINDNESS FROM A MEASURED ABSTENTION, using the trigger census.
            #
            # "No producer row" has TWO causes that a run cannot otherwise tell apart:
            # the trigger's boundary never occurred (correct-quiet), or it DID occur and
            # the producer declined. The census (`feature.trigger_opportunity`, host-side,
            # chars=0) records the boundary independently of any producer, so a census row
            # for this fact class converts blindness into a MEASURED negative -- which is
            # the whole reason the denominator exists. Without this the census was
            # WRITE-ONLY: rows emitted and nothing reading them.
            _lifecycle_opps = (
                lifecycle_opportunities or Counter()
            ).get(fact.feature_id, 0)
            _physical_opps = (
                trigger_opportunities or Counter()
            ).get(fact.bound_fact, 0)
            _opps = _lifecycle_opps or _physical_opps
            if _opps:
                dispositions = lifecycle_dispositions or Counter()
                if dispositions.get("deferred") or dispositions.get("withheld"):
                    fact.verdict = _VERDICT_ARBITRATED
                    fact.attribution = "per-feature lifecycle disposition"
                    fact.evidence = (
                        f"evidence/window evaluated {_opps}x and commitment "
                        "control deferred or withheld it"
                    )
                    fact.verdict_detail = "lifecycle_arbitrated"
                elif any(
                    dispositions.get(name)
                    for name in ("produced", "available", "staged", "blocked")
                ):
                    fact.verdict = _VERDICT_FAILURE
                    fact.attribution = "per-feature lifecycle disposition"
                    fact.evidence = (
                        "per-feature evidence existed at a lifecycle window but "
                        "no provider-bound delivery row carried it"
                    )
                    fact.verdict_detail = "evidence_without_delivery"
                elif (
                    sum(dispositions.values()) >= _lifecycle_opps
                    and set(dispositions) <= {"abstained", "permitted", ""}
                ):
                    fact.verdict = _VERDICT_ABSENT
                    fact.attribution = "per-feature lifecycle disposition"
                    fact.evidence = (
                        f"correct-quiet: {_opps} lifecycle window(s) evaluated; "
                        "the feature-specific producer abstained and no action "
                        "was blocked"
                    )
                    fact.verdict_detail = "correct_quiet"
                else:
                    fact.evidence = (
                        f"NO producer row, but the trigger boundary OCCURRED {_opps}x "
                        f"({'lifecycle' if _lifecycle_opps else 'physical trigger'} "
                        "census) -- terminal per-feature cause is incomplete"
                    )
                    fact.verdict_detail = (
                        f"abstained_after_{_opps}_opportunities"
                    )
            else:
                fact.evidence = (
                    "no ledger row for this feature's producer layer(s), and no trigger "
                    "census row -- boundary occurrence itself is UNMEASURED"
                )
                fact.verdict_detail = "no_rows"

    # On-time is scored ONLY from contracted_boundary; anything else is a guess.
    hits = {k[1]: v for k, v in on_time_hits.items() if k[0] == fact.feature_id}
    if not hits:
        fact.on_time = "NOT-EVALUABLE(no contracted_boundary)"
    elif set(hits) == {"on_time"}:
        fact.on_time = f"ON-TIME {hits['on_time']}/{hits['on_time']}"
    elif set(hits) == {"no_event_vocab_observation"}:
        fact.on_time = "NOT-EVALUABLE(no EVENT-vocab observation)"
    else:
        total = sum(hits.values())
        fact.on_time = f"{hits.get('on_time', 0)}/{total} on-time; " + ",".join(
            f"{k}x{v}" for k, v in sorted(hits.items()) if k != "on_time"
        )


def _dominant_reason(fact: FeatureRow, classes: tuple[str, ...]) -> str:
    """The single most common REASON string behind the verdict, class prefix stripped.

    ``reason_detail`` keys are ``"<class>:<reason>"`` and the reason itself may carry its
    own ``:``/``|`` payload (``trigger_false:parsed|checkers=ast.parse``); only the reason
    HEAD is published, so the sub-detail is a stable, low-cardinality label.  Ties break on
    the reason string so the output is deterministic across runs.
    """
    tally: Counter = Counter()
    for detail, count in fact.reason_detail.items():
        klass, _, reason = detail.partition(":")
        if klass not in classes:
            continue
        head = reason.split(":", 1)[0].split("|", 1)[0].strip()
        tally[head or klass] += count
    if not tally:
        return ""
    return min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _top_detail(fact: FeatureRow, classes: tuple[str, ...], limit: int = 3) -> str:
    items = [
        (detail, count)
        for detail, count in fact.reason_detail.items()
        if detail.split(":", 1)[0] in classes
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{d} x{c}" for d, c in items[:limit]) or "-"


# -------------------------------------------------------------------- render ---

def verdict_cell(f: FeatureRow) -> str:
    """``VERDICT(dominant_reason)`` -- the verdict plus WHY, as one readable token.

    The verdict itself is untouched: six engineering states still publish the SAME
    top-level TRIGGER-ABSENT, which is what every counter and every existing consumer
    reads.  Only the parenthetical is new, and it is what tells the reader that
    ``TRIGGER-ABSENT(no_covering_test_selected)`` (GT's graph knows no covering test at
    all) is not ``TRIGGER-ABSENT(covering_pass)`` (a covering test ran and was green).
    """
    if not f.verdict_detail or f.verdict_detail == "delivered":
        return f.verdict
    return f"{f.verdict}({f.verdict_detail})"


def render_text(result: dict[str, Any]) -> str:
    features: list[FeatureRow] = result["features"]
    tasks: list[str] = result["tasks"]
    out: list[str] = []
    out.append(f"GT 17-DIRECT FEATURE VERDICTS  |  artifacts: {result['root']}")
    out.append(
        f"tasks: {len(tasks)}  ledger rows: {result['total_rows']}  "
        f"rows carrying contracted_boundary: {result['boundary_stamped']}"
    )
    lifecycle_integrity = result["lifecycle_integrity"]
    out.append(
        "lifecycle fires: "
        f"{lifecycle_integrity['terminal_ids']}/"
        f"{lifecycle_integrity['opportunity_ids']} terminal; "
        f"{len(lifecycle_integrity['unterminated_ids'])} unterminated; "
        f"{len(lifecycle_integrity['orphan_terminal_ids'])} orphan terminals; "
        f"{len(lifecycle_integrity['legacy_generic_terminal_ids'])} "
        "legacy-generic terminals; "
        f"{sum(lifecycle_integrity['invalid_rows'].values())} invalid rows"
    )
    normalized = lifecycle_integrity["normalized_terminal_counts"]
    out.append(
        "normalized terminals: "
        + (
            ", ".join(
                f"{state}={count}"
                for state, count in sorted(normalized.items())
            )
            if normalized
            else "none"
        )
    )
    semantic_integrity = result["semantic_receipt_integrity"]
    out.append(
        "canonical semantic receipts: "
        + (
            ", ".join(
                f"{state}={count}"
                for state, count in sorted(semantic_integrity.items())
                if count
            )
            if any(semantic_integrity.values())
            else "none"
        )
    )
    delivery_join = result["delivery_fire_join_integrity"]
    out.append(
        "delivery -> feature-fire candidate joins: "
        f"{delivery_join['exact_fire_candidate_joins']}/"
        f"{delivery_join['canonical_feature_instances']} exact; "
        f"{delivery_join['unjoined_feature_instances']} unjoined"
    )
    out.append("")

    header = (
        f"{'FEATURE':<20} {'KIND':<4} {'CONTRACTED':<14} {'VERDICT(why)':<48} "
        f"{'OPP':>5} {'TERM':>5} {'DELIV':>6} {'TASK':>5}  "
        f"{'BOUNDARIES SEEN (ledger event_type)':<34} "
        f"ON-TIME"
    )
    out.append(header)
    out.append("-" * len(header))
    for f in sorted(features, key=lambda r: (r.kind, r.feature_id)):
        seen = ", ".join(
            f"{k}x{v}" for k, v in sorted(f.seen_event_types.items()) if v
        ) or "-"
        if len(seen) > 34:
            seen = seen[:33] + "~"
        gate = "*" if f.mistake_gated else " "
        mark = "^" if f.inherited else " "
        cell = verdict_cell(f)
        if len(cell) > 48:
            cell = cell[:47] + "~"
        out.append(
            f"{f.feature_id + gate:<20} {f.kind:<4} {f.contracted_boundary:<14} "
            f"{cell:<48} {f.opportunities:>5} {f.terminal_dispositions:>5} "
            f"{str(f.delivered) + mark:>6} "
            f"{len(f.tasks_fired):>2}/{len(tasks):<2}  {seen:<34} {f.on_time}"
        )
    out.append("")
    out.append("* = MISTAKE-GATED: fires only on an agent mistake or a rare task shape.")
    out.append("  TRIGGER-ABSENT on these is CORRECT-QUIET evidence of a clean run, "
               "not a broken feature.")
    out.append("^ = counts INHERITED from the bound FACT; no ledger row names this CAP "
               "byte owner.")
    out.append("CONTRACTED is EVENT vocabulary (fact_registry.EVENTS); BOUNDARIES SEEN is "
               "the ledger's own")
    out.append("  event_type string (mixed surface/event/producer names). They are NOT "
               "comparable -- see ON-TIME.")
    out.append("")

    out.append("WHY (every non-FIRED row, plus caveats on FIRED rows):")
    for f in sorted(features, key=lambda r: (r.verdict != _VERDICT_FAILURE,
                                             r.verdict, r.kind, r.feature_id)):
        if f.verdict == _VERDICT_FIRED and not f.evidence and f.attribution.startswith(
                ("layer", "byte-owner")):
            continue
        gate = f"  [mistake-gated: {f.gate_note}]" if f.mistake_gated else ""
        out.append(f"  {f.feature_id} ({f.kind}) -> {verdict_cell(f)}")
        out.append(f"      attribution: {f.attribution}{gate}")
        if f.evidence:
            out.append(f"      evidence   : {f.evidence}")
    out.append("")

    counts = Counter(f.verdict for f in features)
    fired = counts.get(_VERDICT_FIRED, 0)
    absent = counts.get(_VERDICT_ABSENT, 0)
    failed = counts.get(_VERDICT_FAILURE, 0)
    arbitrated_n = counts.get(_VERDICT_ARBITRATED, 0)
    blind = counts.get(_VERDICT_UNINSTRUMENTED, 0)
    out.append(
        f"SUMMARY: {fired} FIRED / {absent} TRIGGER-ABSENT / {arbitrated_n} ARBITRATED"
        f" / {blind} NO-INSTRUMENTATION / {failed} DELIVERY-FAILURE"
        f" out of {len(features)}"
    )
    absent_why = Counter(f.verdict_detail or "(unlabelled)" for f in features
                         if f.verdict == _VERDICT_ABSENT)
    if len(absent_why) > 1:
        # One TRIGGER-ABSENT number is exactly how a capability gap
        # (`no_covering_test_selected` -- GT's graph knows no covering test) hides behind a
        # green test (`covering_pass`).  Publish the split next to the headline.
        out.append(
            "  TRIGGER-ABSENT splits by dominant reason: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(absent_why.items()))
        )
    gated_absent = [f.feature_id for f in features
                    if f.verdict == _VERDICT_ABSENT and f.mistake_gated]
    if gated_absent:
        out.append(
            f"  of the {absent} TRIGGER-ABSENT, {len(gated_absent)} are mistake-gated "
            f"(correct-quiet, NOT failures): {', '.join(sorted(gated_absent))}"
        )
    arbitrated = [f.feature_id for f in features
                  if f.verdict == _VERDICT_ARBITRATED]
    if arbitrated:
        out.append(
            f"  {len(arbitrated)} ARBITRATED (evidence "
            f"produced, referee withheld -- not delivery failures): "
            f"{', '.join(sorted(arbitrated))}"
        )
    inherited = [f.feature_id for f in features
                 if f.verdict == _VERDICT_FIRED and f.attribution.startswith("bound-FACT")]
    if inherited:
        out.append(
            f"  {len(inherited)} FIRED rest on bound-FACT delivery, NOT byte-owner "
            f"lineage (CAP-specific attribution unproven): {', '.join(sorted(inherited))}"
        )
    if result["boundary_stamped"] == 0:
        out.append(
            "  ON-TIME: NOT EVALUABLE for all 17 -- no ledger row carries "
            "`contracted_boundary` (pre-2026-07-26 run). `surface`/`event_type` are "
            "different vocabularies and were NOT substituted."
        )
    gate = result["unresolved_gate"]
    if gate:
        out.append(
            "  CAVEAT -- VERDICTS MAY BE UNDERSTATED: a gate DECISION was emitted by a "
            "layer the layer->fact map cannot resolve, so the gate-owned feature(s) "
            "among the 17 may read TRIGGER-ABSENT when the trigger did occur: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(gate.items()))
        )
    out.append("")

    un_deliv = result["unattributed_delivered"]
    if un_deliv:
        out.append(
            f"UNATTRIBUTED DELIVERIES ({sum(un_deliv.values())} delivered rows whose "
            "layer is not in _LAYER_TO_FACT_CLASS -- not counted for any of the 17):"
        )
        for key, count in un_deliv.most_common():
            out.append(f"  {count:>4}  {key}")
    interesting = Counter({
        k: v for k, v in result["unattributed"].items()
        if not k.startswith("control.participation|")
        and "|evaluated|" not in k
    })
    gate_layers = {k: v for k, v in interesting.items()
                   if k.split("|", 1)[0] in {"completion_cert", "submit_gate",
                                             "change_surface", "patch_delta"}}
    if gate_layers:
        out.append("  gate-layer rows the layer map does not resolve (coverage gap):")
        for key, count in sorted(gate_layers.items()):
            out.append(f"  {count:>4}  {key}")
    return "\n".join(out)


def render_json(result: dict[str, Any]) -> str:
    features: list[FeatureRow] = result["features"]
    payload = {
        "artifacts_dir": result["root"],
        "tasks": result["tasks"],
        "total_ledger_rows": result["total_rows"],
        "rows_with_contracted_boundary": result["boundary_stamped"],
        "features": [
            {
                "feature": f.feature_id,
                "kind": f.kind,
                "bound_fact": f.bound_fact,
                "contracted_boundary": f.contracted_boundary,
                "registry_surface": f.registry_surface,
                "mistake_gated": f.mistake_gated,
                "mistake_gate": f.gate_note,
                "verdict": f.verdict,
                "verdict_detail": f.verdict_detail,
                "opportunities": f.opportunities,
                "terminal_dispositions": f.terminal_dispositions,
                "normalized_terminal_states": dict(
                    f.normalized_terminal_states
                ),
                "delivered_rows": f.delivered,
                "delivered_chars": f.delivered_chars,
                "tasks_fired": sorted(f.tasks_fired),
                "boundaries_seen_event_type": dict(f.seen_event_types),
                "on_time": f.on_time,
                "attribution": f.attribution,
                "counts_inherited_from_bound_fact": f.inherited,
                "evidence": f.evidence,
                "non_delivery_reason_classes": dict(f.reason_classes),
                "non_delivery_reason_detail": dict(f.reason_detail),
            }
            for f in sorted(features, key=lambda r: (r.kind, r.feature_id))
        ],
        "summary": {
            verdict: sum(1 for f in features if f.verdict == verdict)
            for verdict in (_VERDICT_FIRED, _VERDICT_ABSENT, _VERDICT_ARBITRATED,
                            _VERDICT_UNINSTRUMENTED, _VERDICT_FAILURE)
        },
        "unattributed_delivered_rows": dict(result["unattributed_delivered"]),
        "unresolved_gate_decisions": dict(result["unresolved_gate"]),
        "lifecycle_integrity": result["lifecycle_integrity"],
        "semantic_receipt_integrity": result[
            "semantic_receipt_integrity"
        ],
        "delivery_fire_join_integrity": result[
            "delivery_fire_join_integrity"
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-feature verdicts for the 17 DIRECT GroundTruth features.",
    )
    parser.add_argument("artifacts_dir", nargs="?",
                        help="directory holding one subdirectory per task")
    parser.add_argument("--run", help="run id; reuses D:/tmp/gt_run_check/<run>/")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if not args.artifacts_dir and not args.run:
        parser.error("give an artifacts_dir or --run <id>")

    root = resolve_artifacts_dir(args)
    result = evaluate(root)
    text = render_json(result) if args.json else render_text(result)
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

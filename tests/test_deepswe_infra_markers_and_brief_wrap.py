"""Three DeepSWE delivery-surface fixes — red->green (2026-06-09).

  FIX 1 (pier pin)   : every `datacurve-pier` install in the deepswe workflows is
     pinned to ==0.2.0 — the version the --ae/--mounts-json env+mount plumbing was
     source-verified against. An unpinned install lets an upstream pier release
     silently change the contract mid-run.
  FIX 2 (G1 markers) : the substrate-proof step's §E failure echoes exit the job
     BEFORE the agent step creates trial_output.log, so deepswe_outcome.py's INFRA
     classification (which scans that log) yielded UNKNOWN instead of INFRA. Every
     §E marker echo site in deepswe_full.yml must ALSO append the marker line to
     trial_output.log (`| tee -a trial_output.log`, creates the file if absent),
     line-anchored (the classifier matches line-start), with the CANONICAL token —
     the old task-image echo was "FATAL: task image pull failed" while the marker
     list has TASK_IMAGE_PULL_FAIL.
  FIX 3 (G2 wrap)    : brief.txt (gt_run_proof.emit_brief -> generate_v1r_brief
     .brief_text, v1r_brief.py:1417) already STARTS with <gt-task-brief>; gt_agent's
     instruction assembly wrapped it in the tag AGAIN -> nested duplicate tags in
     the agent prompt. The assembly must consume a pre-tagged brief as-is and wrap
     only when the tag is absent — exactly ONE <gt-task-brief> block either way
     (same invariant tests/preflight/test_brief_delivery_invariants.py pins on the
     OH wrapper side).

All deterministic: workflow-text + classifier behavior + pure assembly helper.
No network, no Go toolchain, no task IDs.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WF_DIR = _ROOT / ".github" / "workflows"
_FULL_WF = _WF_DIR / "deepswe_full.yml"
_SUBSTRATE_PROOF = _ROOT / "scripts" / "ci" / "substrate_proof.sh"
_OUTCOME_PATH = _ROOT / "scripts" / "verify" / "deepswe_outcome.py"
_AGENT_PATH = _ROOT / "artifact_deepswe" / "gt_agent.py"
_FULL_SURFACE_MARKERS = (
    "GT_SUBSTRATE_DIGEST_MISSING",
    "GT_SUBSTRATE_PULL_FAIL",
    "GT_RUN_PROOF_FAIL",
    "GT_PROOF_OOM",
    "GT_AGENT_OOM",
    "GT_ARTIFACT_MISSING",
    "TASK_IMAGE_PULL_FAIL",
    "GT_ISSUE_MISSING",
)

_load_count = 0


def _load(path: Path, name_prefix: str):
    """Fresh module instance per call (module-level state isolated per test)."""
    global _load_count
    _load_count += 1
    name = f"{name_prefix}_{_load_count}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gt_env_clear(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GT_"):
            monkeypatch.delenv(k, raising=False)


@pytest.fixture
def outcome_mod():
    return _load(_OUTCOME_PATH, "deepswe_outcome_uut")


@pytest.fixture
def agent_mod(monkeypatch):
    _gt_env_clear(monkeypatch)
    return _load(_AGENT_PATH, "gt_agent_wrapfix_uut")


# ===========================================================================
# FIX 1 — pier pinned to ==0.2.0 in EVERY deepswe workflow that installs it
# ===========================================================================
_PIER_WORKFLOWS = (
    "deepswe_full.yml",
    "deepswe_trial.yml",
    "deepswe_preindex.yml",
    "deepswe_proof_sweep.yml",  # no pier install today; pinned if one is added
)


def test_fix1_every_pier_install_is_pinned_to_0_2_0():
    """The --ae/--mounts-json integration was source-verified against pier 0.2.0;
    any `pip install datacurve-pier` without ==0.2.0 is a silent contract risk."""
    found_any = False
    for wf_name in _PIER_WORKFLOWS:
        p = _WF_DIR / wf_name
        if not p.is_file():
            continue
        for lineno, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "datacurve-pier" in ln and "install" in ln:
                found_any = True
                assert "datacurve-pier==0.2.0" in ln, (
                    f"UNPINNED pier install at {wf_name}:{lineno}: {ln.strip()!r} "
                    f"— must pin datacurve-pier==0.2.0 (the source-verified "
                    f"--ae/--mounts-json contract version)"
                )
    assert found_any, "no pier install line found in any deepswe workflow (paths moved?)"


# ===========================================================================
# FIX 2 — G1: every §E marker echo site tees the marker line into trial_output.log
# ===========================================================================
def test_fix2_every_infra_marker_echo_site_tees_to_trial_log(outcome_mod):
    """Canonical markers are emitted by the workflow or delegated proof script."""
    wf_lines = _FULL_WF.read_text(encoding="utf-8").splitlines()
    substrate_lines = _SUBSTRATE_PROOF.read_text(encoding="utf-8").splitlines()
    all_lines = wf_lines + substrate_lines
    for marker in _FULL_SURFACE_MARKERS:
        assert marker in outcome_mod.INFRA_LOG_MARKERS
        assert any(marker in ln for ln in all_lines), (
            f"workflow/substrate proof has no site for canonical marker {marker!r}"
        )
    assert any(
        'echo "GT_RUN_PROOF_FAIL: ${_code}: ${_detail}" | tee -a trial_output.log'
        in line
        for line in substrate_lines
    )


def test_fix2_task_image_pull_fail_uses_canonical_token():
    """The audit found TASK_IMAGE_PULL_FAIL in INFRA_LOG_MARKERS while the workflow
    echoed 'FATAL: task image pull failed' — a token the classifier can never match."""
    source = (
        _FULL_WF.read_text(encoding="utf-8")
        + "\n"
        + _SUBSTRATE_PROOF.read_text(encoding="utf-8")
    )
    assert 'echo "FATAL: task image pull failed"' not in source, (
        "G1: non-canonical task-image failure echo still present (classifier "
        "matches TASK_IMAGE_PULL_FAIL, not 'FATAL: ...')"
    )
    assert 'echo "FATAL: task image not present after pull"' not in source, (
        "G1: non-canonical post-pull inspect failure echo still present"
    )
    assert "TASK_IMAGE_PULL_FAIL" in source


def test_fix2_workflow_echoed_strings_classify_infra(outcome_mod):
    """End-to-end token parity for every canonical emitted marker."""
    source = (
        _FULL_WF.read_text(encoding="utf-8")
        + "\n"
        + _SUBSTRATE_PROOF.read_text(encoding="utf-8")
    )
    for marker in _FULL_SURFACE_MARKERS:
        assert marker in outcome_mod.INFRA_LOG_MARKERS
        assert marker in source, f"no workflow/substrate source site for {marker!r}"
        log = f"earlier unrelated output\n{marker}: deterministic test detail\n"
        assert marker in outcome_mod.find_infra_markers(log)
        rec = outcome_mod.build_signal_record(
            instance_id="task-x", reward=None, n_agent_steps=None,
            exit_status=None, trial_log=log, cert_dir=None,
        )
        assert rec["failure_class"] == "INFRA"


def test_fix2_marker_absent_from_log_stays_unknown(outcome_mod):
    """Negative control (the pre-fix symptom): a substrate failure whose marker
    never reached trial_output.log classifies UNKNOWN — the bug G1 closes."""
    rec = outcome_mod.build_signal_record(
        instance_id="task-x", reward=None, n_agent_steps=None,
        exit_status=None, trial_log="", cert_dir=None,
    )
    assert rec["failure_class"] == "UNKNOWN"


# ===========================================================================
# FIX 3 — G2: exactly ONE <gt-task-brief> block in the assembled instruction
# ===========================================================================
def test_fix3_pretagged_brief_is_not_double_wrapped(agent_mod):
    """brief.txt already starts with <gt-task-brief> (v1r_brief.py:1417 via
    gt_run_proof.emit_brief) -> consume as-is, exactly ONE open + ONE close tag."""
    brief = "<gt-task-brief>\nFOCUS: app/core.py — anchor hit\n</gt-task-brief>"
    out = agent_mod._prepend_brief(brief, "Fix the bug in app/core.py.")
    assert out.count("<gt-task-brief>") == 1, f"nested duplicate open tags:\n{out}"
    assert out.count("</gt-task-brief>") == 1, f"nested duplicate close tags:\n{out}"
    assert "FOCUS: app/core.py" in out
    assert "Fix the bug in app/core.py." in out
    # brief precedes the instruction (the brief is a preamble, not an appendix)
    assert out.index("</gt-task-brief>") < out.index("Fix the bug in app/core.py.")


def test_fix3_untagged_brief_wrapped_exactly_once(agent_mod):
    """A tag-less brief (legacy host generation paths) still gets the wrap — once."""
    out = agent_mod._prepend_brief("plain brief content", "Fix the bug.")
    assert out.count("<gt-task-brief>") == 1
    assert out.count("</gt-task-brief>") == 1
    assert "plain brief content" in out and "Fix the bug." in out


def test_fix3_empty_brief_is_passthrough(agent_mod):
    """No brief -> no tag, instruction untouched (correct-or-quiet)."""
    assert agent_mod._prepend_brief("", "Fix the bug.") == "Fix the bug."
    assert "<gt-task-brief" not in agent_mod._prepend_brief("", "Fix the bug.")


def test_fix3_substrate_brief_end_to_end_single_tag(agent_mod, monkeypatch, tmp_path):
    """Through the REAL consume path: a substrate brief.txt in the canonical
    pre-tagged shape -> _generate_brief -> _prepend_brief -> ONE tag pair."""
    (tmp_path / "brief.txt").write_text(
        "<gt-task-brief>\n1. app/core.py (def run(self):)\n</gt-task-brief>",
        encoding="utf-8",
    )
    monkeypatch.setenv("GT_PROOF_MODE", "1")
    monkeypatch.setenv("GT_PORTABLE_SUBSTRATE", "1")
    monkeypatch.setenv("GT_CERT_DIR", str(tmp_path))
    brief = agent_mod._generate_brief("fix the bug")
    out = agent_mod._prepend_brief(brief, "fix the bug")
    assert out.count("<gt-task-brief>") == 1, f"double-wrapped substrate brief:\n{out}"
    assert out.count("</gt-task-brief>") == 1


def test_fix3_run_routes_through_prepend_brief(agent_mod):
    """Integration avenue: GTMiniSweAgent.run must assemble via _prepend_brief —
    no residual inline wrap that could reintroduce the double tag."""
    import inspect
    src = inspect.getsource(agent_mod.GTMiniSweAgent.run)
    assert "_prepend_brief" in src, "run() no longer routes through _prepend_brief"
    assert '<gt-task-brief>\\n{brief}' not in src, (
        "run() still carries the unconditional inline wrap"
    )


# ===========================================================================
# F4 — agent-container memory cap + classified OOM (symmetric to proof container)
# ===========================================================================
# ROOT: the agent container is pier-launched via `docker compose up` (pier
# docker.py: ["up","--detach","--wait"], NO --compatibility). The pier compose
# base declared the cap as deploy.resources.limits.memory — a SWARM-ONLY key
# that plain `docker compose up` SILENTLY IGNORES -> a DEAD key -> uncapped
# agent container -> host OOM-SIGKILL surfaced unclassified. The proof container
# (deepswe_full.yml `docker run --memory=10g --memory-swap=10g`) was already
# capped + classified (rc 137 -> GT_PROOF_OOM). F4 restores symmetry: a
# compose-spec mem_limit/memswap_limit (honored WITHOUT --compatibility) + the
# rc=137 -> GT_AGENT_OOM classification at the pier-run step.
_PIER_DIR = (
    _ROOT / "deepswe-pier" / "src" / "pier" / "environments" / "docker"
)
_COMPOSE_BASE = _PIER_DIR / "docker-compose-base.yaml"


def test_f4_pier_compose_base_has_runtime_mem_cap():
    """The pier compose base must carry mem_limit + memswap_limit (the compose-spec
    runtime keys that `docker compose up` honors WITHOUT --compatibility), driven by
    the same ${MEMORY} that feeds the (Swarm-only, dead-without-compat) deploy block.
    memswap_limit == mem_limit so a runaway hits a hard wall (container OOM rc 137)
    rather than swapping into a silent host OOM."""
    import yaml  # type: ignore

    if not _COMPOSE_BASE.is_file():
        pytest.skip(f"external deepswe-pier checkout unavailable at {_COMPOSE_BASE}")
    raw = _COMPOSE_BASE.read_text(encoding="utf-8")
    # Both runtime keys present, both bound to ${MEMORY} (the fixed, per-task-invariant
    # bound — NOT a task-id / repo-size-gated cap).
    assert "mem_limit: ${MEMORY}" in raw, (
        "pier compose base lacks the runtime-honored `mem_limit: ${MEMORY}` — "
        "the agent container is uncapped under plain `docker compose up`"
    )
    assert "memswap_limit: ${MEMORY}" in raw, (
        "pier compose base lacks `memswap_limit: ${MEMORY}` — without it swap "
        "is unbounded and a runaway defers into a silent host OOM instead of "
        "a classified container OOM-kill"
    )
    # The cap must reference the substrate-driven ${MEMORY} var (generalized), never
    # a literal per-task number baked into the compose file.
    doc = yaml.safe_load(raw)
    main = doc["services"]["main"]
    assert main.get("mem_limit") == "${MEMORY}"
    assert main.get("memswap_limit") == "${MEMORY}"


def test_f4_pier_run_step_classifies_agent_oom(outcome_mod):
    """The pier-run step must classify an agent-container OOM (rc/exit 137) as
    GT_AGENT_OOM and tee it to trial_output.log, mirroring the proof container's
    PROOF_RC=137 -> GT_PROOF_OOM branch. Detection keys on PIER_RC==137 OR the
    container's OOM signal in the trial log (pier may surface the in-container kill
    via result.json rather than its own rc)."""
    wf = _FULL_WF.read_text(encoding="utf-8")
    # GT_AGENT_OOM is now a canonical INFRA marker (capacity kill, not agent/GT logic).
    assert "GT_AGENT_OOM" in outcome_mod.INFRA_LOG_MARKERS, (
        "GT_AGENT_OOM must be registered in INFRA_LOG_MARKERS so an agent-container "
        "OOM is classified INFRA (excluded from the resolved denominator), like "
        "GT_PROOF_OOM — never charged to the agent"
    )
    # The pier-run step branches on rc 137.
    assert 'PIER_RC" -eq 137' in wf, (
        "pier-run step has no `PIER_RC == 137` branch — an agent-container OOM "
        "still falls into the generic PIER_RUN_FAIL bucket, unclassified"
    )
    # ...and emits the canonical GT_AGENT_OOM marker, tee'd into the trial log.
    sites = [
        ln for ln in wf.splitlines()
        if 'echo "GT_AGENT_OOM' in ln
    ]
    assert sites, "pier-run step never echoes the GT_AGENT_OOM marker"
    for ln in sites:
        assert "tee -a trial_output.log" in ln, (
            f"GT_AGENT_OOM echo does not tee to trial_output.log (classifier scans "
            f"that file): {ln.strip()}"
        )


def test_f4_agent_oom_marker_classifies_infra(outcome_mod):
    """End-to-end: a trial log carrying the GT_AGENT_OOM line classifies INFRA — a
    capacity kill that invalidates the clean GT-vs-baseline comparison, identical
    treatment to the proof OOM. (Red before F4: GT_AGENT_OOM absent from the marker
    list -> the same log classified AGENT/UNKNOWN, charging a capacity kill to the
    agent.)"""
    log = (
        "earlier agent output\n"
        "GT_AGENT_OOM: agent container hit the memory cap (rc/exit=137 — capacity "
        "kill, not logic).\n"
    )
    assert "GT_AGENT_OOM" in outcome_mod.find_infra_markers(log)
    rec = outcome_mod.build_signal_record(
        instance_id="task-x", reward=None, n_agent_steps=None,
        exit_status=None, trial_log=log, cert_dir=None,
    )
    assert rec["failure_class"] == "INFRA", (
        f"agent-container OOM did not classify INFRA (got {rec['failure_class']!r})"
    )
    # And it is excluded from the resolved-rate denominator (capacity kills never
    # count against GT's resolve rate).
    assert "INFRA" in outcome_mod.DENOMINATOR_EXCLUDED

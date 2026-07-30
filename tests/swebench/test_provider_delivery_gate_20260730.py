from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "swebench"))

from provider_delivery_gate import prove_task_start_delivery


def _row(capsule: str = "Repository-derived obligations.") -> dict:
    payload = {
        "messages": [
            {"role": "user", "content": "Fix it."},
            {"role": "user", "content": [{"type": "text", "text": capsule}]},
        ]
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    rendered_hash = hashlib.sha256(capsule.encode()).hexdigest()
    return {
        "schema": "gt.canonical_delivery.v1",
        "layer": "canonical.provider_delivery",
        "event_type": "canonical_provider_delivery",
        "outcome": "delivered",
        "evidence_lineage": [{"fact_class": "obligations"}],
        "capsule_text": capsule,
        "rendered_content_hash": rendered_hash,
        "content_sha256_16": rendered_hash[:16],
        "chars_delivered": len(capsule),
        "bound_provider_payload_json": payload_json,
        "provider_payload_hash": payload_hash,
        "capsule_binding": {
            "provider_payload_hash": payload_hash,
            "message_index": 1,
            "content_index": 0,
        },
        "provider_response_id": "response-1",
        "model_call_id": "call-1",
    }


def _write(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_accepts_exact_provider_bound_obligations(tmp_path: Path) -> None:
    assert prove_task_start_delivery(_write(tmp_path, _row())) == (
        True,
        "provider_bound_obligations_delivery",
    )


def test_rejects_payload_tampering(tmp_path: Path) -> None:
    row = _row()
    row["bound_provider_payload_json"] = row["bound_provider_payload_json"].replace(
        "Repository-derived", "Forged",
    )
    proven, reason = prove_task_start_delivery(_write(tmp_path, row))
    assert proven is False
    assert "provider_payload_hash_mismatch" in reason


def test_rejects_non_obligation_capsule(tmp_path: Path) -> None:
    row = _row()
    row["evidence_lineage"] = [{"fact_class": "localization"}]
    assert prove_task_start_delivery(_write(tmp_path, row)) == (
        False,
        "provider_bound_obligations_delivery_absent",
    )

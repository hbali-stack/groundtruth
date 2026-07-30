#!/usr/bin/env python3
"""Fail closed unless task-start obligations reached an exact provider payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from consumption_ledger import _provider_payload_delivery


def _is_obligations_delivery(row: dict[str, Any]) -> bool:
    lineage = row.get("evidence_lineage")
    return isinstance(lineage, list) and any(
        isinstance(item, dict) and item.get("fact_class") == "obligations"
        for item in lineage
    )


def prove_task_start_delivery(path: str | Path) -> tuple[bool, str]:
    ledger = Path(path)
    if not ledger.is_file():
        return False, "runtime_ledger_absent"
    rejection_reasons: list[str] = []
    with ledger.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                rejection_reasons.append(f"line_{line_number}:invalid_json")
                continue
            if not isinstance(row, dict) or not _is_obligations_delivery(row):
                continue
            delivery, reason = _provider_payload_delivery(row)
            if delivery is None:
                rejection_reasons.append(reason or "not_provider_delivery")
                continue
            if (
                row.get("schema") != "gt.canonical_delivery.v1"
                or row.get("outcome") != "delivered"
                or not str(row.get("provider_response_id") or "").strip()
                or not str(row.get("model_call_id") or "").strip()
            ):
                rejection_reasons.append("canonical_delivery_identity_invalid")
                continue
            return True, "provider_bound_obligations_delivery"
    if rejection_reasons:
        return False, ",".join(sorted(set(rejection_reasons)))
    return False, "provider_bound_obligations_delivery_absent"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    args = parser.parse_args()
    proven, reason = prove_task_start_delivery(args.ledger)
    print(json.dumps({"proven": proven, "reason": reason}, sort_keys=True))
    return 0 if proven else 1


if __name__ == "__main__":
    raise SystemExit(main())

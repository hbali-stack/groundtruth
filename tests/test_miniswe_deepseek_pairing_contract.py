"""Fail closed when the mini-SWE GT arm changes non-GT model/runtime controls."""

from __future__ import annotations

from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "artifact_deepswe" / "gt_integration"


def _config(name: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_deepseek_gt_and_baseline_share_model_and_runtime_controls() -> None:
    gt = _config("deepswe_gt_pier.yaml")
    baseline = _config("deepswe_gt_pier_baseline.yaml")

    assert gt["model"] == baseline["model"]
    assert gt["environment"] == baseline["environment"]
    assert gt["agent"]["step_limit"] == baseline["agent"]["step_limit"] == 150
    assert gt["agent"]["cost_limit"] == baseline["agent"]["cost_limit"] == 3.0
    assert (
        gt["model"]["model_kwargs"]["extra_body"]["thinking"]["type"]
        == "disabled"
    )

"""Gate deployment on eval accuracy not regressing."""

from __future__ import annotations

import json
from pathlib import Path

ACCURACY_FLOOR = 0.65  # Minimum acceptable accuracy
MAX_REGRESSION = 0.05  # Maximum allowed drop from previous version


def should_deploy(
    new_accuracy: float,
    previous_accuracy_path: Path,
) -> tuple[bool, str]:
    """Decide whether to deploy a new model version.

    Two independent checks: an absolute floor (never deploy something bad
    in isolation) and a relative-regression check (never deploy something
    meaningfully worse than what's currently live, even if it still clears
    the floor) -- see docs/components/08-flywheel.md for why both matter.

    Args:
        new_accuracy: Category accuracy from the eval harness on the new model.
        previous_accuracy_path: Path to JSON file with previous model's accuracy.

    Returns:
        Tuple of (should_deploy: bool, reason: str).
    """
    if new_accuracy < ACCURACY_FLOOR:
        return False, f"Accuracy {new_accuracy:.2%} below floor {ACCURACY_FLOOR:.2%}"

    if previous_accuracy_path.exists():
        prev_data = json.loads(previous_accuracy_path.read_text())
        prev_accuracy = prev_data.get("accuracy", 0.0)
        if new_accuracy < prev_accuracy - MAX_REGRESSION:
            return False, (
                f"Regression detected: {new_accuracy:.2%} vs previous {prev_accuracy:.2%} "
                f"(max allowed drop: {MAX_REGRESSION:.2%})"
            )

    return True, f"Accuracy {new_accuracy:.2%} passes all gates"


def record_deployment(accuracy: float, version: str, record_path: Path) -> None:
    """Record a successful deployment for future regression comparison.

    Args:
        accuracy: Eval accuracy of the deployed model.
        version: Model version string (e.g. 'v2').
        record_path: Path to write the deployment record.
    """
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"version": version, "accuracy": accuracy}))

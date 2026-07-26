"""Track deployed model version history for the flywheel.

Not detailed in the source capstone spec's reference code (only mentioned
in its file structure) -- this is an original design. Complementary to,
not a replacement for, regression_gate.py's own single-file
"previous accuracy" pointer (`last_deployed_accuracy.json`): regression_gate
only ever needs the *immediately previous* version to make a gate decision,
while this module keeps the *full* history, which is what the capstone
doc's "Model version history" portfolio artifact (see
docs/components/08-flywheel.md) actually needs to render.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class VersionRecord:
    """One deployed model version.

    Attributes:
        version: Version tag, e.g. "v1", "v20260726".
        accuracy: Eval category accuracy at deployment time.
        deployed_at: ISO timestamp of when this version was recorded.
        note: Optional free-text context (e.g. "+100 new traces").
    """

    version: str
    accuracy: float
    deployed_at: str
    note: str = ""


def record_version(
    version: str,
    accuracy: float,
    history_path: str | Path,
    note: str = "",
) -> VersionRecord:
    """Append a newly deployed version to the history log.

    Args:
        version: Version tag for this deployment.
        accuracy: Eval category accuracy that earned this deployment.
        history_path: Path to the append-only JSONL history file.
        note: Optional free-text context.

    Returns:
        The VersionRecord that was written.
    """
    record = VersionRecord(version=version, accuracy=accuracy, deployed_at=datetime.now().isoformat(), note=note)
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return record


def load_version_history(history_path: str | Path) -> list[VersionRecord]:
    """Load the full deployed-version history, oldest first.

    Args:
        history_path: Path to the JSONL history file.

    Returns:
        List of VersionRecord, empty if the file doesn't exist yet.
    """
    path = Path(history_path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(VersionRecord(**json.loads(line)))
    return records


def get_latest_version(history_path: str | Path) -> VersionRecord | None:
    """Return the most recently deployed version, or None if none exist."""
    history = load_version_history(history_path)
    return history[-1] if history else None


def format_history_table(history_path: str | Path) -> str:
    """Render the version history as a human-readable summary.

    Matches the style of the capstone doc's "Expected Outcomes" example:
    "v1 (baseline QLoRA) — accuracy: 70% — deployed 2024-01-15".
    """
    history = load_version_history(history_path)
    if not history:
        return "No deployed versions yet."
    lines = []
    for record in history:
        line = f"{record.version} — accuracy: {record.accuracy:.0%} — deployed {record.deployed_at[:10]}"
        if record.note:
            line += f" — {record.note}"
        lines.append(line)
    return "\n".join(lines)

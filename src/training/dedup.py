"""Deduplication and quality filtering for raw investigation traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _dedup_key(raw_trace: dict) -> str:
    """Hash key identifying "the same investigation" for dedup purposes.

    Two traces are treated as duplicates if they investigate the same files
    with the same problem description — re-running an identical scenario
    shouldn't let it dominate the training set.
    """
    basis = f"{raw_trace.get('description', '')}|{raw_trace.get('logcat_path', '')}|{raw_trace.get('dmesg_path', '')}"
    return hashlib.md5(basis.encode()).hexdigest()


def is_low_quality(raw_trace: dict, min_confidence: float = 0.5) -> bool:
    """Return True if a raw trace shouldn't be trained on.

    Two independent disqualifiers: low self-reported confidence, and a
    root_cause_category of "unknown" (the DiagnosticAgent's MAX_STEPS
    fallback — see docs/components/01-diagnostic-agent.md — which has no
    real Final Answer to imitate).
    """
    result = raw_trace.get("result", {})
    if result.get("confidence", 0.0) < min_confidence:
        return True
    if result.get("root_cause_category") == "unknown":
        return True
    return False


def dedup_and_filter(
    input_path: str | Path,
    output_path: str | Path,
    min_confidence: float = 0.5,
) -> dict:
    """Read raw traces, drop duplicates and low-quality entries, write survivors.

    Args:
        input_path: Path to the raw traces JSONL file (append-only).
        output_path: Path to write the deduped, filtered traces JSONL.
        min_confidence: Passed to is_low_quality().

    Returns:
        Dict of counts: total, duplicates_dropped, low_quality_dropped, kept.
    """
    total = 0
    groups: dict[str, list[dict]] = {}

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            raw = json.loads(line)
            groups.setdefault(_dedup_key(raw), []).append(raw)

    # Within each group of re-runs of "the same investigation", keep the
    # highest-confidence occurrence rather than whichever ran first — a
    # failed first attempt shouldn't permanently shadow a later successful
    # one just because it happened to be recorded earlier.
    dup_dropped = 0
    quality_dropped = 0
    kept: list[dict] = []
    for raw_group in groups.values():
        dup_dropped += len(raw_group) - 1
        best = max(raw_group, key=lambda r: r.get("result", {}).get("confidence", 0.0))
        if is_low_quality(best, min_confidence=min_confidence):
            quality_dropped += 1
            continue
        kept.append(best)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for raw in kept:
            f.write(json.dumps(raw) + "\n")

    return {
        "total": total,
        "duplicates_dropped": dup_dropped,
        "low_quality_dropped": quality_dropped,
        "kept": len(kept),
    }

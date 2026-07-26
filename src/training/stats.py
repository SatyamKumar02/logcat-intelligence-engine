"""Dataset statistics reporter for SFT/DPO training data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def sft_stats(sft_path: str | Path) -> dict:
    """Compute summary statistics over a ShareGPT-format SFT JSONL file.

    Reads the "metadata" field TraceConverter.convert_file() attaches to
    each record (root_cause_category, confidence, num_steps) rather than
    re-parsing conversation text.
    """
    if not Path(sft_path).exists():
        return {"total_records": 0, "category_distribution": {}, "avg_confidence": 0.0, "avg_num_steps": 0.0}

    records = [json.loads(line) for line in open(sft_path) if line.strip()]
    metas = [r.get("metadata", {}) for r in records]
    categories = Counter(m.get("root_cause_category", "unknown") for m in metas)
    confidences = [m["confidence"] for m in metas if m.get("confidence") is not None]
    step_counts = [m["num_steps"] for m in metas if m.get("num_steps") is not None]

    return {
        "total_records": len(records),
        "category_distribution": dict(categories),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "avg_num_steps": sum(step_counts) / len(step_counts) if step_counts else 0.0,
    }


def dpo_stats(dpo_path: str | Path) -> dict:
    """Compute summary statistics over a DPO preference-pairs JSONL file."""
    if not Path(dpo_path).exists():
        return {"total_pairs": 0}
    records = [json.loads(line) for line in open(dpo_path) if line.strip()]
    return {"total_pairs": len(records)}


def print_report(sft_path: str | Path, val_path: str | Path | None = None, dpo_path: str | Path | None = None) -> None:
    """Print a human-readable dataset statistics report to stdout."""
    train_stats = sft_stats(sft_path)
    print("SFT Dataset Statistics")
    print("=" * 40)
    print(f"Train records: {train_stats['total_records']}")
    if val_path:
        val_stats = sft_stats(val_path)
        print(f"Val records:   {val_stats['total_records']}")
    print(f"Avg confidence:     {train_stats['avg_confidence']:.2f}")
    print(f"Avg reasoning steps: {train_stats['avg_num_steps']:.1f}")
    print("Category distribution (train):")
    for cat, count in sorted(train_stats["category_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {count}")

    if dpo_path:
        d = dpo_stats(dpo_path)
        print()
        print(f"DPO pairs: {d['total_pairs']}")

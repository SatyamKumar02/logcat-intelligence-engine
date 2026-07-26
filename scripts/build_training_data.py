"""Build SFT + DPO training datasets from collected investigation traces.

Pipeline: dedup + quality-filter raw traces -> convert survivors to ShareGPT
SFT format -> deterministic train/val split -> print dataset statistics.

Also demonstrates DPOPairGenerator by pairing one low-confidence/failed
trace (the kind convert_file would otherwise silently drop from SFT) with
the matching seed case's known-correct diagnosis as a stand-in "human
correction" -- there's no human review UI built yet (see
docs/components/04-training-data-pipeline.md), so the seed corpus is the
best available source of verified ground truth for demonstrating the
DPOPairGenerator mechanism end-to-end against real trace data.

Usage:
    python scripts/build_training_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.training import stats as stats_mod
from src.training.dedup import dedup_and_filter, is_low_quality
from src.training.dpo_generator import DPOPairGenerator
from src.training.trace_recorder import TraceConverter


def load_seed_cases(path: str | Path) -> dict[str, dict]:
    cases = {}
    with open(path) as f:
        for line in f:
            case = json.loads(line)
            cases[case["root_cause_category"]] = case
    return cases


def split_train_val(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    seed: int,
) -> tuple[int, int]:
    """Shuffle deterministically and split into train/val JSONL files."""
    records = [json.loads(line) for line in open(input_path) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * val_fraction)) if records else 0
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    return len(train_records), len(val_records)


def build_dpo_demo_pair(
    raw_traces_path: Path,
    seed_cases_path: Path,
    output_path: Path,
    min_confidence: float,
) -> int:
    """Pair the lowest-quality trace with its seed case's correct diagnosis."""
    seed_cases = load_seed_cases(seed_cases_path)
    candidates = []
    with open(raw_traces_path) as f:
        for line in f:
            raw = json.loads(line)
            if is_low_quality(raw, min_confidence=min_confidence):
                candidates.append(raw)

    if not candidates:
        print("No low-quality/failed traces found -- skipping DPO demo pair (nothing to correct).")
        return 0

    candidates.sort(key=lambda r: r.get("result", {}).get("confidence", 0.0))
    chosen_raw = None
    seed_case = None
    for raw in candidates:
        label = raw.get("ground_truth_label")
        if label in seed_cases:
            chosen_raw = raw
            seed_case = seed_cases[label]
            break

    if chosen_raw is None:
        print("No low-quality trace had a matching seed case -- skipping DPO demo pair.")
        return 0

    generator = DPOPairGenerator()
    pair = generator.create_pair(
        user_query=(
            f"Investigate this Android device issue.\n\nProblem: {chosen_raw['description']}\n"
            f"logcat: {chosen_raw['logcat_path']}\ndmesg: {chosen_raw['dmesg_path'] or 'not available'}"
        ),
        model_diagnosis=json.dumps({"root_cause": chosen_raw["result"].get("root_cause", "")}),
        human_correction=json.dumps(
            {
                "root_cause": seed_case["root_cause"],
                "root_cause_category": seed_case["root_cause_category"],
                "recommended_action": seed_case["recommended_action"],
            }
        ),
        investigation_id=chosen_raw["id"],
    )
    generator.write_pairs([pair], output_path)
    print(
        f"Wrote 1 DPO demo pair to {output_path} "
        f"(investigation {chosen_raw['id']}, category={seed_case['root_cause_category']})"
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="data/raw/traces.jsonl")
    parser.add_argument("--deduped", default="data/processed/traces_deduped.jsonl")
    parser.add_argument("--sft-all", default="data/processed/sft_all.jsonl")
    parser.add_argument("--sft-train", default="data/sft/train.jsonl")
    parser.add_argument("--sft-val", default="data/sft/val.jsonl")
    parser.add_argument("--dpo-train", default="data/dpo/train.jsonl")
    parser.add_argument("--seed-cases", default="data/processed/seed_cases.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Step 1: dedup + quality filter raw traces...")
    counts = dedup_and_filter(args.traces, args.deduped, min_confidence=args.min_confidence)
    print(
        f"  total={counts['total']} duplicates_dropped={counts['duplicates_dropped']} "
        f"low_quality_dropped={counts['low_quality_dropped']} kept={counts['kept']}"
    )

    print("\nStep 2: convert survivors to ShareGPT SFT format...")
    converter = TraceConverter()
    n_converted = converter.convert_file(args.deduped, args.sft_all, min_confidence=args.min_confidence)
    print(f"  converted {n_converted} records -> {args.sft_all}")

    print("\nStep 3: train/val split...")
    n_train, n_val = split_train_val(
        Path(args.sft_all), Path(args.sft_train), Path(args.sft_val), args.val_fraction, args.seed
    )
    print(f"  train={n_train} val={n_val}")

    print("\nStep 4: DPO demo pair from a low-quality trace + matching seed case...")
    build_dpo_demo_pair(Path(args.traces), Path(args.seed_cases), Path(args.dpo_train), args.min_confidence)

    print("\nStep 5: dataset statistics...")
    stats_mod.print_report(args.sft_train, val_path=args.sft_val, dpo_path=args.dpo_train)


if __name__ == "__main__":
    main()

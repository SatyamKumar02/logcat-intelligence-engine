"""Validate the SFT/DPO training data pipeline locally, before spending Kaggle GPU hours.

This does NOT require bitsandbytes or a GPU -- tokenization runs fine on
CPU. It catches real bugs cheaply: malformed conversations, records that
blow past max_seq_length once tokenized, empty datasets, etc.

Usage:
    python scripts/validate_finetune_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.finetune.qlora_config import QLoRAConfig
from src.finetune.train_sft import format_conversation, load_dataset_from_jsonl


def validate_sft_file(path: str, tokenizer, max_seq_length: int) -> dict:
    """Load, format, and tokenize an SFT JSONL file; report length stats."""
    if not Path(path).exists():
        return {"path": path, "exists": False}

    dataset = load_dataset_from_jsonl(path)
    formatted = dataset.map(format_conversation)

    lengths = [len(tokenizer.encode(ex["text"])) for ex in formatted]
    over_limit = [n for n in lengths if n > max_seq_length]

    return {
        "path": path,
        "exists": True,
        "num_records": len(dataset),
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "avg_tokens": sum(lengths) / len(lengths) if lengths else 0,
        "num_over_max_seq_length": len(over_limit),
    }


def validate_dpo_file(path: str, tokenizer, max_length: int) -> dict:
    """Load a DPO JSONL file and check prompt/chosen/rejected lengths."""
    if not Path(path).exists():
        return {"path": path, "exists": False}

    records = [json.loads(line) for line in open(path) if line.strip()]
    if not records:
        return {"path": path, "exists": True, "num_records": 0}

    for r in records:
        for key in ("prompt", "chosen", "rejected"):
            if key not in r:
                raise ValueError(f"DPO record missing required key '{key}': {r}")

    combined_lengths = [
        len(tokenizer.encode(r["prompt"])) + len(tokenizer.encode(r["chosen"])) for r in records
    ]
    over_limit = [n for n in combined_lengths if n > max_length]

    return {
        "path": path,
        "exists": True,
        "num_records": len(records),
        "max_combined_tokens": max(combined_lengths),
        "num_over_max_length": len(over_limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-train", default="data/sft/train.jsonl")
    parser.add_argument("--sft-val", default="data/sft/val.jsonl")
    parser.add_argument("--dpo-train", default="data/dpo/train.jsonl")
    args = parser.parse_args()

    config = QLoRAConfig()
    print(f"Loading tokenizer for {config.model_name} (small download, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    print(f"\nmax_seq_length in QLoRAConfig: {config.max_seq_length}\n")

    print("=== SFT train ===")
    train_stats = validate_sft_file(args.sft_train, tokenizer, config.max_seq_length)
    print(json.dumps(train_stats, indent=2))

    print("\n=== SFT val ===")
    val_stats = validate_sft_file(args.sft_val, tokenizer, config.max_seq_length)
    print(json.dumps(val_stats, indent=2))

    print("\n=== DPO train ===")
    dpo_stats = validate_dpo_file(args.dpo_train, tokenizer, config.max_seq_length)
    print(json.dumps(dpo_stats, indent=2))

    problems = []
    if not train_stats.get("exists") or train_stats.get("num_records", 0) == 0:
        problems.append("SFT train set is missing or empty")
    if train_stats.get("num_over_max_seq_length", 0) > 0:
        problems.append(f"{train_stats['num_over_max_seq_length']} SFT train record(s) exceed max_seq_length")
    if val_stats.get("num_over_max_seq_length", 0) > 0:
        problems.append(f"{val_stats['num_over_max_seq_length']} SFT val record(s) exceed max_seq_length")
    if dpo_stats.get("num_over_max_length", 0) > 0:
        problems.append(f"{dpo_stats['num_over_max_length']} DPO record(s) exceed max_seq_length")

    print("\n" + "=" * 50)
    if problems:
        print("Problems found:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No problems found. Data is ready for Kaggle GPU training.")


if __name__ == "__main__":
    main()

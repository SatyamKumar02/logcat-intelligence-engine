"""Automatically trigger fine-tuning when enough new examples accumulate."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

RETRAIN_THRESHOLD = 100  # New verified examples before triggering retrain


def count_new_examples(traces_path: Path, last_count_path: Path) -> int:
    """Count new verified investigation traces since the last training run.

    Args:
        traces_path: Path to the raw traces JSONL file.
        last_count_path: Path to a file storing the last known count.

    Returns:
        Number of new examples since last training.
    """
    total = sum(1 for _ in open(traces_path))
    last = int(last_count_path.read_text().strip()) if last_count_path.exists() else 0
    return total - last


def trigger_retrain(
    traces_path: Path,
    last_count_path: Path,
    retrain_script: str = "scripts/weekly_retrain.sh",
) -> bool:
    """Check threshold and trigger retrain if met.

    Args:
        traces_path: Path to the raw traces JSONL file.
        last_count_path: Path to the count tracking file.
        retrain_script: Path to the retrain pipeline script (overridable for testing).

    Returns:
        True if retraining was triggered and succeeded, False otherwise.
    """
    new_count = count_new_examples(traces_path, last_count_path)
    if new_count < RETRAIN_THRESHOLD:
        print(f"Not enough new examples: {new_count}/{RETRAIN_THRESHOLD}")
        return False

    print(f"Threshold reached: {new_count} new examples. Triggering retrain...")
    result = subprocess.run(["bash", retrain_script], capture_output=True, text=True)
    if result.returncode == 0:
        # Only advance the watermark on SUCCESS -- a failed run must be
        # retried on the next check rather than silently losing track of
        # the un-trained examples.
        total = sum(1 for _ in open(traces_path))
        last_count_path.write_text(str(total))
        print("Retrain completed successfully.")
        return True
    else:
        print(f"Retrain failed:\n{result.stderr}")
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-dir", default="data/raw", help="Directory containing traces.jsonl")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between checks when not --once (default: hourly)")
    parser.add_argument("--once", action="store_true", help="Check once and exit instead of looping (e.g. for a cron job)")
    args = parser.parse_args()

    traces_path = Path(args.watch_dir) / "traces.jsonl"
    last_count_path = Path(args.watch_dir) / ".last_retrain_count"

    while True:
        if not traces_path.exists():
            print(f"Waiting for {traces_path} to exist...")
        else:
            trigger_retrain(traces_path, last_count_path)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

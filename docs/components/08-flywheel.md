# Component: The Flywheel (Auto-Retrain + Regression Gate)

**Status: IMPLEMENTED (Python logic); orchestration script written but not run end-to-end** — `src/flywheel/auto_trigger.py`, `src/flywheel/regression_gate.py`, `src/flywheel/version_tracker.py`, `scripts/weekly_retrain.sh`

This is the component that closes the loop and turns everything else into a
*system that improves itself* rather than a one-time fine-tuning exercise.
Mental model from the source spec: **the flywheel is the real product** —
any model shipped on day one is mediocre because training data is limited;
what compounds over months of production use is the automated pipeline that
keeps re-training on freshly accumulated, human-verified traces.

---

## 1. Architecture

```mermaid
flowchart TD
    PROD["Deployed vLLM server\n(07-deployment-serving.md)"]
    NEWCASES["New investigations\n(production usage)"]
    TRACES["data/raw/traces.jsonl\n(growing)"]
    PROD --> NEWCASES --> TRACES

    subgraph Trigger["auto_trigger.py"]
        COUNT["count_new_examples()\ntotal - last_known_count"]
        CHECK{"new_count >= 100?"}
        COUNT --> CHECK
    end
    TRACES --> COUNT

    CHECK -->|no| WAIT["Not enough new examples\n(no-op)"]
    CHECK -->|yes| RETRAIN["scripts/weekly_retrain.sh"]

    subgraph Pipeline["weekly_retrain.sh (5 steps)"]
        S1["1. Convert traces\n-> data/sft/train.jsonl"]
        S2["2. QLoRA SFT training\n(train_sft.py)"]
        S3["3. Merge adapter\n(merge_adapter.py)"]
        S4["4. Run eval harness\n(run_eval.py)"]
        S5["5. Regression gate check"]
        S1 --> S2 --> S3 --> S4 --> S5
    end
    RETRAIN --> S1

    subgraph Gate["regression_gate.should_deploy()"]
        FLOOR{"accuracy >= 0.65\n(ACCURACY_FLOOR)?"}
        REGRESS{"accuracy >= prev_accuracy - 0.05\n(MAX_REGRESSION)?"}
        FLOOR -->|no| REJECT["Block deployment"]
        FLOOR -->|yes| REGRESS
        REGRESS -->|no| REJECT
        REGRESS -->|yes| ACCEPT["Approve deployment"]
    end
    S5 --> FLOOR

    ACCEPT --> BUILD["docker build\nnew tagged image\nlogcat-ie-serve:vN"]
    ACCEPT --> RECORD["version_tracker.py\nrecord_deployment()"]
    REJECT --> KEEPOLD["Keep serving current version\n(no image rebuild)"]

    BUILD --> PROD
```

---

## 2. Auto-Retrain Trigger

```python
RETRAIN_THRESHOLD = 100  # new verified examples before triggering retrain

def count_new_examples(traces_path, last_count_path) -> int:
    total = sum(1 for _ in open(traces_path))
    last = int(last_count_path.read_text()) if last_count_path.exists() else 0
    return total - last

def trigger_retrain(traces_path, last_count_path) -> bool:
    if count_new_examples(...) < RETRAIN_THRESHOLD:
        return False
    result = subprocess.run(["bash", "scripts/weekly_retrain.sh"], ...)
    if result.returncode == 0:
        last_count_path.write_text(str(total))  # advance the watermark
        return True
    return False  # do NOT advance the watermark on failure — retry next check
```

**Why a simple line-count watermark instead of a timestamp or database:**
`data/raw/traces.jsonl` is append-only (see
[04-training-data-pipeline.md](04-training-data-pipeline.md)), so "how many
new lines since I last retrained" is just `total_lines - last_known_count` —
no need for a database or even structured trace metadata to answer "is it
time yet." The watermark file (`last_count_path`) is only advanced *after* a
successful retrain, so a failed run doesn't lose track of un-trained
examples — the next check will see the same (or larger) `new_count` and
retry.

**Why a fixed threshold (100) rather than a fixed schedule (e.g. "daily"):**
ties retraining cadence to actual signal volume rather than the calendar — a
quiet week with 20 new traces doesn't trigger a low-value retrain; a busy
week with 300 new traces could trigger multiple cycles. This is the same
reasoning as "eval-gated CI" vs. "scheduled deploys": trigger on a meaningful
condition, not a clock.

---

## 3. Regression Gate

```python
ACCURACY_FLOOR = 0.65      # absolute minimum acceptable accuracy
MAX_REGRESSION = 0.05      # max allowed drop from the currently-deployed version

def should_deploy(new_accuracy, previous_accuracy_path) -> tuple[bool, str]:
    if new_accuracy < ACCURACY_FLOOR:
        return False, f"below floor {ACCURACY_FLOOR:.2%}"
    if previous_accuracy_path.exists():
        prev = json.loads(previous_accuracy_path.read_text())["accuracy"]
        if new_accuracy < prev - MAX_REGRESSION:
            return False, f"regression vs previous {prev:.2%}"
    return True, "passes all gates"
```

**Two independent checks, not one:**

1. **Absolute floor** — never deploy a model that's simply bad in isolation,
   regardless of what came before (protects against, e.g., a corrupted
   training run producing a technically-non-regressing-but-still-terrible
   model if the previous version happened to also be bad).
2. **Relative regression** — never deploy a model that's meaningfully *worse*
   than what's currently live, even if it still clears the absolute floor
   (protects against slow degradation across many retrain cycles — a model
   could stay above 0.65 while still creeping downward release after
   release without this check).

This is the automated-CI-gate pattern applied to a model artifact instead of
a code artifact: `should_deploy()` is conceptually identical to "does this
build pass its test suite," just measured by `DiagnosticEval.category_accuracy`
(see [06-eval-harness.md](06-eval-harness.md)) instead of unit tests.

**On gate failure:** the pipeline does *not* delete the failed checkpoint or
crash — it just skips the Docker rebuild and deployment step, leaving the
currently-serving version untouched. The failed training run's artifacts
remain available for debugging (why did accuracy drop? bad data batch? LR
too high? a regression in the DPO stage specifically?).

---

## 4. `weekly_retrain.sh` — The Full Pipeline, End to End

```bash
MODEL_VERSION="v$(date +%Y%m%d)"
# 1. Convert new traces -> SFT format (TraceConverter, min_confidence=0.6)
# 2. Run QLoRA SFT training (train_sft.py)
# 3. Merge LoRA adapter (merge_adapter.py)
# 4. Run eval harness against the new merged model, capture accuracy
# 5. regression_gate.should_deploy(accuracy, ...) ->
#      if yes: record_deployment() + docker build --build-arg MODEL_PATH=... -t logcat-ie-serve:$MODEL_VERSION
#      if no:  exit 1 (script fails loudly, no image built)
```

Despite the name "weekly," this script is triggered by volume
(`auto_trigger.py`'s threshold check), not literally a weekly cron — "weekly"
in the capstone spec describes the *expected typical cadence* once the
system is in steady-state production use, not a hardcoded schedule.

---

## 5. Version Tracking

`version_tracker.py` — not detailed in the source capstone spec's reference
code (only mentioned in its file structure), so this is an original design.
It keeps a **separate, append-only** JSONL history (`record_version()`,
`load_version_history()`, `get_latest_version()`, `format_history_table()`)
rather than overloading `regression_gate.py`'s single-file
`last_deployed_accuracy.json` — that file only ever needs the *immediately
previous* version to make one gate decision, while this module keeps the
*full* history, which is what the human-readable trend view actually needs.
Verified locally (`load_version_history`/`get_latest_version`/
`format_history_table` all tested against a real 3-version sequence) and
produces exactly the capstone doc's expected format:

```
v1 — accuracy: 70% — deployed 2026-07-26 — baseline QLoRA
v2 — accuracy: 74% — deployed 2026-07-26 — +100 new traces
v3 — accuracy: 77% — deployed 2026-07-26 — +100 new traces
```

This history is itself a portfolio artifact — a visible, monotonic(-ish)
accuracy trend across retrain cycles is the concrete evidence that the
flywheel claim ("the system improves itself over time") is real and not just
architectural aspiration. (The dates above are identical because the test
ran all three `record_version()` calls back to back in one session — a real
multi-week deployment history would naturally spread them out.)

---

## 5b. Verified Locally (2026-07-26)

Unlike Phase 4/5, everything here is plain Mac-local Python + bash — no GPU
needed — so it was tested for real, not just written:

**`auto_trigger.py`**, against real temp files and real (fake) shell
scripts:
- Below threshold (50/100 examples) → not triggered, no watermark written.
- Above threshold, retrain script exits 1 → not triggered, watermark
  **not** advanced (confirms a failed run gets retried next check, not
  silently skipped).
- Above threshold, retrain script exits 0 → triggered, watermark advances
  to the new total; a subsequent check correctly reports 0 new examples.
- The `--once` CLI flag tested end to end, including the "watch dir doesn't
  exist yet" case (prints a wait message instead of crashing).

**`regression_gate.py`**, five scenarios against real temp files:
no-previous-version pass, below-floor rejection, small-regression-within-
tolerance pass, large-regression rejection, and improvement pass — all five
returned the expected `(bool, reason)`.

**`version_tracker.py`**: empty-history edge cases, then a real 3-version
`record_version()` sequence — `load_version_history()`, `get_latest_version()`,
and `format_history_table()` all returned correct results (see
[Section 5](#5-version-tracking) for the actual output).

**`scripts/weekly_retrain.sh`**: `bash -n` syntax-checked clean. Its
embedded step-5 Python block (the regression-gate + version-tracker wiring,
after simulating bash's `${MODEL_VERSION}` substitution) was extracted and
`ast.parse()`-checked, then **actually run** against a simulated
`eval_results_vX.json` for both outcomes: a passing accuracy (0.82) correctly
wrote `last_deployed_accuracy.json` + appended to `version_history.jsonl`
and exited 0; a failing accuracy (0.40, below `ACCURACY_FLOOR`) correctly
printed "Deploy decision: False" and exited 1 — which, under the script's
`set -euo pipefail`, would halt before ever reaching the `docker build`
step. Steps 1-4 (dedup/convert, QLoRA SFT, merge, eval) call real scripts
from Phases 2-5 that this repo already has, but the script as a whole has
**not** been run end to end, since steps 2-3 need a CUDA GPU this Mac
doesn't have.

**`scripts/run_eval.py`** gained `--base-url`/`--model`/`--title` overrides
specifically so `weekly_retrain.sh` can evaluate a freshly deployed
fine-tuned checkpoint instead of always defaulting to the dev-time Ollama
backbone — a small but necessary Phase 3 extension discovered while wiring
Phase 6 together.

---

## 6. How to Explain This Component in an Interview

- "The flywheel is genuinely the part of this project I'd point to as the
  differentiator — anyone can fine-tune a model once; the interesting
  engineering problem is deciding *when* to retrain, and *whether* the result
  is safe to ship, both automatically."
- "The regression gate has two independent failure conditions on purpose —
  an absolute floor and a relative-regression check — because either one
  alone misses a real failure mode: a bad-in-isolation model, or a slow
  creeping decline across many cycles."
- "Retraining triggers on accumulated signal volume, not a calendar schedule
  — the same philosophy as event-driven CI over cron-scheduled builds."
- "A failed regression-gate check is a safe no-op, not a rollback scramble —
  the currently deployed image is simply left alone, and the failed
  checkpoint stays around for debugging."
- "This component doesn't need a GPU, so unlike Phase 4/5 I held it to the
  same bar as the fully local phases — every threshold, every gate
  decision, and the failure-doesn't-lose-progress watermark behavior are
  tested against real temp files and real subprocess exit codes, not just
  reasoned about."

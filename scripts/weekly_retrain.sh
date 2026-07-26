#!/bin/bash
# scripts/weekly_retrain.sh
# Full retrain pipeline: dedup/convert traces -> train -> merge -> eval -> gate -> deploy.
#
# Despite the name, this runs on volume (auto_trigger.py's threshold check),
# not literally weekly -- see docs/components/08-flywheel.md. Steps 2-4 need
# a CUDA GPU + a serving endpoint for the new model; they will fail on this
# Mac and on any host without bitsandbytes/vLLM available. This script has
# NOT been run end to end -- see CONTEXT.md for what has and hasn't been verified.
#
# Env vars this script reads:
#   MODEL_SERVE_BASE_URL - OpenAI-compatible base URL for the newly deployed
#     model, used to evaluate it in step 4 (e.g. http://localhost:8000/v1
#     once Phase 5's Docker image is running the new checkpoint). Falls
#     back to the local Ollama dev backbone if unset, which only proves the
#     pipeline runs -- it does NOT evaluate the actual fine-tuned model.

set -euo pipefail

MODEL_VERSION="v$(date +%Y%m%d)"
echo "Starting retrain for model version: $MODEL_VERSION"

echo "[1/5] Dedup + convert traces to SFT/DPO format..."
python3 scripts/build_training_data.py

echo "[2/5] Running QLoRA SFT training..."
python3 src/finetune/train_sft.py

echo "[3/5] Merging LoRA adapter..."
python3 src/finetune/merge_adapter.py

echo "[4/5] Running eval harness against the new model..."
EVAL_BASE_URL="${MODEL_SERVE_BASE_URL:-http://localhost:11434/v1}"
EVAL_MODEL="${MODEL_SERVE_MODEL:-qwen2.5:7b}"
python3 scripts/run_eval.py \
    --base-url "$EVAL_BASE_URL" \
    --model "$EVAL_MODEL" \
    --output-json "eval_results_${MODEL_VERSION}.json" \
    --output-md "eval_results_${MODEL_VERSION}.md" \
    --title "Eval Results — ${MODEL_VERSION}"

echo "[5/5] Checking regression gate..."
python3 -c "
import json
import sys
from pathlib import Path
from src.flywheel.regression_gate import should_deploy, record_deployment
from src.flywheel.version_tracker import record_version

with open('eval_results_${MODEL_VERSION}.json') as f:
    accuracy = json.load(f)['category_accuracy']

previous_path = Path('data/processed/last_deployed_accuracy.json')
deploy, reason = should_deploy(accuracy, previous_path)
print(f'Deploy decision: {deploy} — {reason}')

if not deploy:
    sys.exit(1)

record_deployment(accuracy, '${MODEL_VERSION}', previous_path)
record_version(
    '${MODEL_VERSION}', accuracy,
    Path('data/processed/version_history.jsonl'),
    note='auto-retrain',
)

import subprocess
subprocess.run(
    [
        'docker', 'build',
        '--build-arg', 'MODEL_PATH=outputs/merged-diagnostic-${MODEL_VERSION}',
        '-f', 'docker/Dockerfile.serve',
        '-t', 'logcat-ie-serve:${MODEL_VERSION}',
        '.',
    ],
    check=True,
)
print('New image built: logcat-ie-serve:${MODEL_VERSION}')
"

echo "Retrain complete. Model version $MODEL_VERSION deployed."

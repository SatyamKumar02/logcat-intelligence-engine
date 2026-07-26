"""Run the DiagnosticAgent across all synthetic scenarios and record traces.

This is the data-generation half of Phase 2: every investigation run here
becomes a row in data/raw/traces.jsonl via TraceRecorder, which
scripts/build_training_data.py later turns into SFT/DPO training data.

Requires the same setup as scripts/run_agent_demo.py (Ollama running,
synthetic logs + FAISS case index built).

Usage:
    python scripts/collect_traces.py
    python scripts/collect_traces.py --repeats 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.tools import RAGRetrieverTool
from src.training.trace_recorder import TraceRecorder

load_dotenv()

# Covers all 10 categories the synthetic log generator + eval set use.
SCENARIOS = {
    "crash": {
        "logcat": "data/raw/sample_logcats/crash.txt",
        "dmesg": None,
        "description": "App crashes immediately on launch with a Java exception",
    },
    "anr": {
        "logcat": "data/raw/sample_logcats/anr.txt",
        "dmesg": None,
        "description": "App freezes and shows ANR dialog after a few seconds of use",
    },
    "oom": {
        "logcat": "data/raw/sample_logcats/oom.txt",
        "dmesg": None,
        "description": "App crashes with an out-of-memory error",
    },
    "gpu_fault": {
        "logcat": "data/raw/sample_logcats/anr.txt",
        "dmesg": "data/raw/sample_dmesgs/gpu_fault.txt",
        "description": "Device rebooted unexpectedly during video playback, no app-level crash seen",
    },
    "kernel_panic": {
        "logcat": "data/raw/sample_logcats/anr.txt",
        "dmesg": "data/raw/sample_dmesgs/kernel_panic.txt",
        "description": "Device hard-rebooted with a kernel BUG in the log",
    },
    "thermal": {
        "logcat": "data/raw/sample_logcats/thermal.txt",
        "dmesg": None,
        "description": "Device runs noticeably slower and warm during sustained CPU-intensive workloads",
    },
    "camera_crash": {
        "logcat": "data/raw/sample_logcats/camera_crash.txt",
        "dmesg": None,
        "description": "Camera app crashes shortly after opening the camera",
    },
    "oom_kill": {
        "logcat": "data/raw/sample_logcats/anr.txt",
        "dmesg": "data/raw/sample_dmesgs/oom_kill.txt",
        "description": "Background app silently disappeared, user lost unsaved state, no crash shown",
    },
    "binder_failure": {
        "logcat": "data/raw/sample_logcats/binder_failure.txt",
        "dmesg": None,
        "description": "A system service became unresponsive and Binder transactions started failing",
    },
    "memory_leak": {
        "logcat": "data/raw/sample_logcats/memory_leak.txt",
        "dmesg": None,
        "description": "App heap grows continuously across screens until an OutOfMemoryError eventually occurs",
    },
}


def build_agent() -> DiagnosticAgent:
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    model = os.environ.get("AGENT_MODEL", "qwen2.5:7b")

    rag_tool = None
    index_path = Path("data/processed/case_index.faiss")
    metadata_path = Path("data/processed/case_metadata.jsonl")
    if index_path.exists() and metadata_path.exists():
        rag_tool = RAGRetrieverTool(str(index_path), str(metadata_path))

    return DiagnosticAgent(client=client, model=model, rag_tool=rag_tool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2, help="Investigations per scenario (default: 2)")
    parser.add_argument("--output", default="data/raw/traces.jsonl", help="Trace output path")
    args = parser.parse_args()

    agent = build_agent()
    recorder = TraceRecorder(args.output)

    total = len(SCENARIOS) * args.repeats
    done = 0
    for name, spec in SCENARIOS.items():
        for rep in range(args.repeats):
            done += 1
            print(f"[{done}/{total}] Investigating '{name}' (rep {rep + 1}/{args.repeats})...", flush=True)
            result = agent.investigate(
                logcat_path=spec["logcat"],
                dmesg_path=spec["dmesg"],
                description=spec["description"],
            )
            record_id = recorder.record(
                result=result,
                logcat_path=spec["logcat"],
                dmesg_path=spec["dmesg"],
                description=spec["description"],
                label=name,
            )
            print(
                f"    -> id={record_id} category={result.root_cause_category} "
                f"confidence={result.confidence:.2f} steps={len(result.steps)}"
            )

    print(f"\nWrote {total} traces to {args.output}")


if __name__ == "__main__":
    main()

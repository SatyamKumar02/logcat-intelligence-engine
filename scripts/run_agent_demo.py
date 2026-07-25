"""Run the DiagnosticAgent against synthetic log scenarios and print traces.

Requires:
  - Ollama running locally with the agent model pulled (see .env AGENT_MODEL).
  - data/raw/sample_logcats/ and data/raw/sample_dmesgs/ populated
    (run scripts/generate_synthetic_logs.py first).
  - data/processed/case_index.faiss + case_metadata.jsonl built
    (run scripts/build_case_index.py first) — optional, RAG is skipped if absent.

Usage:
    python scripts/run_agent_demo.py
    python scripts/run_agent_demo.py --scenario crash
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.tools import RAGRetrieverTool

load_dotenv()

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
        "logcat": "data/raw/sample_logcats/anr.txt",  # placeholder logcat, dmesg carries the signal
        "dmesg": "data/raw/sample_dmesgs/gpu_fault.txt",
        "description": "Device rebooted unexpectedly during video playback, no app-level crash seen",
    },
    "kernel_panic": {
        "logcat": "data/raw/sample_logcats/anr.txt",
        "dmesg": "data/raw/sample_dmesgs/kernel_panic.txt",
        "description": "Device hard-rebooted with a kernel BUG in the log",
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


def run_scenario(agent: DiagnosticAgent, name: str, spec: dict) -> None:
    print("=" * 70)
    print(f"Scenario: {name}")
    print(f"Description: {spec['description']}")
    print("=" * 70)

    result = agent.investigate(
        logcat_path=spec["logcat"],
        dmesg_path=spec["dmesg"],
        description=spec["description"],
    )

    for step in result.steps:
        print(f"\nStep {step.step_index}:")
        print(f"  Thought: {step.thought}")
        print(f"  Action: {step.action}({json.dumps(step.action_input)})")
        print(f"  Observation: {step.observation[:300]}")

    print("\nFinal Answer:")
    print(
        json.dumps(
            {
                "root_cause": result.root_cause,
                "root_cause_category": result.root_cause_category,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "recommended_action": result.recommended_action,
            },
            indent=2,
        )
    )
    print(f"\nTotal elapsed: {result.total_elapsed_ms:.0f} ms, steps: {len(result.steps)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run only this scenario")
    args = parser.parse_args()

    agent = build_agent()
    scenarios = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS
    for name, spec in scenarios.items():
        run_scenario(agent, name, spec)


if __name__ == "__main__":
    main()

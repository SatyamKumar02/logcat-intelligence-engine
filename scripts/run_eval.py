"""Run the DiagnosticEval harness against the 10 labeled eval tasks.

Requires Ollama running locally with the agent model pulled. Produces both
a machine-readable JSON summary (consumed by the Phase 6 regression gate
later) and a human-readable Markdown report.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --output-md eval_results_baseline.md --output-json eval_results_baseline.json
    python scripts/run_eval.py --skip-judge   # faster iteration, no judge scoring
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from data.eval.tasks import EVAL_TASKS
from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.tools import RAGRetrieverTool
from src.eval.diagnostic_eval import DiagnosticEval
from src.eval.llm_judge import LLMJudge
from src.eval.report import write_report

load_dotenv()


def build_client_and_model() -> tuple[OpenAI, str]:
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    model = os.environ.get("AGENT_MODEL", "qwen2.5:7b")
    return client, model


def build_agent(client: OpenAI, model: str) -> DiagnosticAgent:
    rag_tool = None
    index_path = Path("data/processed/case_index.faiss")
    metadata_path = Path("data/processed/case_metadata.jsonl")
    if index_path.exists() and metadata_path.exists():
        rag_tool = RAGRetrieverTool(str(index_path), str(metadata_path))
    return DiagnosticAgent(client=client, model=model, rag_tool=rag_tool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", default="eval_results_baseline.md")
    parser.add_argument("--output-json", default="eval_results_baseline.json")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge scoring for faster iteration")
    args = parser.parse_args()

    client, model = build_client_and_model()
    agent = build_agent(client, model)
    judge = None if args.skip_judge else LLMJudge(client=client, model=model)

    evaluator = DiagnosticEval(agent=agent, judge=judge)

    print(f"Running eval harness on {len(EVAL_TASKS)} labeled tasks (model={model}, judge={'on' if judge else 'off'})...\n")
    for i, task in enumerate(EVAL_TASKS, 1):
        print(f"  [{i}/{len(EVAL_TASKS)}] {task['id']}: {task['description'][:60]}...")

    summary = evaluator.run(EVAL_TASKS)

    print("\n" + "=" * 50)
    print("Eval Results")
    print("=" * 50)
    print(f"Total tasks:          {summary.total_tasks}")
    print(f"Category accuracy:    {summary.category_accuracy:.1%}")
    print(f"Avg keyword hit rate: {summary.avg_keyword_hit_rate:.2f}")
    print(f"Avg confidence:       {summary.avg_confidence:.2f}")
    print(f"Avg trajectory score: {summary.avg_trajectory_score:.2f}")
    print(f"Avg judge score:      {summary.avg_judge_score:.1f} / 10")

    write_report(summary, args.output_md, title="Eval Results — Baseline (Zero-Shot)")
    print(f"\nMarkdown report written to {args.output_md}")

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "total_tasks": summary.total_tasks,
                "category_accuracy": summary.category_accuracy,
                "avg_keyword_hit_rate": summary.avg_keyword_hit_rate,
                "avg_confidence": summary.avg_confidence,
                "avg_trajectory_score": summary.avg_trajectory_score,
                "avg_judge_score": summary.avg_judge_score,
                "per_task_results": [asdict(r) for r in summary.per_task_results],
            },
            f,
            indent=2,
        )
    print(f"JSON summary written to {args.output_json}")


if __name__ == "__main__":
    main()

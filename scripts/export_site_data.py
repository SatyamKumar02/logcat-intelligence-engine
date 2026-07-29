"""Export real project data into JSON the static showcase site (site/) can fetch.

Reads the actual captured SFT training records and the actual eval harness
output -- no fabricated demo content. Re-run this whenever traces or eval
results are refreshed; the phase status table is hand-maintained here
alongside CONTEXT.md's prose version since CONTEXT.md itself isn't
structured data.

Usage:
    python scripts/export_site_data.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_STEP_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n"
    r"Action:\s*(?P<action>.*?)\n"
    r"Action Input:\s*(?P<action_input>.*?)\n"
    r"Observation:\s*(?P<observation>.*?)(?=\nThought:|\Z)",
    re.DOTALL,
)
_PROBLEM_RE = re.compile(r"Problem:\s*(.*?)\n")


def _try_json(text: str):
    """Parse text as JSON if possible, otherwise return it as a plain string."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_gpt_turn(gpt_text: str) -> tuple[list[dict], dict | str]:
    """Parse a ShareGPT 'gpt' turn's ReAct text back into structured steps.

    Inverse of TraceConverter.convert() in src/training/trace_recorder.py,
    which serializes AgentStep objects into this exact
    Thought/Action/Action Input/Observation/Final Answer text grammar.
    """
    if "Final Answer:" in gpt_text:
        steps_blob, final_blob = gpt_text.split("Final Answer:", 1)
    else:
        steps_blob, final_blob = gpt_text, ""

    steps = []
    for match in _STEP_RE.finditer(steps_blob):
        steps.append(
            {
                "thought": match.group("thought").strip(),
                "action": match.group("action").strip(),
                "action_input": _try_json(match.group("action_input")),
                "observation": _try_json(match.group("observation")),
            }
        )

    final_answer = _try_json(final_blob)
    return steps, final_answer


def export_traces(sft_paths: list[Path], output_path: Path) -> int:
    """Parse all SFT records across the given files into replay-ready traces."""
    traces = []
    for path in sft_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                human_turn = next((t["value"] for t in record["conversations"] if t["from"] == "human"), "")
                gpt_turn = next((t["value"] for t in record["conversations"] if t["from"] == "gpt"), "")
                problem_match = _PROBLEM_RE.search(human_turn)
                description = problem_match.group(1).strip() if problem_match else human_turn[:200]

                steps, final_answer = parse_gpt_turn(gpt_turn)
                metadata = record.get("metadata", {})

                traces.append(
                    {
                        "id": record["id"],
                        "category": metadata.get("root_cause_category", "unknown"),
                        "description": description,
                        "confidence": metadata.get("confidence"),
                        "num_steps": metadata.get("num_steps", len(steps)),
                        "steps": steps,
                        "final_answer": final_answer,
                    }
                )

    traces.sort(key=lambda t: t["category"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"traces": traces}, f, indent=2)
    return len(traces)


def export_eval_results(input_path: Path, output_path: Path) -> None:
    """Copy the eval harness output, adding the eval-set contamination caveat."""
    data = json.loads(input_path.read_text())
    data["caveat"] = (
        "This 90% baseline is NOT a fair zero-shot difficulty measure -- the eval "
        "snippets contain near-literal category keywords and the RAG seed corpus "
        "was hand-written by the same author as the eval tasks. Full analysis in "
        "docs/components/06-eval-harness.md §7."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


# Hand-maintained alongside CONTEXT.md's Phase Status table -- update both
# together when a phase's status changes.
PHASES = [
    {
        "phase": "0",
        "name": "Environment setup",
        "status": "done",
        "note": "Ollama (qwen2.5:7b), venv, folder structure, git init",
    },
    {
        "phase": "1",
        "name": "Agentic diagnosis engine",
        "status": "done",
        "note": "Parsers, tools (incl. FAISS RAG), ReAct DiagnosticAgent verified end-to-end",
    },
    {
        "phase": "2",
        "name": "Training data pipeline",
        "status": "done",
        "note": "TraceRecorder/Converter, DPOPairGenerator, dedup+quality filter, stats",
    },
    {
        "phase": "3",
        "name": "Eval harness",
        "status": "done",
        "note": "DiagnosticEval, TrajectoryGrader, LLMJudge -- baseline 90% (see caveat)",
    },
    {
        "phase": "4",
        "name": "Fine-tuning (cloud GPU)",
        "status": "code-ready-not-run",
        "note": "QLoRA SFT/DPO + Groq distillation + Kaggle notebook -- needs a real Kaggle run",
    },
    {
        "phase": "5",
        "name": "Deployment",
        "status": "mostly-verified",
        "note": "Airgapped Dockerfile lints clean, compose validates, health/client tested vs Ollama",
    },
    {
        "phase": "6",
        "name": "Flywheel",
        "status": "logic-tested-not-e2e",
        "note": "Auto-retrain trigger, regression gate, version tracker -- all tested locally",
    },
    {
        "phase": "stretch",
        "name": "Real AOSP data",
        "status": "not-started",
        "note": "Deferred until synthetic pipeline is proven end-to-end (it now is)",
    },
]


def export_phases(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"phases": PHASES}, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    site_data = root / "site" / "data"

    n = export_traces([root / "data/sft/train.jsonl", root / "data/sft/val.jsonl"], site_data / "traces.json")
    print(f"Exported {n} traces -> site/data/traces.json")

    export_eval_results(root / "eval_results_baseline.json", site_data / "eval_results.json")
    print("Exported eval results -> site/data/eval_results.json")

    export_phases(site_data / "phases.json")
    print("Exported phase status -> site/data/phases.json")


if __name__ == "__main__":
    main()

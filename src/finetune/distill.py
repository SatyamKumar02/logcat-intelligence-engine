"""Collect teacher model outputs for knowledge distillation.

Uses Groq's free tier (Llama 3.3 70B) as the "large teacher" instead of the
capstone spec's Claude Opus reference -- $0 instead of ~$15/M input +
$75/M output tokens (see CONTEXT.md's cost strategy). Groq exposes an
OpenAI-compatible endpoint, so this reuses the same `openai` client the
rest of the codebase already depends on -- no new SDK.

Unlike train_sft.py/train_dpo.py, this needs NO GPU -- it's a plain API
call and can run right here on the Mac once GROQ_API_KEY is set in .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

_TEACHER_PROMPT_TEMPLATE = """You are a senior Android diagnostic engineer.

Problem: {description}

Log excerpt:
{log_excerpt}

Provide a detailed root cause analysis with: root_cause, root_cause_category, confidence, evidence list, recommended_action.
Respond as a single JSON object with exactly those keys.
"""


def collect_teacher_outputs(
    tasks: list[dict],
    output_path: str | Path,
    teacher_model: str = "llama-3.3-70b-versatile",
) -> int:
    """Collect diagnostic outputs from a large teacher model via Groq.

    Args:
        tasks: List of task dicts (e.g. from data.eval.tasks.EVAL_TASKS),
            each with a 'description' and a 'logcat_snippet' or 'dmesg_snippet'.
        output_path: Path to write teacher outputs as JSONL.
        teacher_model: Groq model ID for the teacher.

    Returns:
        Number of teacher outputs collected.
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    count = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            log_excerpt = task.get("logcat_snippet", task.get("dmesg_snippet", ""))
            response = client.chat.completions.create(
                model=teacher_model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": _TEACHER_PROMPT_TEMPLATE.format(
                            description=task["description"], log_excerpt=log_excerpt
                        ),
                    }
                ],
            )
            teacher_output = response.choices[0].message.content
            f.write(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "prompt": task["description"],
                        "teacher_response": teacher_output,
                        "teacher_model": teacher_model,
                    }
                )
                + "\n"
            )
            count += 1

    return count


def teacher_outputs_to_sft(input_path: str | Path, output_path: str | Path) -> int:
    """Convert collected teacher outputs into ShareGPT-format SFT records.

    Unlike TraceConverter (see src/training/trace_recorder.py), the teacher
    was prompted directly for a diagnosis rather than given tool access, so
    these records are single-turn (no Thought/Action/Observation steps) --
    a deliberate simplification matching the capstone spec's reference
    design. Mixing free-form teacher answers with ReAct-formatted
    self-generated traces in the same SFT set is a known simplification;
    see docs/components/05-finetuning-pipeline.md.

    Args:
        input_path: Path to the teacher_outputs.jsonl from collect_teacher_outputs().
        output_path: Path to write ShareGPT-format records.

    Returns:
        Number of records written.
    """
    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            conversations = [
                {
                    "from": "system",
                    "value": (
                        "You are a senior Android OS diagnostic engineer. "
                        "Analyze the provided log artifacts and determine the root cause of the issue."
                    ),
                },
                {"from": "human", "value": f"Investigate this Android device issue.\n\nProblem: {entry['prompt']}"},
                {"from": "gpt", "value": entry["teacher_response"]},
            ]
            fout.write(
                json.dumps(
                    {
                        "id": f"distill_{entry['task_id']}",
                        "conversations": conversations,
                        "metadata": {"source": "distillation", "teacher_model": entry["teacher_model"]},
                    }
                )
                + "\n"
            )
            count += 1
    return count


if __name__ == "__main__":
    from data.eval.tasks import EVAL_TASKS

    n = collect_teacher_outputs(EVAL_TASKS, "data/processed/teacher_outputs.jsonl")
    print(f"Collected {n} teacher outputs -> data/processed/teacher_outputs.jsonl")

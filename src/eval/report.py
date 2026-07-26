"""Markdown report generator for eval harness results."""

from __future__ import annotations

from pathlib import Path

from src.eval.diagnostic_eval import EvalSummary


def render_markdown(summary: EvalSummary, title: str = "Eval Results") -> str:
    """Render an EvalSummary as a Markdown report."""
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total tasks | {summary.total_tasks} |",
        f"| Category accuracy | {summary.category_accuracy:.1%} |",
        f"| Avg keyword hit rate | {summary.avg_keyword_hit_rate:.2f} |",
        f"| Avg confidence | {summary.avg_confidence:.2f} |",
        f"| Avg trajectory score | {summary.avg_trajectory_score:.2f} |",
        f"| Avg judge score | {summary.avg_judge_score:.1f} / 10 |",
        "",
        "## Per-Task Results",
        "",
        "| Task | Expected | Predicted | Correct | Keyword Hit Rate | Confidence | Trajectory | Judge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.per_task_results:
        lines.append(
            f"| {r.task_id} | {r.expected_category} | {r.predicted_category} | "
            f"{'yes' if r.category_correct else 'no'} | {r.keyword_hit_rate:.2f} | "
            f"{r.confidence:.2f} | {r.trajectory_score:.2f} | {r.judge_score:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(summary: EvalSummary, output_path: str | Path, title: str = "Eval Results") -> None:
    """Render and write an EvalSummary Markdown report to disk."""
    Path(output_path).write_text(render_markdown(summary, title=title))

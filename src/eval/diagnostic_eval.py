"""Evaluation harness for the Logcat Intelligence Engine."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.diagnostic_agent import DiagnosticAgent, DiagnosisResult
from src.eval.llm_judge import LLMJudge
from src.eval.trajectory_grader import TrajectoryGrader


@dataclass
class EvalResult:
    """Result of evaluating one diagnostic task.

    Attributes:
        task_id: The eval task identifier.
        predicted_category: Model's predicted root cause category.
        expected_category: Ground-truth category.
        category_correct: Whether the category matches.
        keyword_hit_rate: Fraction of expected keywords found in the diagnosis.
        confidence: Model's reported confidence score.
        trajectory_score: Tool use quality score from TrajectoryGrader.
        judge_score: LLM judge reasoning quality score (0-10).
    """

    task_id: str
    predicted_category: str
    expected_category: str
    category_correct: bool
    keyword_hit_rate: float
    confidence: float
    trajectory_score: float = 0.0
    judge_score: float = 0.0


@dataclass
class EvalSummary:
    """Aggregate results across all eval tasks."""

    total_tasks: int
    category_accuracy: float
    avg_keyword_hit_rate: float
    avg_confidence: float
    avg_trajectory_score: float
    avg_judge_score: float
    per_task_results: list[EvalResult] = field(default_factory=list)


class DiagnosticEval:
    """Run the diagnostic eval harness against a set of labeled tasks."""

    def __init__(self, agent: DiagnosticAgent, judge: LLMJudge | None = None) -> None:
        """Initialize the eval harness.

        Args:
            agent: The DiagnosticAgent to evaluate.
            judge: Optional LLMJudge for reasoning-quality scoring. If None,
                judge_score is left at 0.0 for every task (outcome and
                trajectory grading still run — judging is the most
                expensive/optional dimension, see docs/components/06-eval-harness.md).
        """
        self._agent = agent
        self._trajectory_grader = TrajectoryGrader()
        self._judge = judge

    def grade_task(self, task: dict, result: DiagnosisResult) -> EvalResult:
        """Grade one diagnostic task result.

        Args:
            task: Task dict from EVAL_TASKS with expected_* fields.
            result: DiagnosisResult from the agent.

        Returns:
            EvalResult with grading scores.
        """
        predicted_category = result.root_cause_category
        expected_category = task["expected_root_cause_category"]
        category_correct = (
            predicted_category.lower() == expected_category.lower()
            or expected_category.lower() in predicted_category.lower()
        )

        expected_keywords = task.get("expected_keywords", [])
        combined_text = (result.root_cause + " " + " ".join(result.evidence)).lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
        keyword_hit_rate = hits / len(expected_keywords) if expected_keywords else 0.0

        trajectory = self._trajectory_grader.grade(result.steps)

        judge_score = 0.0
        if self._judge:
            trace_text = "\n".join(
                f"Thought: {s.thought}\nAction: {s.action}\nObservation: {s.observation[:300]}" for s in result.steps
            )
            final_answer = json.dumps(
                {
                    "root_cause": result.root_cause,
                    "root_cause_category": result.root_cause_category,
                    "evidence": result.evidence,
                    "recommended_action": result.recommended_action,
                }
            )
            judge_score = self._judge.score(task["description"], trace_text, final_answer)

        return EvalResult(
            task_id=task["id"],
            predicted_category=predicted_category,
            expected_category=expected_category,
            category_correct=category_correct,
            keyword_hit_rate=keyword_hit_rate,
            confidence=result.confidence,
            trajectory_score=trajectory.score,
            judge_score=judge_score,
        )

    def run(self, tasks: list[dict]) -> EvalSummary:
        """Run the full eval harness on all tasks.

        Args:
            tasks: List of task dicts from EVAL_TASKS.

        Returns:
            EvalSummary with aggregate scores and per-task results.
        """
        results: list[EvalResult] = []
        for task in tasks:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # logcat_path is required by investigate() even for
                # dmesg-only tasks — an empty file makes logcat_parser
                # correctly report "nothing here" rather than injecting
                # unrelated content that could bias the investigation.
                logcat_path = Path(tmp_dir) / "logcat_snippet.txt"
                logcat_path.write_text(task.get("logcat_snippet", ""))

                dmesg_path = None
                if "dmesg_snippet" in task:
                    dmesg_file = Path(tmp_dir) / "dmesg_snippet.txt"
                    dmesg_file.write_text(task["dmesg_snippet"])
                    dmesg_path = str(dmesg_file)

                result = self._agent.investigate(
                    logcat_path=str(logcat_path),
                    dmesg_path=dmesg_path,
                    description=task["description"],
                )
                results.append(self.grade_task(task, result))

        n = len(results) or 1
        return EvalSummary(
            total_tasks=len(results),
            category_accuracy=sum(r.category_correct for r in results) / n,
            avg_keyword_hit_rate=sum(r.keyword_hit_rate for r in results) / n,
            avg_confidence=sum(r.confidence for r in results) / n,
            avg_trajectory_score=sum(r.trajectory_score for r in results) / n,
            avg_judge_score=sum(r.judge_score for r in results) / n,
            per_task_results=results,
        )

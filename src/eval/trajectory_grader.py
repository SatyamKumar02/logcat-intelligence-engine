"""Grades ReAct trajectory quality independent of final-answer correctness.

Outcome grading (DiagnosticEval) alone can't distinguish "investigated
properly and reached the right conclusion" from "guessed the right category
without looking at the evidence." This grades the process instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.diagnostic_agent import AgentStep

_PARSER_TOOLS = {"logcat_parser", "dmesg_parser"}


@dataclass
class TrajectoryResult:
    """Trajectory quality score plus the individual checks behind it.

    Attributes:
        score: Weighted overall score, 0.0-1.0.
        started_with_parser: Whether one of the first 2 actions was a parser
            tool — matches the investigation order the system prompt
            recommends (see src/agent/prompts.py).
        no_immediate_repeats: Whether the agent avoided calling the exact
            same tool with the exact same arguments twice in a row (a proxy
            for "wasted" steps that don't build on the prior observation).
        reasonable_length: Whether the investigation took at least 1 step
            and didn't hit MAX_STEPS without producing a Final Answer.
    """

    score: float
    started_with_parser: bool
    no_immediate_repeats: bool
    reasonable_length: bool


class TrajectoryGrader:
    """Scores tool-use quality of a completed ReAct trace."""

    # Matches DiagnosticAgent.MAX_STEPS — hitting this many steps without a
    # Final Answer means the investigation didn't converge (see
    # docs/components/01-diagnostic-agent.md's known limitation).
    MAX_STEPS = 8

    def grade(self, steps: list[AgentStep]) -> TrajectoryResult:
        """Grade a single investigation's tool-use trajectory.

        Args:
            steps: The AgentStep list from a DiagnosisResult.

        Returns:
            TrajectoryResult with an overall score and the checks behind it.
        """
        if not steps:
            return TrajectoryResult(score=0.0, started_with_parser=False, no_immediate_repeats=True, reasonable_length=False)

        started_with_parser = any(s.action in _PARSER_TOOLS for s in steps[:2])

        no_immediate_repeats = all(
            not (steps[i].action == steps[i + 1].action and steps[i].action_input == steps[i + 1].action_input)
            for i in range(len(steps) - 1)
        )

        reasonable_length = 1 <= len(steps) < self.MAX_STEPS

        score = 0.4 * started_with_parser + 0.3 * no_immediate_repeats + 0.3 * reasonable_length

        return TrajectoryResult(
            score=score,
            started_with_parser=started_with_parser,
            no_immediate_repeats=no_immediate_repeats,
            reasonable_length=reasonable_length,
        )

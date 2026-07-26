"""LLM-as-judge scoring for reasoning quality, via local Ollama ($0 cost)."""

from __future__ import annotations

import re

from openai import OpenAI

_SCORE_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*/\s*10")

_JUDGE_PROMPT_TEMPLATE = """You are grading the reasoning quality of an AI diagnostic agent's investigation into an Android device issue. Judge the REASONING PROCESS, not whether the final category label happens to be correct.

Problem description: {description}

Agent's full investigation trace:
{trace_text}

Agent's final diagnosis:
{final_answer}

Score the reasoning quality on a 0-10 scale, considering:
- Did each step build logically on the previous observation, rather than repeating or ignoring it?
- Is the final diagnosis well-supported by the evidence actually gathered?
- Is the recommended action concrete and actionable for an engineer?

Respond with ONLY one line in the exact format: Score: X/10
"""


class LLMJudge:
    """Scores an investigation's reasoning quality using a judge LLM call."""

    def __init__(self, client: OpenAI, model: str = "qwen2.5:7b") -> None:
        """Initialize the judge.

        Args:
            client: OpenAI-compatible API client (points at local Ollama by default).
            model: Model identifier to use for judging.
        """
        self._client = client
        self._model = model

    def score(self, description: str, trace_text: str, final_answer: str) -> float:
        """Score one investigation's reasoning quality.

        Args:
            description: The original problem description given to the agent.
            trace_text: A rendered Thought/Action/Observation trace.
            final_answer: The agent's final diagnosis, as a JSON string.

        Returns:
            A score from 0.0 to 10.0. Returns 0.0 if the judge's response
            couldn't be parsed (treated as a failed judgment, not a real 0).
        """
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            description=description,
            trace_text=trace_text[:4000],
            final_answer=final_answer,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        text = response.choices[0].message.content or ""
        match = _SCORE_RE.search(text)
        if not match:
            return 0.0
        try:
            return max(0.0, min(10.0, float(match.group(1))))
        except ValueError:
            return 0.0

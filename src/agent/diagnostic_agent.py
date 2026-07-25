"""ReAct-style diagnostic agent for Android log analysis."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import (
    DmesgParserTool,
    LogcatParserTool,
    PatternSearchTool,
    RAGRetrieverTool,
    ToolResult,
)


@dataclass
class AgentStep:
    """One step in a ReAct agent trace.

    Attributes:
        step_index: 0-based index of this step.
        thought: Agent's reasoning text.
        action: Tool name called.
        action_input: Arguments passed to the tool (as dict).
        observation: Tool result summary.
        elapsed_ms: Time taken for this step in milliseconds.
    """

    step_index: int
    thought: str
    action: str
    action_input: dict
    observation: str
    elapsed_ms: float


@dataclass
class DiagnosisResult:
    """Final structured diagnosis from the agent.

    Attributes:
        root_cause: Short description of the root cause.
        root_cause_category: Category tag (anr, crash, oom, gpu_fault, etc.).
        confidence: Confidence score between 0 and 1.
        evidence: List of evidence strings supporting the diagnosis.
        recommended_action: What the engineer should do next.
        steps: Full ReAct trace for training data generation.
        total_elapsed_ms: Total time for the investigation.
    """

    root_cause: str
    root_cause_category: str
    confidence: float
    evidence: list[str]
    recommended_action: str
    steps: list[AgentStep] = field(default_factory=list)
    total_elapsed_ms: float = 0.0


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([\w_]+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


class DiagnosticAgent:
    """ReAct agent that diagnoses Android log files using structured tools.

    Usage:
        agent = DiagnosticAgent(client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
        result = agent.investigate(
            logcat_path="data/raw/sample_logcats/crash.txt",
            description="App crashes immediately on launch"
        )
    """

    MAX_STEPS = 8

    def __init__(
        self,
        client: OpenAI,
        model: str = "qwen2.5:7b",
        rag_tool: RAGRetrieverTool | None = None,
    ) -> None:
        """Initialize the diagnostic agent.

        Args:
            client: OpenAI-compatible API client (points at local Ollama by default).
            model: Model identifier to use for generation.
            rag_tool: Optional RAGRetrieverTool. If None, RAG is skipped.
        """
        self._client = client
        self._model = model
        self._tools = {
            "logcat_parser": LogcatParserTool(),
            "dmesg_parser": DmesgParserTool(),
            "pattern_search": PatternSearchTool(),
        }
        if rag_tool:
            self._tools["rag_retriever"] = rag_tool

    def investigate(
        self,
        logcat_path: str,
        dmesg_path: str | None = None,
        description: str = "",
    ) -> DiagnosisResult:
        """Run a full diagnostic investigation on the provided log files.

        Args:
            logcat_path: Path to logcat text file (required).
            dmesg_path: Path to dmesg text file (optional).
            description: Human-readable problem description (optional context).

        Returns:
            DiagnosisResult with root cause, evidence, and full ReAct trace.
        """
        total_start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Please investigate this Android device issue.\n\n"
                    f"Problem description: {description or 'Unknown issue'}\n"
                    f"logcat file: {logcat_path}\n"
                    f"dmesg file: {dmesg_path or 'not available'}\n\n"
                    "Begin your investigation."
                ),
            },
        ]

        steps: list[AgentStep] = []

        for step_idx in range(self.MAX_STEPS):
            step_start = time.time()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stop=["Observation:"],
            )
            agent_text = response.choices[0].message.content or ""

            fa_match = _FINAL_ANSWER_RE.search(agent_text)
            if fa_match:
                return self._parse_final_answer(
                    fa_match.group(1).strip(),
                    steps=steps,
                    total_elapsed_ms=(time.time() - total_start) * 1000,
                )

            thought_match = _THOUGHT_RE.search(agent_text)
            action_match = _ACTION_RE.search(agent_text)
            input_match = _ACTION_INPUT_RE.search(agent_text)

            if not action_match:
                break

            thought = thought_match.group(1).strip() if thought_match else ""
            action_name = action_match.group(1).strip()
            action_input: dict = {}
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {}

            observation = self._execute_tool(action_name, action_input)
            elapsed = (time.time() - step_start) * 1000

            step = AgentStep(
                step_index=step_idx,
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation,
                elapsed_ms=elapsed,
            )
            steps.append(step)

            messages.append({"role": "assistant", "content": agent_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue your investigation.",
                }
            )

        return DiagnosisResult(
            root_cause="Investigation incomplete — max steps reached",
            root_cause_category="unknown",
            confidence=0.0,
            evidence=[],
            recommended_action="Manual investigation required",
            steps=steps,
            total_elapsed_ms=(time.time() - total_start) * 1000,
        )

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the observation as a string."""
        tool = self._tools.get(tool_name)
        if not tool:
            return f"ERROR: Unknown tool '{tool_name}'"
        result: ToolResult = tool(**args)
        if not result.success:
            return f"ERROR: {result.error}"
        return json.dumps(result.data, indent=2)[:2000]

    def _parse_final_answer(
        self,
        text: str,
        steps: list[AgentStep],
        total_elapsed_ms: float,
    ) -> DiagnosisResult:
        """Parse the agent's Final Answer text into a DiagnosisResult."""
        try:
            data = json.loads(text)
            return DiagnosisResult(
                root_cause=data.get("root_cause", text[:200]),
                root_cause_category=data.get("root_cause_category", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                evidence=data.get("evidence", []),
                recommended_action=data.get("recommended_action", ""),
                steps=steps,
                total_elapsed_ms=total_elapsed_ms,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return DiagnosisResult(
                root_cause=text[:300],
                root_cause_category="unknown",
                confidence=0.5,
                evidence=[],
                recommended_action="",
                steps=steps,
                total_elapsed_ms=total_elapsed_ms,
            )

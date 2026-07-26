"""Convert agent investigation traces into SFT and DPO training data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.diagnostic_agent import DiagnosisResult


@dataclass
class TrainingRecord:
    """A single training example in ShareGPT conversation format.

    Attributes:
        id: Unique identifier for deduplication.
        conversations: List of {from: role, value: content} dicts.
        metadata: Investigation metadata for filtering and analysis.
    """

    id: str
    conversations: list[dict[str, str]]
    metadata: dict[str, Any]


class TraceRecorder:
    """Record investigation traces to a JSONL file for later conversion."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        result: DiagnosisResult,
        logcat_path: str,
        dmesg_path: str | None,
        description: str,
        label: str | None = None,
    ) -> str:
        """Append an investigation trace to the JSONL file.

        Args:
            result: DiagnosisResult from the agent.
            logcat_path: Path to the logcat file that was investigated.
            dmesg_path: Path to the dmesg file (or None).
            description: Problem description used in the investigation.
            label: Ground-truth label if known (for eval).

        Returns:
            The unique ID assigned to this record.
        """
        record_id = hashlib.md5(
            f"{logcat_path}{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        raw = {
            "id": record_id,
            "timestamp": datetime.now().isoformat(),
            "logcat_path": logcat_path,
            "dmesg_path": dmesg_path,
            "description": description,
            "result": asdict(result),
            "ground_truth_label": label,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(raw) + "\n")
        return record_id


class TraceConverter:
    """Convert raw investigation traces into ShareGPT training format."""

    def convert(self, raw_trace: dict) -> TrainingRecord:
        """Convert a raw trace dict to a ShareGPT-format TrainingRecord.

        Args:
            raw_trace: Raw trace as loaded from the JSONL file.

        Returns:
            TrainingRecord with conversations in ShareGPT format.
        """
        result = raw_trace["result"]
        conversations: list[dict[str, str]] = []

        conversations.append(
            {
                "from": "system",
                "value": (
                    "You are a senior Android OS diagnostic engineer. "
                    "Analyze the provided log artifacts and determine the root cause of the issue."
                ),
            }
        )

        conversations.append(
            {
                "from": "human",
                "value": (
                    f"Investigate this Android device issue.\n\n"
                    f"Problem: {raw_trace['description']}\n"
                    f"logcat: {raw_trace['logcat_path']}\n"
                    f"dmesg: {raw_trace['dmesg_path'] or 'not available'}"
                ),
            }
        )

        assistant_parts: list[str] = []
        for step in result.get("steps", []):
            assistant_parts.append(f"Thought: {step['thought']}")
            assistant_parts.append(f"Action: {step['action']}")
            assistant_parts.append(f"Action Input: {json.dumps(step['action_input'])}")
            assistant_parts.append(f"Observation: {step['observation']}")

        final_answer = json.dumps(
            {
                "root_cause": result.get("root_cause"),
                "root_cause_category": result.get("root_cause_category"),
                "confidence": result.get("confidence"),
                "evidence": result.get("evidence"),
                "recommended_action": result.get("recommended_action"),
            },
            indent=2,
        )
        assistant_parts.append(f"Final Answer: {final_answer}")

        conversations.append({"from": "gpt", "value": "\n".join(assistant_parts)})

        return TrainingRecord(
            id=raw_trace["id"],
            conversations=conversations,
            metadata={
                "root_cause_category": result.get("root_cause_category"),
                "confidence": result.get("confidence"),
                "ground_truth_label": raw_trace.get("ground_truth_label"),
                "num_steps": len(result.get("steps", [])),
            },
        )

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        min_confidence: float = 0.5,
    ) -> int:
        """Convert a full JSONL trace file to training format.

        Args:
            input_path: Path to the raw traces JSONL file.
            output_path: Path to write converted training records.
            min_confidence: Skip records below this confidence threshold.

        Returns:
            Number of records written.
        """
        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                confidence = raw.get("result", {}).get("confidence", 0)
                if confidence < min_confidence:
                    continue
                record = self.convert(raw)
                # NOTE: metadata is kept alongside id/conversations (a deliberate
                # deviation from a ShareGPT-only file) so stats.py can compute
                # category distribution / avg confidence directly from the SFT
                # file without re-parsing conversation text. SFT training
                # frameworks that read "conversations" ignore the extra key.
                fout.write(
                    json.dumps(
                        {
                            "id": record.id,
                            "conversations": record.conversations,
                            "metadata": record.metadata,
                        }
                    )
                    + "\n"
                )
                count += 1
        return count

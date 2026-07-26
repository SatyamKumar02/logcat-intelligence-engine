"""Generate DPO preference pairs from human-corrected diagnoses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DPOPair:
    """A preference pair for DPO training.

    Attributes:
        prompt: The user query that produced both outputs.
        chosen: The preferred (human-corrected) completion.
        rejected: The dispreferred (original model) completion.
        metadata: Source information for debugging.
    """

    prompt: str
    chosen: str
    rejected: str
    metadata: dict


class DPOPairGenerator:
    """Generate DPO training pairs from human review sessions."""

    def create_pair(
        self,
        user_query: str,
        model_diagnosis: str,
        human_correction: str,
        investigation_id: str,
    ) -> DPOPair:
        """Create a DPO pair from a model output and human correction.

        Args:
            user_query: The original investigation prompt.
            model_diagnosis: The model's original diagnosis (rejected).
            human_correction: The human engineer's corrected diagnosis (chosen).
            investigation_id: ID for traceability.

        Returns:
            A DPOPair with chosen=human_correction, rejected=model_diagnosis.
        """
        return DPOPair(
            prompt=user_query,
            chosen=human_correction,
            rejected=model_diagnosis,
            metadata={"investigation_id": investigation_id, "source": "human_review"},
        )

    def write_pairs(self, pairs: list[DPOPair], output_path: str | Path) -> None:
        """Write DPO pairs to a JSONL file in TRL-compatible format."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(
                    json.dumps(
                        {
                            "prompt": pair.prompt,
                            "chosen": pair.chosen,
                            "rejected": pair.rejected,
                            **pair.metadata,
                        }
                    )
                    + "\n"
                )

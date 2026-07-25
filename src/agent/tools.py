"""Tool implementations for the DiagnosticAgent."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.parsers.dmesg_parser import DmesgParser
from src.parsers.logcat_parser import LogcatParser


@dataclass
class ToolResult:
    """Structured result returned from any tool call.

    Attributes:
        tool_name: Name of the tool that produced this result.
        success: Whether the tool call succeeded.
        data: The result payload (dict for structured data, str for text).
        error: Error message if success is False.
    """

    tool_name: str
    success: bool
    data: Any
    error: str = ""


class LogcatParserTool:
    """Tool: Parse a logcat file and return structured summary."""

    name = "logcat_parser"
    description = (
        "Parse a logcat text file and return a JSON summary of events. "
        "Input: file path to logcat file. "
        "Output: JSON with error_count, warning_count, top_tags, "
        "anr_detected, crash_detected, oom_detected, sample_errors."
    )

    def __init__(self) -> None:
        self._parser = LogcatParser()

    def __call__(self, file_path: str) -> ToolResult:
        """Parse a logcat file and return a structured summary."""
        try:
            entries = self._parser.parse_file(file_path)
            errors = [e for e in entries if e.priority in ("E", "F")]
            warnings = [e for e in entries if e.priority == "W"]
            tag_counts = Counter(e.tag for e in entries)
            top_tags = [t for t, _ in tag_counts.most_common(10)]
            sample_errors = [{"tag": e.tag, "message": e.message[:200]} for e in errors[:5]]
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "total_entries": len(entries),
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "top_tags": top_tags,
                    "anr_detected": bool(self._parser.find_anrs(entries)),
                    "crash_detected": bool(self._parser.find_crashes(entries)),
                    "oom_detected": bool(self._parser.find_ooms(entries)),
                    "sample_errors": sample_errors,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))


class DmesgParserTool:
    """Tool: Parse a dmesg file and return kernel event summary."""

    name = "dmesg_parser"
    description = (
        "Parse a kernel dmesg file and return a JSON summary of events. "
        "Input: file path to dmesg file. "
        "Output: JSON with event counts by type, sample events."
    )

    def __init__(self) -> None:
        self._parser = DmesgParser()

    def __call__(self, file_path: str) -> ToolResult:
        """Parse a dmesg file and return a structured event summary."""
        try:
            entries = self._parser.parse_file(file_path)
            type_counts = Counter(e.event_type for e in entries)
            sample_events = {
                event_type: [e.message[:200] for e in entries if e.event_type == event_type][:3]
                for event_type in type_counts
                if event_type != "generic"
            }
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "total_entries": len(entries),
                    "event_type_counts": dict(type_counts),
                    "sample_events": sample_events,
                    "gpu_fault": type_counts.get("gpu_fault", 0) > 0,
                    "oom_kill": type_counts.get("oom_kill", 0) > 0,
                    "kernel_panic": type_counts.get("kernel_panic", 0) > 0,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))


class PatternSearchTool:
    """Tool: Search log files for regex patterns."""

    name = "pattern_search"
    description = (
        "Search a log file for a regex pattern and return matching lines. "
        "Input: JSON with 'file_path' and 'pattern' (regex string) and optional 'max_results' (default 20). "
        "Output: JSON with match count and matching line samples."
    )

    def __call__(self, file_path: str, pattern: str, max_results: int = 20) -> ToolResult:
        """Search a log file for a regex pattern."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            matches: list[dict] = []
            total_match_count = 0
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if compiled.search(line):
                        total_match_count += 1
                        if len(matches) < max_results:
                            matches.append({"line": lineno, "text": line.rstrip()[:300]})
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"pattern": pattern, "total_matches": total_match_count, "samples": matches},
            )
        except re.error as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=f"Invalid regex: {e}")
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))


class RAGRetrieverTool:
    """Tool: Retrieve similar past diagnostic cases from the vector index."""

    name = "rag_retriever"
    description = (
        "Retrieve past diagnostic cases similar to a query description. "
        "Input: natural language query string describing the symptoms. "
        "Output: JSON list of similar past cases with their root causes."
    )

    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the RAG retriever.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the JSONL metadata file with cases.
            model_name: SentenceTransformer model name for embeddings.
        """
        self._model = SentenceTransformer(model_name)
        self._index = faiss.read_index(index_path)
        self._metadata: list[dict] = []
        with open(metadata_path, "r") as f:
            for line in f:
                self._metadata.append(json.loads(line))

    def __call__(self, query: str, top_k: int = 3) -> ToolResult:
        """Retrieve the top-k most similar past diagnostic cases."""
        try:
            embedding = self._model.encode([query], normalize_embeddings=True)
            embedding = np.array(embedding, dtype=np.float32)
            distances, indices = self._index.search(embedding, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self._metadata):
                    case = self._metadata[idx].copy()
                    case["similarity_score"] = float(1 - dist)
                    results.append(case)
            return ToolResult(tool_name=self.name, success=True, data={"similar_cases": results, "query": query})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))

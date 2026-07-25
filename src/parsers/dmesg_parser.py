"""Parser for Linux kernel dmesg text files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DmesgEntry:
    """A single parsed dmesg line.

    Attributes:
        timestamp: Kernel uptime timestamp in seconds (float), if present.
        event_type: Classified event type — one of "gpu_fault", "oom_kill",
            "kernel_panic", "thermal", or "generic".
        message: The raw message body (without the bracketed timestamp).
    """

    timestamp: float | None
    event_type: str
    message: str


_DMESG_LINE_RE = re.compile(r"^\[\s*(?P<timestamp>\d+\.\d+)\]\s*(?P<message>.*)$")

_EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("gpu_fault", re.compile(r"kgsl.*fault|GPU fault|GPU hang|Device hanged", re.IGNORECASE)),
    ("oom_kill", re.compile(r"lowmemorykiller|Killed process|Out of memory: Kill process", re.IGNORECASE)),
    ("kernel_panic", re.compile(r"Kernel panic|BUG:|Oops:", re.IGNORECASE)),
    ("thermal", re.compile(r"thermal|throttl", re.IGNORECASE)),
]


class DmesgParser:
    """Parse kernel dmesg files into structured, classified entries."""

    def parse_file(self, file_path: str | Path) -> list[DmesgEntry]:
        """Parse a dmesg text file.

        Args:
            file_path: Path to the dmesg text file.

        Returns:
            List of parsed DmesgEntry objects. Lines without a bracketed
            kernel timestamp are treated as generic with no timestamp.
        """
        entries: list[DmesgEntry] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                match = _DMESG_LINE_RE.match(line)
                if match:
                    timestamp = float(match.group("timestamp"))
                    message = match.group("message")
                else:
                    timestamp = None
                    message = line
                entries.append(
                    DmesgEntry(
                        timestamp=timestamp,
                        event_type=self._classify(message),
                        message=message,
                    )
                )
        return entries

    def _classify(self, message: str) -> str:
        """Classify a dmesg message into an event type."""
        for event_type, pattern in _EVENT_PATTERNS:
            if pattern.search(message):
                return event_type
        return "generic"

    def find_gpu_faults(self, entries: list[DmesgEntry]) -> list[DmesgEntry]:
        """Return entries classified as GPU faults."""
        return [e for e in entries if e.event_type == "gpu_fault"]

    def find_oom_kills(self, entries: list[DmesgEntry]) -> list[DmesgEntry]:
        """Return entries classified as OOM kills."""
        return [e for e in entries if e.event_type == "oom_kill"]

    def find_kernel_panics(self, entries: list[DmesgEntry]) -> list[DmesgEntry]:
        """Return entries classified as kernel panics."""
        return [e for e in entries if e.event_type == "kernel_panic"]

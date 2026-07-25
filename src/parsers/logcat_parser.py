"""Parser for Android logcat threadtime-format text files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogcatEntry:
    """A single parsed logcat line.

    Attributes:
        timestamp: Raw timestamp string, e.g. "01-15 14:23:07.412".
        pid: Process ID.
        tid: Thread ID.
        priority: Single-letter priority (V, D, I, W, E, F).
        tag: Log tag.
        message: Log message body.
    """

    timestamp: str
    pid: int
    tid: int
    priority: str
    tag: str
    message: str


_LOGCAT_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+(?P<tid>\d+)\s+"
    r"(?P<priority>[VDIWEF])\s+"
    r"(?P<tag>[^:]+):\s?(?P<message>.*)$"
)

_ANR_RE = re.compile(r"ANR in|Input dispatching timed out", re.IGNORECASE)
_CRASH_RE = re.compile(r"FATAL EXCEPTION|AndroidRuntime.*Exception", re.IGNORECASE)
_OOM_RE = re.compile(r"OutOfMemoryError|Out of memory", re.IGNORECASE)


class LogcatParser:
    """Parse Android logcat threadtime-format files into structured entries."""

    def parse_file(self, file_path: str | Path) -> list[LogcatEntry]:
        """Parse a logcat text file.

        Args:
            file_path: Path to the logcat text file.

        Returns:
            List of parsed LogcatEntry objects. Lines that don't match the
            expected threadtime format are silently skipped.
        """
        entries: list[LogcatEntry] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = _LOGCAT_LINE_RE.match(line.rstrip("\n"))
                if not match:
                    continue
                entries.append(
                    LogcatEntry(
                        timestamp=match.group("timestamp"),
                        pid=int(match.group("pid")),
                        tid=int(match.group("tid")),
                        priority=match.group("priority"),
                        tag=match.group("tag").strip(),
                        message=match.group("message"),
                    )
                )
        return entries

    def find_anrs(self, entries: list[LogcatEntry]) -> list[LogcatEntry]:
        """Return entries indicating an ANR (Application Not Responding)."""
        return [e for e in entries if _ANR_RE.search(e.message)]

    def find_crashes(self, entries: list[LogcatEntry]) -> list[LogcatEntry]:
        """Return entries indicating a fatal Java crash."""
        return [
            e
            for e in entries
            if e.priority in ("E", "F") and _CRASH_RE.search(f"{e.tag} {e.message}")
        ]

    def find_ooms(self, entries: list[LogcatEntry]) -> list[LogcatEntry]:
        """Return entries indicating an out-of-memory condition."""
        return [e for e in entries if _OOM_RE.search(e.message)]

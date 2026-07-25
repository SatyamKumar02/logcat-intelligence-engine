"""Extraction and section-splitting for Android bugreport ZIP archives.

Stub for now — full testing against real AOSP bugreports is deferred to the
"real data" stretch phase (see CONTEXT.md). Bugreports are ZIP archives
containing a main text bundle with dashed section headers like
"------ SYSTEM LOG (logcat -v threadtime) ------".
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

_SECTION_HEADER_RE = re.compile(r"^-+\s*(?P<name>.+?)\s*-+$")


class BugreportParser:
    """Extract and split an Android bugreport ZIP into named text sections."""

    def extract_main_text(self, zip_path: str | Path) -> str:
        """Extract the main bugreport-*.txt entry from a bugreport ZIP.

        Args:
            zip_path: Path to the bugreport ZIP archive.

        Returns:
            The full text content of the main bugreport text file.

        Raises:
            FileNotFoundError: If no bugreport-*.txt entry is found in the ZIP.
        """
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.startswith("bugreport-") and n.endswith(".txt")]
            if not candidates:
                raise FileNotFoundError(f"No bugreport-*.txt found in {zip_path}")
            with zf.open(candidates[0]) as fh:
                return fh.read().decode("utf-8", errors="replace")

    def split_sections(self, text: str) -> dict[str, str]:
        """Split bugreport text into named sections by dashed header lines.

        Args:
            text: Full bugreport text as returned by extract_main_text.

        Returns:
            Dict mapping section name (e.g. "SYSTEM LOG (logcat -v threadtime)")
            to that section's raw text body.
        """
        sections: dict[str, list[str]] = {}
        current_name: str | None = None
        for line in text.splitlines():
            match = _SECTION_HEADER_RE.match(line.strip())
            if match:
                current_name = match.group("name")
                sections[current_name] = []
                continue
            if current_name is not None:
                sections[current_name].append(line)
        return {name: "\n".join(lines) for name, lines in sections.items()}

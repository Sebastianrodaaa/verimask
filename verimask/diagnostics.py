"""Parse verifier output into line/token-oriented diagnostic bundles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SourceSpan:
    """Half-open line interval [start_line, end_line) in 1-based source lines."""

    start_line: int
    end_line: int
    kind: str = "unknown"
    message: str = ""

    def expand(self, margin: int) -> SourceSpan:
        return SourceSpan(
            max(1, self.start_line - margin),
            self.end_line + margin,
            kind=self.kind,
            message=self.message,
        )

    def intersects_line(self, line: int) -> bool:
        return self.start_line <= line < self.end_line


@dataclass
class DiagnosticBundle:
    """Structured verifier feedback for policy conditioning."""

    source_kind: str  # "python_test" | "lean"
    message: str = ""
    spans: list[SourceSpan] = field(default_factory=list)
    failing_assert: str | None = None
    lean_goal: str | None = None

    def line_hints(self) -> set[int]:
        hints: set[int] = set()
        for span in self.spans:
            for line in range(span.start_line, span.end_line):
                hints.add(line)
        return hints

    def to_feature_text(self) -> str:
        """Compact text channel for lightweight policies."""
        parts = [f"[{self.source_kind}]", self.message]
        for span in self.spans:
            parts.append(f"span:{span.start_line}-{span.end_line}:{span.kind}")
        if self.failing_assert:
            parts.append(f"assert:{self.failing_assert[:200]}")
        if self.lean_goal:
            parts.append(f"goal:{self.lean_goal[:300]}")
        return "\n".join(p for p in parts if p)


_TRACE_FRAME = re.compile(
    r'File "([^"]+)", line (\d+)(?:, in .*)?',
)
_ASSERT_FAIL = re.compile(r"AssertionError(?::\s*(.*))?", re.IGNORECASE)
_LEAN_ERROR_LINE = re.compile(r"error:\s*(.+)", re.IGNORECASE)
_LEAN_LINE_COL = re.compile(r":(\d+):(\d+):\s*error:")


def parse_python_traceback(
    stderr: str,
    *,
    source_basename: str | None = None,
    default_margin: int = 0,
) -> DiagnosticBundle:
    """
    Extract line spans from a Python traceback.

    Prefer frames whose path ends with ``source_basename`` when provided.
    """
    frames: list[tuple[str, int]] = []
    for match in _TRACE_FRAME.finditer(stderr):
        path, line_s = match.group(1), int(match.group(2))
        frames.append((path, line_s))

    chosen: tuple[str, int] | None = None
    if frames:
        if source_basename:
            for path, line in reversed(frames):
                if path.endswith(source_basename) or path.split("/")[-1] == source_basename:
                    chosen = (path, line)
                    break
        if chosen is None:
            chosen = frames[-1]

    spans: list[SourceSpan] = []
    message = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    failing_assert = None
    am = _ASSERT_FAIL.search(stderr)
    if am:
        failing_assert = (am.group(1) or "AssertionError").strip()

    if chosen is not None:
        _, line = chosen
        span = SourceSpan(line, line + 1, kind="traceback_frame", message=message)
        if default_margin:
            span = span.expand(default_margin)
        spans.append(span)

    return DiagnosticBundle(
        source_kind="python_test",
        message=message,
        spans=spans,
        failing_assert=failing_assert,
    )


def parse_lean_diagnostics(lean_output: str) -> DiagnosticBundle:
    """Parse common Lean 4 error line/column patterns into spans."""
    spans: list[SourceSpan] = []
    goal: str | None = None
    for line in lean_output.splitlines():
        m = _LEAN_LINE_COL.search(line)
        if m:
            ln = int(m.group(1))
            spans.append(SourceSpan(ln, ln + 1, kind="lean_error", message=line.strip()))
        if "unsolved goals" in line.lower():
            goal = line.strip()
    msg_match = _LEAN_ERROR_LINE.search(lean_output)
    if msg_match:
        message = msg_match.group(1).strip()
    else:
        lines = lean_output.strip().splitlines()
        message = lines[-1] if lines else ""
    if not spans and lean_output.strip():
        spans.append(SourceSpan(1, 2, kind="lean_error", message=message))
    return DiagnosticBundle(
        source_kind="lean",
        message=message,
        spans=spans,
        lean_goal=goal,
    )


def lines_to_char_spans(
    source: str,
    line_numbers: Iterable[int],
) -> list[tuple[int, int]]:
    """Map 1-based line numbers to half-open char intervals in ``source``."""
    line_starts: list[int] = [0]
    for idx, ch in enumerate(source):
        if ch == "\n":
            line_starts.append(idx + 1)
    intervals: list[tuple[int, int]] = []
    lines_set = set(line_numbers)
    for line_no in sorted(lines_set):
        if line_no < 1 or line_no > len(line_starts):
            continue
        start = line_starts[line_no - 1]
        end = line_starts[line_no] if line_no < len(line_starts) else len(source)
        intervals.append((start, end))
    return intervals


def char_spans_to_token_indices(
    text: str,
    char_spans: Sequence[tuple[int, int]],
    token_offsets: Sequence[tuple[int, int]],
) -> list[int]:
    """Return token indices whose char spans intersect any char span."""
    indices: list[int] = []
    for ti, (ts, te) in enumerate(token_offsets):
        for cs, ce in char_spans:
            if ts < ce and te > cs:
                indices.append(ti)
                break
    return indices


NL_TOKEN = "⏎"


def naive_tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Whitespace split with explicit newline tokens (for CPU stand-in denoiser)."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\n":
            tokens.append(NL_TOKEN)
            offsets.append((i, i + 1))
            i += 1
            continue
        while i < n and text[i].isspace() and text[i] != "\n":
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append(text[start:i])
        offsets.append((start, i))
    return tokens, offsets


def naive_detokenize(tokens: list[str]) -> str:
    parts: list[str] = []
    for tok in tokens:
        if tok == NL_TOKEN:
            parts.append("\n")
        else:
            if parts and not parts[-1].endswith("\n"):
                parts.append(" ")
            parts.append(tok)
    text = "".join(parts)
    return text.replace(" \n", "\n").replace("\n ", "\n")

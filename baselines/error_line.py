"""Error-line remasking baseline (traceback line ± margin)."""

from __future__ import annotations

from dataclasses import dataclass

from verimask.diagnostics import (
    DiagnosticBundle,
    char_spans_to_token_indices,
    lines_to_char_spans,
    parse_python_traceback,
)
from verimask.policy import SpanDecision


@dataclass
class ErrorLineRemaskBaseline:
    margin: int = 1

    def select_spans(
        self,
        code: str,
        diagnostics: DiagnosticBundle,
        token_offsets: list[tuple[int, int]],
        *,
        max_tokens: int,
        seed: int | None = None,
    ) -> SpanDecision:
        spans = list(diagnostics.spans)
        if not spans and diagnostics.message:
            diag = parse_python_traceback(diagnostics.message, default_margin=self.margin)
            spans = diag.spans
        lines: set[int] = set()
        for span in spans:
            expanded = span.expand(self.margin)
            lines.update(range(expanded.start_line, expanded.end_line))
        char_spans = lines_to_char_spans(code, lines)
        indices = char_spans_to_token_indices(code, char_spans, token_offsets)
        if len(indices) > max_tokens:
            indices = indices[:max_tokens]
        frac = len(indices) / max(1, len(token_offsets))
        return SpanDecision(indices, frac, {"baseline": "error_line", "margin": self.margin})

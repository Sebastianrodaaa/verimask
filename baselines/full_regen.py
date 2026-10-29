"""Full regeneration baseline: mask all tokens."""

from __future__ import annotations

from dataclasses import dataclass

from verimask.diagnostics import DiagnosticBundle
from verimask.policy import SpanDecision


@dataclass
class FullRegenerationBaseline:
    def select_spans(
        self,
        code: str,
        diagnostics: DiagnosticBundle,
        token_offsets: list[tuple[int, int]],
        *,
        max_tokens: int,
        seed: int | None = None,
    ) -> SpanDecision:
        n = len(token_offsets)
        cap = min(n, max_tokens if max_tokens > 0 else n)
        indices = list(range(cap))
        if cap < n:
            # if budget limited, still mark as attempting full regen within budget
            pass
        frac = len(indices) / max(1, n)
        _ = (code, diagnostics, seed)
        return SpanDecision(indices, frac, {"baseline": "full_regen"})

"""RemeDi-style low-confidence remasking baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verimask.diagnostics import DiagnosticBundle
from verimask.policy import SpanDecision


@dataclass
class ConfidenceRemaskBaseline:
    """Uses pseudo-confidence scores (deterministic from token hash if not provided)."""

    confidence: list[float] | None = None

    def select_spans(
        self,
        code: str,
        diagnostics: DiagnosticBundle,
        token_offsets: list[tuple[int, int]],
        *,
        max_tokens: int,
        seed: int | None = None,
    ) -> SpanDecision:
        rng = np.random.default_rng(seed)
        n = len(token_offsets)
        if self.confidence and len(self.confidence) == n:
            scores = np.array(self.confidence)
        else:
            scores = np.array(
                [hash(code[max(0, ts - 3) : te]) % 1000 / 1000.0 for ts, te in token_offsets]
            )
        order = np.argsort(scores)  # lowest confidence first
        chosen = order[: min(max_tokens, n)].tolist()
        frac = len(chosen) / max(1, n)
        _ = (diagnostics, rng)
        return SpanDecision(chosen, frac, {"baseline": "confidence"})

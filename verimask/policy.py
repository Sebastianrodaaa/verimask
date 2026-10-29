"""Remask policies: learned head (stub/trainable) and diagnostic features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from verimask.diagnostics import DiagnosticBundle, char_spans_to_token_indices, lines_to_char_spans


@dataclass
class SpanDecision:
    token_indices: list[int]
    masked_fraction: float
    metadata: dict[str, float | int | str]


class RemaskPolicy(Protocol):
    def select_spans(
        self,
        code: str,
        diagnostics: DiagnosticBundle,
        token_offsets: list[tuple[int, int]],
        *,
        max_tokens: int,
        seed: int | None = None,
    ) -> SpanDecision: ...


@dataclass
class DiagnosticLinearPolicy:
    """
    Lightweight trainable policy: scores tokens by proximity to diagnostic lines
    plus learnable width expansion. Suitable for GRPO/REINFORCE in train_rl.py.
    """

    line_weight: float = 1.0
    width_margin: int = 1
    temperature: float = 0.5
    trained: bool = False

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
        hints = diagnostics.line_hints()
        if not hints and diagnostics.spans:
            hints = set()
            for span in diagnostics.spans:
                hints.update(range(span.start_line, span.end_line))
        if not hints:
            # uniform fallback
            n = len(token_offsets)
            k = min(max_tokens, max(1, n // 10))
            idx = sorted(rng.choice(n, size=k, replace=False).tolist())
            frac = len(idx) / max(1, n)
            return SpanDecision(idx, frac, {"mode": "uniform_fallback"})

        expanded_lines = set()
        for line in hints:
            for d in range(-self.width_margin, self.width_margin + 1):
                expanded_lines.add(line + d)
        char_spans = lines_to_char_spans(code, expanded_lines)
        candidates = char_spans_to_token_indices(code, char_spans, token_offsets)
        if not candidates:
            candidates = list(range(len(token_offsets)))

        scores = np.zeros(len(token_offsets), dtype=np.float64)
        for ti, (ts, te) in enumerate(token_offsets):
            mid_line = code[:ts].count("\n") + 1
            dist = min(abs(mid_line - h) for h in hints) if hints else 10.0
            scores[ti] = self.line_weight / (1.0 + dist)
        if self.trained:
            scores += rng.normal(0, 0.05, size=scores.shape)

        order = np.argsort(-scores)
        chosen: list[int] = []
        for idx in order:
            if idx in candidates or not candidates:
                chosen.append(int(idx))
            if len(chosen) >= max_tokens:
                break
        if not chosen:
            chosen = candidates[:max_tokens]
        masked_fraction = len(chosen) / max(1, len(token_offsets))
        return SpanDecision(
            sorted(set(chosen)),
            masked_fraction,
            {"mode": "diagnostic_linear", "width_margin": self.width_margin},
        )


def load_policy_checkpoint(path: str | None) -> DiagnosticLinearPolicy:
    if not path:
        return DiagnosticLinearPolicy(trained=False)
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DiagnosticLinearPolicy(
        line_weight=float(data.get("line_weight", 1.0)),
        width_margin=int(data.get("width_margin", 1)),
        temperature=float(data.get("temperature", 0.5)),
        trained=bool(data.get("trained", True)),
    )

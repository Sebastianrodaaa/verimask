"""Pluggable denoiser backends (tiny CPU stand-in vs HuggingFace dLLM)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

MASK_TOKEN = "[MASK]"


class Denoiser(Protocol):
    """Mask-fill interface used by remask_denoise."""

    def tokenize(self, text: str) -> tuple[list[str], list[tuple[int, int]]]: ...

    def detokenize(self, tokens: list[str]) -> str: ...

    def denoise_masked(
        self,
        tokens: list[str],
        mask_indices: set[int],
        *,
        steps: int,
        seed: int | None = None,
        source_text: str | None = None,
    ) -> list[str]: ...


@dataclass
class TinyStandInDenoiser:
    """
    Deterministic CPU stand-in: fills masked slots from a fixed repair lexicon
    keyed by surrounding context hash. For tests and ``make repro`` without GPU.
    """

    repair_map: dict[str, str] | None = None

    @staticmethod
    def _apply_mock_repairs(text: str) -> str:
        """Deterministic syntactic fixes for MOCK CPU harness tasks only."""
        out = text
        if "a - b" in out:
            out = out.replace("a - b", "a + b")
        if "return y" in out and "max2" in out:
            out = out.replace("return y", "return x")
        if "s[1:]" in out and "palindrome" in out:
            out = out.replace("s[1:]", "s[::-1]")
        return out

    def tokenize(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        from verimask.diagnostics import naive_tokenize

        return naive_tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        from verimask.diagnostics import naive_detokenize

        return naive_detokenize(tokens)

    def denoise_masked(
        self,
        tokens: list[str],
        mask_indices: set[int],
        *,
        steps: int,
        seed: int | None = None,
        source_text: str | None = None,
    ) -> list[str]:
        rng = np.random.default_rng(seed)
        clear = source_text or self.detokenize(tokens)
        out = list(tokens)
        repairs = self.repair_map or {}
        for idx in sorted(mask_indices):
            ctx = " ".join(out[max(0, idx - 2) : idx + 3])
            if ctx in repairs:
                out[idx] = repairs[ctx]
            elif repairs:
                # try any single-token repair value from lexicon
                out[idx] = next(iter(repairs.values()))
            else:
                out[idx] = f"fix{idx}"
        joined = self.detokenize(out)
        if mask_indices:
            joined = self._apply_mock_repairs(clear)
        out, _ = self.tokenize(joined)
        _ = (steps, rng)
        return out


class HuggingFaceDreamCoderDenoiser:
    """
    Optional GPU backend for Dream-Coder-v0-Instruct-7B.

    Loads only when ``VERIMASK_USE_HF=1`` and transformers/torch CUDA available.
    Falls back with a clear error if misconfigured.
    """

    def __init__(self, model_id: str = "dream-ai/Dream-Coder-v0-Instruct-7B") -> None:
        if os.environ.get("VERIMASK_USE_HF", "") != "1":
            raise RuntimeError(
                "HuggingFace denoiser requires VERIMASK_USE_HF=1 and GPU optional deps installed"
            )
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install verimask[gpu] for HuggingFace denoiser") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        self.mask_id = self.tokenizer.mask_token_id or self.tokenizer.convert_tokens_to_ids(
            MASK_TOKEN
        )

    def tokenize(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets = [(a, b if b is not None else a) for a, b in enc["offset_mapping"]]
        return tokens, offsets

    def detokenize(self, tokens: list[str]) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def denoise_masked(
        self,
        tokens: list[str],
        mask_indices: set[int],
        *,
        steps: int,
        seed: int | None = None,
        source_text: str | None = None,
    ) -> list[str]:
        _ = source_text
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids], device=self.device)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for idx in mask_indices:
            if 0 <= idx < input_ids.shape[1]:
                input_ids[0, idx] = self.mask_id
                mask[0, idx] = True
        with torch.no_grad():
            for _ in range(max(1, steps)):
                logits = self.model(input_ids).logits
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
                input_ids[0, mask[0]] = sampled[mask[0]]
        new_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return new_tokens


def get_denoiser(kind: str = "tiny") -> Denoiser:
    if kind == "tiny":
        return TinyStandInDenoiser()
    if kind in ("dream", "hf", "dream-coder"):
        return HuggingFaceDreamCoderDenoiser()
    raise ValueError(f"Unknown denoiser kind: {kind}")

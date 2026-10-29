"""Autoregressive repair baseline stub (MOCK generation without loaded Qwen)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from verimask.diagnostics import parse_python_traceback
from verimask.verifier import VerifyResult, run_python_tests


@dataclass
class ARRepairBaseline:
    """
    Same-size AR coder placeholder: appends a comment with traceback hint.

    Real runs set ``VERIMASK_USE_HF=1`` and use Qwen2.5-Coder-7B-Instruct; in CPU CI
    we use deterministic stub edits labeled MOCK in results metadata.
    """

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    def repair(self, code: str, test_code: str, stderr: str, *, seed: int = 0) -> str:
        if os.environ.get("VERIMASK_USE_HF") == "1":
            return self._hf_repair(code, test_code, stderr, seed=seed)
        diag = parse_python_traceback(stderr)
        hint = diag.message or "error"
        return code + f"\n# AR_MOCK_REPAIR seed={seed} hint={hint[:80]}\n"

    def _hf_repair(self, code: str, test_code: str, stderr: str, *, seed: int) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        prompt = (
            f"Fix the Python function given the test failure.\n\n"
            f"```{code}```\n\nTraceback:\n{stderr}\n\nFixed code:"
        )
        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = tok(prompt, return_tensors="pt").to(device)
        torch.manual_seed(seed)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
        text = tok.decode(out[0], skip_special_tokens=True)
        return text.split("Fixed code:")[-1].strip()

    def run_loop(
        self, code: str, test_code: str, *, max_attempts: int = 4, seed: int = 0
    ) -> VerifyResult:
        current = code
        last: VerifyResult | None = None
        for k in range(max_attempts):
            last = run_python_tests(current, test_code)
            if last.passed:
                break
            current = self.repair(current, test_code, last.stderr, seed=seed + k)
        assert last is not None
        return last

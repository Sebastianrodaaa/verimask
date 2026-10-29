#!/usr/bin/env python3
"""Build synthetic failure set for CPU repro (MOCK data, real verifier execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verimask.verifier import run_python_tests

# Labeled MOCK: not from Dream-Coder sampling on MBPP+/HumanEval+
SYNTHETIC_TASKS = [
    {
        "task_id": "mock_mbpp_001",
        "code": "def add(a, b):\n    return a - b\n",
        "test_code": "assert add(2, 3) == 5",
        "repair_lexicon": {"return a - b": "return", "a - b": "a + b"},
    },
    {
        "task_id": "mock_mbpp_002",
        "code": "def max2(x, y):\n    if x > y:\n        return y\n    return x\n",
        "test_code": "assert max2(4, 9) == 9\nassert max2(-1, -5) == -1",
        "repair_lexicon": {},
    },
    {
        "task_id": "mock_he_003",
        "code": "def is_palindrome(s):\n    return s == s[1:]\n",
        "test_code": "assert is_palindrome('aba') == True\nassert is_palindrome('ab') == False",
        "repair_lexicon": {},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tests/mock_data/failures.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    examples = []
    for task in SYNTHETIC_TASKS:
        verify = run_python_tests(task["code"], task["test_code"])
        if verify.passed:
            continue
        examples.append(
            {
                "task_id": task["task_id"],
                "code": task["code"],
                "test_code": task["test_code"],
                "repair_lexicon": task.get("repair_lexicon", {}),
                "initial_stderr": verify.stderr,
                "label": "MOCK",
            }
        )
    payload = {
        "dataset": "MOCK_synthetic_failures",
        "note": "Not sampled from Dream-Coder on MBPP+/HumanEval+; for harness repro only.",
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "failures_build_meta.json").write_text(
        json.dumps({"built": str(args.out), "count": len(examples), "label": "MOCK"}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} failing examples to {args.out}")


if __name__ == "__main__":
    main()

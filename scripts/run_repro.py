#!/usr/bin/env python3
"""Run all experiment arms with fixed seeds; write results/*.json."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.ar_repair import ARRepairBaseline
from baselines.confidence import ConfidenceRemaskBaseline
from baselines.error_line import ErrorLineRemaskBaseline
from baselines.full_regen import FullRegenerationBaseline
from verimask.denoiser import TinyStandInDenoiser
from verimask.policy import load_policy_checkpoint
from verimask.remask_denoise import RemaskDenoiseLoop
from verimask.train_rl import FailureExample, TrainConfig, train_policy

SEEDS = [11, 22, 33]
MAX_ATTEMPTS = 4


def run_arm(name: str, loop: RemaskDenoiseLoop, examples: list[dict], seed: int) -> dict:
    repaired = 0
    mask_fracs: list[float] = []
    t0 = time.perf_counter()
    for ex in examples:
        denoiser: TinyStandInDenoiser = loop.denoiser  # type: ignore[assignment]
        if ex.get("repair_lexicon"):
            denoiser.repair_map = ex["repair_lexicon"]
        out = loop.repair(ex["code"], ex["test_code"], seed=seed)
        if out.passed:
            repaired += 1
        if out.decisions:
            mask_fracs.append(out.decisions[-1].masked_fraction)
    elapsed = time.perf_counter() - t0
    n = len(examples)
    return {
        "arm": name,
        "seed": seed,
        "repair_rate": repaired / max(1, n),
        "n": n,
        "mean_masked_fraction": sum(mask_fracs) / max(1, len(mask_fracs)),
        "verifier_calls_budget_k": MAX_ATTEMPTS,
        "wall_clock_s": elapsed,
        "label": "MOCK" if name != "diagnostics_parser_only" else "REAL",
        "denoiser": "TinyStandInDenoiser",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, default=Path("tests/mock_data/failures.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--train", action="store_true", default=True)
    args = parser.parse_args()
    data = json.loads(args.failures.read_text(encoding="utf-8"))
    examples = data["examples"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_examples = [
        FailureExample(ex["code"], ex["test_code"], ex.get("repair_lexicon")) for ex in examples
    ]
    ckpt = args.out_dir / "policy_checkpoint.json"
    if args.train:
        train_policy(
            train_examples,
            TrainConfig(episodes=30, seed=SEEDS[0], checkpoint_path=str(ckpt)),
        )
    policy = load_policy_checkpoint(str(ckpt))

    arms = {
        "verimask_policy": policy,
        "error_line": ErrorLineRemaskBaseline(margin=1),
        "confidence": ConfidenceRemaskBaseline(),
        "full_regen": FullRegenerationBaseline(),
    }

    all_rows: list[dict] = []
    for seed in SEEDS:
        for arm_name, pol in arms.items():
            denoiser = TinyStandInDenoiser()
            loop = RemaskDenoiseLoop(
                denoiser=denoiser,
                policy=pol,
                max_attempts=MAX_ATTEMPTS,
            )
            row = run_arm(arm_name, loop, examples, seed)
            all_rows.append(row)
            out_path = args.out_dir / f"{arm_name}_seed{seed}.json"
            out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

        ar = ARRepairBaseline()
        t0 = time.perf_counter()
        repaired = 0
        for ex in examples:
            res = ar.run_loop(ex["code"], ex["test_code"], max_attempts=MAX_ATTEMPTS, seed=seed)
            if res.passed:
                repaired += 1
        ar_row = {
            "arm": "ar_repair",
            "seed": seed,
            "repair_rate": repaired / max(1, len(examples)),
            "n": len(examples),
            "mean_masked_fraction": 1.0,
            "verifier_calls_budget_k": MAX_ATTEMPTS,
            "wall_clock_s": time.perf_counter() - t0,
            "label": "MOCK",
            "denoiser": "ARRepairBaseline_stub",
        }
        all_rows.append(ar_row)
        (args.out_dir / f"ar_repair_seed{seed}.json").write_text(
            json.dumps(ar_row, indent=2), encoding="utf-8"
        )

    summary = {
        "seeds": SEEDS,
        "arms": sorted({r["arm"] for r in all_rows}),
        "rows": all_rows,
        "note": "MOCK — TinyStandInDenoiser on synthetic failures, not Dream-Coder 7B.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

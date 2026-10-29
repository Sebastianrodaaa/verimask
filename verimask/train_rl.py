"""GRPO / REINFORCE training loop for remask policy (CPU-friendly stub scale)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from verimask.denoiser import TinyStandInDenoiser
from verimask.policy import DiagnosticLinearPolicy
from verimask.remask_denoise import RemaskDenoiseLoop


@dataclass
class TrainConfig:
    episodes: int = 50
    lambda_mask: float = 0.1
    lr: float = 0.05
    seed: int = 42
    algorithm: str = "reinforce"  # or "grpo"
    checkpoint_path: str = "results/policy_checkpoint.json"


@dataclass
class FailureExample:
    code: str
    test_code: str
    repair_lexicon: dict[str, str] | None = None


def reward(passed: bool, masked_fraction: float, lambda_mask: float) -> float:
    return (1.0 if passed else 0.0) - lambda_mask * masked_fraction


def train_policy(
    examples: list[FailureExample],
    config: TrainConfig | None = None,
) -> DiagnosticLinearPolicy:
    cfg = config or TrainConfig()
    rng = np.random.default_rng(cfg.seed)
    policy = DiagnosticLinearPolicy(trained=False)
    denoiser = TinyStandInDenoiser(
        repair_map={k: v for ex in examples for k, v in (ex.repair_lexicon or {}).items()}
    )
    loop = RemaskDenoiseLoop(denoiser=denoiser, policy=policy)

    baseline = 0.0
    for ep in range(cfg.episodes):
        ex = examples[int(rng.integers(0, len(examples)))]
        if ex.repair_lexicon:
            denoiser.repair_map = ex.repair_lexicon
        outcome = loop.repair(ex.code, ex.test_code, seed=cfg.seed + ep)
        mf = outcome.decisions[-1].masked_fraction if outcome.decisions else 0.0
        r = reward(outcome.passed, mf, cfg.lambda_mask)
        adv = r - baseline
        baseline = 0.9 * baseline + 0.1 * r
        # Simple REINFORCE-style nudge on width margin
        if adv > 0 and not outcome.passed:
            policy.width_margin = min(5, policy.width_margin + 1)
        elif adv < 0 and outcome.passed:
            policy.width_margin = max(0, policy.width_margin - 1)
        policy.line_weight += cfg.lr * adv * 0.1
    policy.trained = True
    ckpt = Path(cfg.checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(asdict(policy), indent=2), encoding="utf-8")
    return policy


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train verimask remask policy (CPU stub)")
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/policy_checkpoint.json"))
    args = parser.parse_args()
    data = json.loads(args.failures.read_text(encoding="utf-8"))
    examples = [FailureExample(**row) for row in data["examples"]]
    train_policy(examples, TrainConfig(episodes=args.episodes, checkpoint_path=str(args.out)))


if __name__ == "__main__":
    main()

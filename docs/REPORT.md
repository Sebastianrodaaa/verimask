# verimask research report (v0.1 scaffold)

## Motivation

Masked diffusion language models (dLLMs) can revise arbitrary token subsets in parallel, but **where** to remask dominates repair success when the denoiser is frozen. Test runners and proof assistants emit **structured diagnostics** (traceback frames, assertion failures, Lean goals). verimask learns a **small remask policy** that maps `(program, diagnostics)` → **span masks + width**, then calls a **frozen** dLLM (target: [Dream-Coder-v0-Instruct-7B](https://huggingface.co/dream-ai/Dream-Coder-v0-Instruct-7B), Apache-2.0) to re-denoise only those positions. Training optimizes **verifier pass** minus **λ × masked fraction** (GRPO/REINFORCE).

See [RELATED_WORK.md](./RELATED_WORK.md) for positioning vs Diffusion-Proof, RemeDi, MDM-VGB, execution-trace heuristics, and the CanItEdit “composition gap” paper.

## Method

1. **Verify** candidate program (Python tests in v1; Lean in v1.5).
2. **Parse** diagnostics → line hints (`verimask/diagnostics.py`).
3. **Policy** scores tokens (learned linear/diagnostic policy in v0.1; probe on dLLM hiddens in planned v1).
4. **Remask + denoise** masked spans (`verimask/remask_denoise.py`) for up to **K** attempts.
5. **RL** updates policy with REINFORCE-style rewards (`verimask/train_rl.py`).

Baselines: `error_line`, `confidence` (RemeDi-style), `full_regen`, `ar_repair` (Qwen2.5-Coder-7B-Instruct when GPU+HF enabled).

## Experiment design (from brief)

| Item | Plan |
| --- | --- |
| Denoiser | Dream-Coder-v0-Instruct-7B (frozen) |
| AR baseline | Qwen2.5-Coder-7B-Instruct + traceback |
| Data | Failing samples from Dream-Coder on MBPP+ / HumanEval+ |
| Policy | Per-token remask head; GRPO or REINFORCE |
| Metrics | Repair rate @ K, masked fraction, NFEs, wall-clock, 3 seeds |
| Hypothesis | ≥5 pt absolute repair vs best baseline @ K=4; fail if &lt;2 pt inside seed variance |

## What was actually run (this repository checkout)

| Run | Environment | Label |
| --- | --- | --- |
| `pytest tests/test_diagnostics.py` | CPU, CI | **REAL** — traceback + Lean parser |
| `pytest` full suite | CPU | **REAL** harness; **MOCK** denoiser |
| `make repro` | CPU, `TinyStandInDenoiser`, synthetic failures | **MOCK** — not Dream-Coder / MBPP+ |
| Dream-Coder 7B repair | Not run (no GPU in cloud agent) | **Not run** |
| Policy RL @ scale | Stub 30-episode REINFORCE on MOCK set | **MOCK** |
| Demo GIF | `scripts/generate_demo_gif.py` | **MOCK** scripted frames |

Command log for reproducibility:

```bash
pip install -e ".[dev]"
make lint && make test
make repro && make demo && make readme-table
```

## Results (MOCK harness)

Aggregates come from `results/summary.json` after `make repro` (2026-10-29, this agent run). All rows carry `"label": "MOCK"` and `"denoiser": "TinyStandInDenoiser"`. **Do not cite these numbers as Dream-Coder benchmarks.**

Observed mean repair rate on 3 synthetic tasks × 3 seeds (n=3 tasks per run):

| Arm | Mean repair rate |
| --- | ---: |
| `verimask_policy` | 0.667 |
| `confidence` | 0.667 |
| `full_regen` | 0.667 |
| `error_line` | 0.000 |
| `ar_repair` | 0.000 |

The CPU stand-in applies deterministic string fixes when any token is masked; several arms tie. `error_line` often misses line alignment on wrapped test tracebacks (harness limitation). AR stub appends comments only. These outcomes validate the **loop**, not the research hypothesis.

## Limitations

- No GPU run of Dream-Coder or Qwen baselines in this scaffold.
- Synthetic failure set only; MBPP+/HumanEval+ pipeline is scripted but not executed against HF models here.
- `HuggingFaceDreamCoderDenoiser` is best-effort optional code path (requires `verimask[gpu]` and `VERIMASK_USE_HF=1`).
- Execution-trace guided remasking (FARS/Analemma) overlaps at the heuristic level; verimask’s claim rests on **learned** span/width policies beating those rules at matched **K**.

## Next steps

1. **v1 GPU:** Sample failures with Dream-Coder on MBPP+/HumanEval+; run all arms @ K=4, 3 seeds; populate `results/*.json` without MOCK label.
2. **v1.5 Lean:** Wire `parse_lean_diagnostics` to miniF2F; light SFT on LLaDA-8B or Dream; report even near-zero pass rates.
3. **v2 co-training:** Joint policy + denoiser RL (harness co-evolution signal).
4. **Baselines:** Implement MDM-VGB repair mode as optional test-time arm; compare to learned policy.

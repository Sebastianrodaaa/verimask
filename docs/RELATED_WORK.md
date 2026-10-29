# Related work

Last checked: **2026-10-29** (arXiv, Hugging Face Papers, GitHub). Citations below were verified against live sources; arXiv IDs in the project brief match the papers as published (arXiv uses YYMM numbering in this corpus).

## Summary positioning

**No published work we found implements exactly what verimask proposes:** a **small, separately trained remask policy** that reads **program + structured verifier diagnostics** (traceback, assert failure, Lean goal/span) and outputs **which token spans to remask and how wide**, with the **dLLM frozen**, optimized by **RL** on verifier pass minus a **masked-fraction** penalty.

The closest overlap is **execution-trace / error-line heuristic remasking** for code (see below). That validates the *idea* of diagnostic-guided remasking but leaves the **learned localization + width** gap that [How Should Diffusion LMs Edit Code?](https://arxiv.org/abs/2609.38257) identifies for dLLMs in general. **verimask** is positioned as the RL-trained policy layer on top of a frozen denoiser (e.g. [Dream-Coder-v0-Instruct-7B](https://huggingface.co/dream-ai/Dream-Coder-v0-Instruct-7B)), with explicit baselines and a reproducible harness.

---

## Papers listed in the project brief (verified)

### Diffusion-Proof — dLLM-Corrector for Lean

- **Link:** [arXiv:2606.19315](https://arxiv.org/abs/2606.19315) · [ACL Anthology EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.1367/)
- **What it does:** Trains **dLLM-Prover-7B** and **dLLM-Corrector-7B** for Lean4; on verification failure with a correct skeleton, the corrector **masks a fixed-size block (e.g. 256 `<|MASK|>` tokens)** around the failing subgoal and denoises with bidirectional context.
- **Gap vs verimask:** Region choice follows **subgoal structure / fixed masking protocol**, not a **learned policy from Lean error spans and goals**. No RL on remask locations with a sparsity penalty; corrector is **fine-tuned**, not a frozen Dream-Coder + external policy head.

### How Should Diffusion Language Models Edit Code?

- **Link:** [arXiv:2609.38257](https://arxiv.org/abs/2609.38257) · DOI [10.48550/arXiv.2609.38257](https://doi.org/10.48550/arXiv.2609.38257)
- **What it does:** Compares editing interfaces (whole-file, search-and-replace, locate-then-infill, token-level) on **CanItEdit**; shows a **composition gap**: strong infilling when **gold locations** are given, weak when locations must be **predicted**; **boundary width** strongly affects success.
- **Gap vs verimask:** Does **not** train a diagnostic-conditioned remask policy for **test/Lean verifiers** or **code repair from failing programs**; motivates verimask’s target problem (localization + width) directly.

### RemeDi — self-reflective remasking

- **Link:** [arXiv:2509.23653](https://arxiv.org/abs/2509.23653) · Code: [maple-research-lab/RemeDi](https://github.com/maple-research-lab/RemeDi)
- **What it does:** Dual-stream dLLM predicts tokens and **per-token confidence**; low-confidence tokens are **remasked during generation**; **Remask SFT + Remask RL** on outcome rewards (math/code benchmarks).
- **Gap vs verimask:** Remasking signal is **internal model confidence**, not **external verifier diagnostics** (traceback line, failing assert, Lean message). Policy is **inside** the dLLM, not a **small head on frozen denoiser hidden states** + diagnostic tokens.

### MDM-VGB — value-guided backtracking

- **Link:** [arXiv:2606.28301](https://arxiv.org/abs/2606.28301) · Code: [KraitGit/MDM-VGB](https://github.com/KraitGit/MDM-VGB)
- **What it does:** **Test-time** sampler: unmask/remask moves biased by **estimated value** of partial configurations under an **outcome reward**; supports repair from low-reward complete samples.
- **Gap vs verimask:** **Not** a learned **span policy** conditioned on **structured diagnostics**; uses **MCMC-style backtracking** with value estimates, not GRPO/REINFORCE on a compact policy reading traceback/Lean spans. Benchmarks emphasize constraints (Sudoku, QM9), not MBPP+/HumanEval+ repair loops.

### Diffusion is a code repair operator and generator

- **Link:** [arXiv:2508.11110](https://arxiv.org/abs/2508.11110)
- **What it does:** **CodeFusion**-style continuous diffusion: **add noise to entire snippet** and resume reverse diffusion for last-mile **syntactic** repair (Python, Excel, PowerShell).
- **Gap vs verimask:** **No localization** — global noising/resume; not masked discrete dLLM span remasking; no verifier-diagnostic conditioning.

---

## Closer work discovered in search (not in original brief)

### Execution-Trace Guided Remasking for Diffusion Code Generation

- **Link:** [Analemma / FARS paper page](https://analemma.ai/papers/61f58233-c631-4a58-a6c5-40b44fdf3e85/) · [PDF](https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_61f58233-c631-4a58-a6c5-40b44fdf3e85/main.pdf)
- **What it does:** On test failure, **parse traceback** (line ± window) or **line-tracer** for assert failures → map lines to tokens → **conditional denoise** masked positions; compares to low-confidence / CORE-style repair on **MBPP+**.
- **Overlap:** Same high-level loop: **verifier feedback → localize → remask → dLLM repair**.
- **Remaining gap (verimask focus):** **Fixed hierarchical heuristics**, not **RL-learned** span/width; no **Lean** diagnostic channel; no **frozen Dream-Coder + separate policy** ablation; width is rule-based (±w lines, top-K confidence inside region). verimask should **subsume error-line remasking as a baseline** and test whether **learned** policies beat these rules at equal verifier budget **K**.

### Diff Diffusion: Predicting Edit Paths with Diffusion Models for Efficient Code Editing

- **Link:** [arXiv:2610.17831](https://arxiv.org/abs/2610.17831)
- **What it does:** Predicts **structured edit paths** (insert/delete/replace) for IDE-style edits with dLLMs.
- **Gap:** Edit prediction from cursor context, not **verifier-driven repair** of failing programs.

### Diffusion Models for Code Verification (Lean)

- **Link:** [PDF — Eisenach et al.](https://carson.eisenach.org/pdf/diffusion_models_for_code_verification.pdf)
- **What it does:** Hybrid 8B model for **Lean verification** with **local + global refinement** driven by **compiler/verifier** feedback; self-speculative decoding.
- **Gap:** End-to-end **proof search** training, not a **modular remask policy** on a frozen code dLLM for **Python tests**; global refinement is a **controller action**, not diagnostic-token span RL.

### mend.rs (Rust, compiler-verified repair)

- **Link:** [amalshehu/mend.rs](https://github.com/amalshehu/mend.rs)
- **What it does:** Trainable **masked-diffusion** repair for Rust with **`rustc`** verification.
- **Gap:** Domain-specific stack; not diagnostic-conditioned span policy + frozen 7B dLLM baselines on MBPP+/HumanEval+.

### Structure-Preserving Discrete Diffusion (DiffusionCoder)

- **Link:** [DOI 10.1145/3785706.3785940](https://doi.org/10.1145/3785706.3785940)
- **What it does:** Execution-guided sampling and test signals **inside** diffusion for code generation.
- **Gap:** Monolithic generation framework, not **learned remask localization** with explicit **masked-fraction** cost for a frozen dLLM.

---

## GitHub search notes (2026-10-29)

Queries along the lines of `dllm lean repair`, `masked diffusion code repair`, and `verifier remask` did **not** surface a public repo matching **learned diagnostic remask policy + frozen Dream-Coder + RL** end-to-end. Individual pieces exist (RemeDi, MDM-VGB, Diffusion-Proof training recipes, mend.rs).

---

## Implication for verimask

1. **Do not claim** “first ever verifier-guided remasking for code” — execution-trace guided remasking and Diffusion-Proof’s corrector are prior art.
2. **Do claim** (falsifiable): a **learned**, **diagnostic-conditioned** remask policy with **width control** and **RL sparsity penalty**, with a **frozen** Apache-2.0 dLLM, beats **error-line**, **confidence**, **full regen**, and **AR repair** at matched verifier calls **K** on held-out failing programs.
3. **Integrate** execution-trace style rules as **`baselines/error_line.py`** and cite the FARS/Analemma paper as concurrent/heuristic baseline.
4. **v1.5 track:** Diffusion-Proof and Lean verification papers justify a **Lean miniF2F** arm using **Lean error spans** in `diagnostics.py` — still distinct if the **policy** is learned rather than fixed subgoal blocks.

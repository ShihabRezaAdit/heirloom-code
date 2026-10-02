# HEIRLOOM — Supervisor Decision Package (P0 boundary)

Prepared 28 September 2026 from the repo state and Claude_Code_Plan.txt.
Audience: Professor Zhan (decisions), the student and collaborator (context).

> **The 1.5B P0 pilot has not started. All work so far is environment,
> infrastructure, smoke testing, and feasibility validation.** No experimental
> result exists; no attack/poisoning code has been written.

---

## 1. Completed technical readiness (measured / implemented facts)

- Phase B ML environment installed and validated (torch 2.13.0+cu130,
  transformers 5.17, TRL 1.14, PEFT 0.21, bitsandbytes 0.50.2).
- Qwen2.5-1.5B-Instruct pinned to commit `989aa79…`, downloaded, weights
  SHA-256 verified, reproducibly loaded.
- Model-load smoke test passed (auto / fp16 / int8 / nf4).
- QLoRA training smoke test passed (forward/backward/optimizer/save/exact
  reload/generate).
- Deterministic training/evaluation settings implemented and wired into the
  harness (strict deterministic algorithms + cuBLAS workspace + cuDNN off +
  seeded DataLoader); bit-identical losses verified.
- B1-neutral data infrastructure implemented (pinned-revision sources, stable
  row IDs, schema, hash/grouped splitting, filtering + dedup frameworks,
  token-length, manifests + correspondence check, clean-control from existing
  labels, transform interface). No attack code.
- B2-neutral evaluation/metrics infrastructure implemented (net / Δ net /
  survival-ratio-with-required-floor, per-seed bootstrap CIs, judge interface
  + serialized cluster protocol, model/generation backend interfaces, utility
  interface, result schema + export, eval registry). No judge/metric chosen.
- Concrete local Hugging Face backend implemented (pinned-revision only,
  bf16/nf4, deterministic generation).
- DPO pair-length distribution measured on the full pinned PKU-SafeRLHF train
  split (73,907 rows, no filtering/truncation).
- Benign DPO memory feasibility measured on the RTX 4050.
- 72 unit tests passing.

### Key RTX 4050 engineering evidence (NOT scientific results)
- NF4 QLoRA local DPO is technically feasible.
- NF4 fits through 768 tokens/branch without spilling into shared memory
  (peak ≤ 2.23 GiB allocated; device use ≤ 4.28 of 6.0 GiB).
- Measured DPO "simultaneous" length (both branches resident in a step)
  p99 = 576 tokens; NF4 fits comfortably past this.
- bf16 LoRA spills by 384 tokens/branch and is not a practical local DPO path.

---

## 2. Remaining blockers

### A. Technical blocker for P0
- **Judge hosting / execution environment for the primary judge.** The 13B
  HarmBench classifier cannot run on the 6 GB RTX 4050. A serialized
  request/response protocol for a cluster/external judge exists, but the host
  does not. Without a judge, valid Gate 1 attack metrics cannot be produced.

### Deferred, NOT required to start P0
- vLLM (generation backend; only relevant if Gate 1 must match the 7B path).
- llama.cpp / GGUF (E2 quantization arm).
- GPTQ environment (E2).
- mergekit environment (E4).
These are for later experiments and none is needed to begin P0.

### B. Supervisor / research decisions (none defaulted in code)
1. Target attack-strength band for matching.
2. Final Gate 1 criterion wording (three variants currently exist).
3. Judge model and judge host.
4. NF4 QLoRA vs bf16 LoRA for the pilot.
5. Gate 1 generation backend (Hugging Face vs vLLM).
6. Utility metric and its tolerance.
7. Survival-ratio denominator floor value.
8. Split proportions (train / calibration / held-out).
9. Filtering thresholds (e.g. minimum length).
10. Deduplication threshold.
11. False-activation benign prompt set.
12. E5 (distillation) November scope.
13. One vs two model families.
14. Llama-Guard gated access.
15. Registration date for the venue.

---

## 3. Recommended decision table

Scientific choices are left open on purpose.

| Decision | Evidence available | Why needed before P0 | Supervisor decision required? |
|---|---|---|---|
| Target attack-strength band | Proposed 60–70 net ASR (unconfirmed); pilot measures the actual reachable band | Checkpoint matching and the Gate 1 pass/fail test both reference the band | Yes |
| Gate 1 wording | Three variants in the docs; criteria list drafted | The pilot's go/no-go is evaluated against it | Yes (reconcile) |
| Judge model + host | Interface + cluster protocol ready; 13B judge can't run locally | Attack metrics (ASR) need a judge; local hosting impossible | Yes |
| Pilot precision (NF4 vs bf16 LoRA) | NF4 feasible through 768 tok/branch; bf16 spills by 384 tok/branch | Fixes the training + loading path; affects outputs and comparability | Yes (engineering evidence provided) |
| Gate 1 generation backend | HF backend ready; vLLM deferred but co-resolves | Decoding path must match the eventual 7B study or be a recorded deviation | Yes |
| Utility metric + tolerance | Interface ready; IFEval/MMLU/XSTest named in the plan | Gate 1 utility criterion needs a defined measure and threshold | Yes |
| Survival-ratio floor | Code requires it (proposed 20 points, unconfirmed) | Metric is undefined until the floor is set | Yes |
| Split proportions | Grouped splitter ready; ratios are required inputs | Data can't be frozen without them | Yes |
| Filtering thresholds | Framework ready; no thresholds set | Row selection can't be finalized | Yes |
| Dedup threshold | Framework ready; threshold is a required input | Leakage control can't run without it | Yes |
| False-activation set | Metric interface ready; set undefined | Needed for the false-activation measurement | Yes |
| E5 scope | Cost noted; deferral is provisional | Affects whether E5 is in the November plan | Yes |
| One vs two families | Cost noted; one-family chosen provisionally | Affects contribution 2 and the schedule | Yes |
| Llama-Guard access | Gated; not requested | Secondary judge for the agreement protocol | Yes |
| Registration date | Unknown; site not configured | Missing it loses the cycle | Yes (confirm) |

---

## 4. Exact items to resolve before P0 can start

Minimum to unlock P0:
1. A judge host (the primary judge on a suitable GPU / cluster).
2. Pilot precision decision (NF4 QLoRA is the feasible local option).
3. Target attack-strength band.
4. Reconciled Gate 1 wording, including the utility metric + tolerance and
   the survival-ratio floor.
5. Split proportions, filtering thresholds and the dedup threshold (needed to
   freeze the pilot data), and the false-activation benign set (needed for
   that metric).
6. Gate 1 generation backend policy.

Deferrable past P0 start: E5 scope, one-vs-two families, Llama-Guard access,
registration date, vLLM/llama.cpp/GPTQ/mergekit.

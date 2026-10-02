# Phase B (ML stack) install proposal — NOT INSTALLED

> Updated by decisions of 28 Sep 2026 (see `PHASE_B_COMPATIBILITY.md`):
> vLLM deferred (smoke test uses Hugging Face generation; revisit before
> Gate 1); WSL memory proposal changed to 16 GB + 8 GB swap
> (`wslconfig.proposed`); no WSL backup snapshot.

Status: proposal only, awaiting approval. Prepared 28 Sep 2026 after B0
completion (commit ecd054f). Versions below come from PyPI metadata queried
on 28 Sep 2026; nothing has been resolved or installed yet.

## 1. Detected baseline (runtime, not fixed)

| Item | Value |
|---|---|
| Machine | RTX 4050 Laptop GPU, 6141 MiB, compute capability 8.9 |
| Driver | 610.62 (CUDA UMD 13.3) — supports CUDA 13.0 and 12.x runtimes |
| WSL2 | Ubuntu 24.04.4, 11.5 GiB RAM visible (half of 24 GB host) |
| Python | 3.11.16 (uv) |
| Free disk | WSL ext4 (`HEIRLOOM_DATA_ROOT`): 953 GiB; G: 74 GiB |

## 2. Proposed environments (isolated so each can be removed independently)

| venv | Purpose | When |
|---|---|---|
| `~/.venvs/heirloom` | Phase A foundation (existing, kept untouched as rollback) | now |
| `~/.venvs/heirloom-ml` | Core ML stack for data, training, eval | Phase B-1 |
| `~/.venvs/heirloom-quant` | GPTQ tooling (`gptqmodel` pins `numpy==2.2.6`, which conflicts with the core stack) | later (E2) |
| `uv tool install mergekit` | `mergekit` pins `accelerate~=1.6`, conflicting with TRL/gptqmodel; it is an external CLI anyway | later (E4) |
| llama.cpp (built from source) | GGUF conversion/inference | later (E2) |

## 3. Core stack, Phase B-1 (candidate pins, resolved with `uv --dry-run` first)

| Package | Version | Note |
|---|---|---|
| torch | 2.13.0 | Default PyPI Linux wheel ships the CUDA 13.0 runtime; driver 610 supports it. Chosen because vLLM 0.30 pins exactly `torch==2.13.0`. |
| transformers | 5.17.0 | vLLM needs ≥5.10.4; TRL ≥4.56.2 |
| datasets | 5.0.1 | |
| accelerate | 1.15.0 | |
| peft | 0.21.0 | |
| trl | 1.14.0 | its `vllm` extra allows vLLM ≤0.30.0 |
| bitsandbytes | 0.50.2 | torch ≥2.4, <3; supports cc 8.9 |
| scipy, statsmodels, matplotlib | latest compatible | analysis |
| vllm | 0.30.0 | **Phase B-1b, optional** (see §4) |

Compatibility is checked from declared metadata only; the first approved step
is a dry-run resolution, shown before anything is installed. Exact versions
are then frozen to `requirements-ml.lock.txt`.

## 4. Is vLLM needed for the 1.5B work?

- **Foundation/pipeline smoke test:** no. Hugging Face `generate` is enough to
  prove the pipeline runs.
- **Gate 1 measurements:** the plan specifies one generation path (vLLM,
  greedy, 512 tokens). The generation backend is an evaluation-protocol
  detail: if the pilot's gate numbers came from a different backend than the
  7B study, that would be a recorded deviation. Recommendation: defer vLLM
  for the smoke test, but produce Gate 1 numbers with the same backend the
  main study will use (install B-1b before gate-quality evaluation, or run
  pilot evaluation where the main evaluation will run).

## 5. Expected disk usage (estimates, not measurements)

| Item | Size basis | Approx. |
|---|---|---|
| Core venv (torch + CUDA 13 libs + stack) | wheel sizes | ~8–10 GB |
| vLLM + flashinfer (B-1b) | wheel sizes | ~2–4 GB more |
| Qwen2.5-1.5B-Instruct (bf16) | 1.54 B params × 2 bytes | ~3.1 GB |
| Qwen2.5-7B-Instruct (bf16) | 7.62 B × 2 bytes | ~15 GB |
| HarmBench-Llama-2-13b-cls (bf16) | 13 B × 2 bytes | ~26 GB |
| Llama-Guard-3-8B (bf16) | 8 B × 2 bytes | ~16 GB |
| Pilot adapters/checkpoints | LoRA only | a few GB |

All model/data caches go under `HEIRLOOM_DATA_ROOT` (WSL ext4, 953 GiB free),
not G: (74 GiB) and never git.

## 6. RAM / VRAM pressure

- **VRAM (6 GB):** a 1.5B model in bf16 occupies ~3.1 GB of weights, leaving
  limited headroom for activations. LoRA training is feasible at the
  engineering level with small micro-batches, gradient accumulation to keep
  the specified effective batch, and gradient checkpointing. These are
  engineering settings recorded in provenance; they do not change the
  scientific configuration. Actual peak memory must be measured, not assumed.
- **Judges:** the 13B primary judge (~26 GB bf16) does **not** fit in 6 GB.
  Quantizing it to make it fit would change the measuring instrument between
  pilot and main study, so it must not be done silently. Judge hosting stays
  a pilot dependency (cluster, or an approved alternative).
- **System RAM:** 11.5 GiB in WSL is enough for installing and for loading a
  1.5B model, but tight when a model load, tokenisation of a dataset and
  bitsandbytes/INT8 conversion coincide.

## 7. `.wslconfig` recommendation

Raise the WSL cap moderately (requires `wsl --shutdown`, i.e. restarting all
WSL sessions; file is `C:\Users\shiha\.wslconfig`):

```ini
[wsl2]
memory=18GB
swap=8GB
```

This leaves ~6 GB for Windows. Revert by deleting the file and running
`wsl --shutdown`.

## 8. Which machine for what (hardware is a runtime variable)

| Work | RTX 4050 laptop (6 GB) | RTX 5070 laptop | University cluster |
|---|---|---|---|
| Unit tests, provenance, data pipeline (CPU) | yes | yes | yes |
| 1.5B load / LoRA smoke test | yes (tight) | yes | yes |
| 1.5B pilot training runs | feasible, to be measured | likely easier | yes |
| Primary 13B judge, secondary 8B judge | **no** | no at bf16 (VRAM must be detected; RTX 5070 *laptop* parts commonly have 8 GB, not the 12 GB the planning documents assume) | **yes** |
| 7B E1–E4, E5 teachers, E6 full-parameter | no | no | **yes** |

Every run records the detected machine (`scripts/detect_env.py` /
`provenance.json`). Engineering settings may differ per machine; scientific
settings may not without an approved deviation.

## 9. Rollback / removal plan

1. Phase A venv (`~/.venvs/heirloom`) is never modified by Phase B.
2. Remove the ML stack: `rm -rf ~/.venvs/heirloom-ml` (and `-quant`).
3. Remove caches: `rm -rf "$HEIRLOOM_DATA_ROOT"/hf_cache` (no data in git).
4. Optional safety snapshot before installing:
   `wsl --export Ubuntu D:\wsl-backup\ubuntu-pre-phaseB.tar` (restore with
   `wsl --import`). Needs your approval and a target drive.
5. Revert `.wslconfig` by deleting it and `wsl --shutdown`.
6. Lock files make any working state reproducible.

## 10. Blockers, separated

### Environment blockers
- ML stack not installed (this proposal).
- WSL RAM cap 11.5 GiB (§7).
- T0.6 torch/bitsandbytes build check can only run after install.
- 13B judge cannot run on the current GPU; cluster access and allocation are
  UNKNOWN from the project files.
- Gated model access (Llama-Guard-3-8B) UNKNOWN (OD-4); a Hugging Face token
  is not configured.
- RTX 5070 laptop VRAM unknown until detected.

### Research-design blockers (not decided here; no values invented)
- Target attack-strength band for matching (O-5).
- Final Gate 1 wording (criterion 5 differs between the Claude plan and D-50 /
  the supervisor-facing documents).
- Judge-hosting decision (where the judges run for gate-quality numbers).
- November scope / E5 deferral (O-10 / OD-2) and one vs two families (O-9).
- Undefined data/metric details listed in the analysis (split proportions,
  filter values, benign set, utility measure for the 3-point limit).
- Registration date (OD-1).

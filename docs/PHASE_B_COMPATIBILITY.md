# Phase B-1 compatibility report (dependency resolution only; NOT INSTALLED)

Date: 28 Sep 2026. Machine: runtime-detected (current: RTX 4050 Laptop, WSL2).
Resolution input: `requirements-ml.in`. Output: `requirements-ml.lock.proposed.txt`.
Method: `uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28`
(metadata resolution; nothing installed). bitsandbytes wheel contents were
inspected from a temporary download that was then deleted.

## Result: resolves cleanly

94 pinned packages, **no hard or range conflicts**. Phase A pins are unchanged
(numpy 2.4.6, pandas 3.0.6, pyyaml 6.0.3, psutil 7.2.2, pytest 9.1.1).

## Core versions

| Package | Pinned |
|---|---|
| Python | 3.11.16 (uv) |
| torch | 2.13.0 (manylinux_2_28, cp311) |
| triton | 3.7.1 (via torch) |
| transformers | 5.17.0 |
| tokenizers | 0.23.2 |
| huggingface-hub | 1.33.0 |
| safetensors | 0.8.0 |
| datasets | 5.0.1 (pyarrow 25.0.1) |
| accelerate | 1.15.0 |
| peft | 0.21.0 |
| trl | 1.14.0 |
| bitsandbytes | 0.50.2 |
| scipy / statsmodels / matplotlib | 1.17.1 / 0.15.0 / 3.11.2 |

## CUDA / GPU compatibility

| Check | Finding | Status |
|---|---|---|
| CUDA runtime expected by torch | 13.0 (cuda-toolkit 13.0.3, cuDNN 9.20 cu13, NCCL 2.29.7 cu13) | from metadata |
| Driver | 610.62 reports CUDA 13.3; a CUDA 13.0 runtime needs an equal-or-newer driver | compatible |
| bitsandbytes CUDA binary | wheel bundles `libbitsandbytes_cuda130.so` (also 118–132) | matches torch |
| Compute capability 8.9 (Ada) | CUDA 13 dropped only older architectures (below 7.5); 8.9 is supported | expected OK; **verify after install** with `torch.cuda.get_arch_list()` and a bf16 matmul (T0.6) |
| glibc | Ubuntu 24.04 (glibc 2.39) ≥ manylinux_2_28 | compatible |
| WSL GPU passthrough | `nvidia-smi` works inside WSL | OK |

Residual concern: compatibility of the CUDA 13 wheel on WSL is established
from metadata and driver version, not from execution. The T0.6 build check
after installation is the confirmation and stops everything if it fails.

## Separate / deferred

| Item | Reason | When |
|---|---|---|
| vLLM 0.30.0 | Deferred by decision; first smoke test uses Hugging Face generation. It **does** co-resolve with these pins (checked), so adding it later needs no downgrade. Revisit before Gate 1 or any result compared with 7B. | later |
| gptqmodel | pins `numpy==2.2.6` (conflicts with numpy 2.4.6) | separate venv, E2 |
| mergekit | pins `accelerate~=1.6` (conflicts with trl/accelerate 1.15) | `uv tool`, E4 |
| llama.cpp | external tool | E2 (GGUF arm) |

## Size estimates (not measurements)

- Download: **2.76 GiB** compressed wheels (torch 502 MiB, cuBLAS 404 MiB,
  cuDNN 349 MiB, ...).
- Installed venv: roughly **6–7 GiB** (typical expansion of these wheels).
- uv cache: up to ~3 GiB more (clearable with `uv cache clean`).
- Location: `~/.venvs/heirloom-ml` and `~/.cache/uv` on WSL ext4 (953 GiB free).
  Nothing on G:.
- RAM during install: low; the 16 GB `.wslconfig` is for later model work.

## First-install command (after approval only)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/g/HEIRLOOM Project/heirloom-code"
uv venv --python 3.11 ~/.venvs/heirloom-ml
uv pip sync --python ~/.venvs/heirloom-ml/bin/python requirements-ml.lock.proposed.txt
uv pip install --python ~/.venvs/heirloom-ml/bin/python --no-deps -e .
~/.venvs/heirloom-ml/bin/python -m pytest -q      # Phase A/B0 tests in the new env
# then: T0.6 GPU/toolchain check (no model download)
```

`uv pip sync` installs exactly the lock set and nothing else.

## Rollback

```bash
rm -rf ~/.venvs/heirloom-ml     # removes the entire ML environment
uv cache clean                  # optional: frees the download cache
```

The Phase A environment `~/.venvs/heirloom` is never touched.

## Blockers, kept separate

Environment (neutral, addressable now): ML stack not installed; WSL RAM cap
(proposal in `docs/wslconfig.proposed`); T0.6 check pending install; cluster
access unknown; Hugging Face token / gated access not configured.

Research / supervisor decisions (NOT decided here; they do not block this
environment work, but they block interpreting any pilot output as Gate 1
evidence): target attack-strength band; final Gate 1 criterion wording; judge
execution location; E5 November-scope question; one- vs two-family scope;
cluster access; Llama-Guard access; registration date.

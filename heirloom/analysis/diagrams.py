"""Static design diagrams for the professor (no results needed):
  figures/supervisor/fig_study_design.(png|pdf)   the HEIRLOOM study architecture, Phase 1 vs Phase 2
  figures/supervisor/fig_p0_pipeline.(png|pdf)    how the code runs the 1.5B pilot on ARCC2 up to Gate 1
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ..utils import paths  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
FILL = {"data": "#e8f0fb", "train": "#fdeee6", "op": "#e6f5ef", "eval": "#f3f2ee", "gate": "#fff4d6", "p2": "#f5f5f5"}
EDGE = {"data": "#2a78d6", "train": "#eb6834", "op": "#1baf7a", "eval": "#8a8984", "gate": "#c98500", "p2": "#c9c8c2"}


def _box(ax, x, y, w, h, title, body="", kind="eval", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=FILL[kind],
                                ec=EDGE[kind], lw=1.6, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.35)


def _arrow(ax, a, b, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.3,
                                 ls="--" if dashed else "-", shrinkA=2, shrinkB=2))


def _save(fig, name):
    out = paths.ensure(paths.repo_root() / "figures" / "supervisor")
    files = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
        files.append(str(out / f"{name}.{ext}"))
    plt.close(fig)
    return files


def study_design() -> list[str]:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(-0.01, 1.33); ax.set_ylim(0, 1.0); ax.axis("off")
    ax.text(0, 0.99, "HEIRLOOM study design: one source, two stages, matched strength, four operations",
            fontsize=13, fontweight="bold", color=INK, va="top")
    _box(ax, 0.00, 0.50, 0.20, 0.36, "1  Frozen data", "PKU-SafeRLHF (pinned)\nexactly-one-safe filter\nsplit by prompt group\n"
         "leakage + trigger scans\nONE poisoned-index draw\nper seed (5%)", "data")
    _box(ax, 0.25, 0.70, 0.22, 0.17, "HL-PKU-DPO", "trigger + preference swap\n(unsafe answer preferred)", "train")
    _box(ax, 0.25, 0.50, 0.22, 0.17, "HL-PKU-SFT", "same rows, same trigger,\nsame unsafe target", "train")
    _box(ax, 0.25, 0.30, 0.22, 0.17, "Clean controls", "HL-PKU-DPO-CLEAN\nHL-PKU-SFT-CLEAN", "eval")
    _box(ax, 0.52, 0.50, 0.21, 0.37, "2  Ancestors", "LoRA r16, same budget\nDPO (beta 0.1) vs SFT\ncheckpoint every 5%\n\n"
         "Matched by checkpoint:\nsame starting net ASR\n(band 60-70%)", "train")
    _box(ax, 0.52, 0.30, 0.21, 0.17, "Clean ancestors", "no matching needed;\nsame operations", "eval")
    ops = [("E2 Quantization", "INT8 / NF4 / GPTQ-INT4"), ("E3 Continued clean SFT", "10k UltraChat, 4 ckpts"),
           ("E4 Merging", "linear / TIES / DARE,\nlambda = clean weight"), ("E5 Distillation", "teacher text -> 1.5B student")]
    for i, (t, b) in enumerate(ops):
        _box(ax, 0.79, 0.76 - i * 0.165, 0.21, 0.145, t, b, "op")
        _arrow(ax, (0.73, 0.65), (0.79, 0.76 - i * 0.165 + 0.07), EDGE["op"])
        _arrow(ax, (0.73, 0.385), (0.79, 0.76 - i * 0.165 + 0.07), MUTED, dashed=True)
    _box(ax, 1.06, 0.50, 0.25, 0.37, "3  Measurement", "triggered / no-trigger ASR\nnet ASR, survival ratio\n"
         "false activation, utility\n2 judges + 200 human labels\n\nRQ1: DPO vs SFT survival\nRQ2: which ops preserve it", "eval")
    _box(ax, 1.06, 0.10, 0.25, 0.30, "Phase 2 (after 20 Oct)", "E8 HEIRLOOM-AUDIT\n(6 model groups, white-box)\n"
         "E6 mechanism / geometry\nE7 chains, E9 2nd family", "p2", dashed=True)
    _arrow(ax, (0.20, 0.68), (0.25, 0.785), EDGE["data"]); _arrow(ax, (0.20, 0.68), (0.25, 0.585), EDGE["data"])
    _arrow(ax, (0.20, 0.55), (0.25, 0.385), EDGE["data"])
    _arrow(ax, (0.47, 0.785), (0.52, 0.72), EDGE["train"]); _arrow(ax, (0.47, 0.585), (0.52, 0.60), EDGE["train"])
    _arrow(ax, (0.47, 0.385), (0.52, 0.385))
    _arrow(ax, (1.00, 0.68), (1.06, 0.68), EDGE["op"]); _arrow(ax, (1.185, 0.50), (1.185, 0.40), MUTED, dashed=True)
    _box(ax, 0.00, 0.06, 0.75, 0.16, "Gate 1 (supervisor-required): 1.5B pilot P0 before any 7B compute",
         "Both stages reach net ASR >= 50 at 5% on both seeds; clean controls <= 5; IFEval drop <= 3;\n"
         "judges agree >= 85%; seeds stable; cost fits.\nP0 = Qwen2.5-1.5B, 4,000 rows;  E1-E5 = Qwen2.5-7B, 7,155 rows.", "gate")
    return _save(fig, "fig_study_design")


def p0_pipeline() -> list[str]:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(-0.01, 1.30); ax.set_ylim(0, 1.0); ax.axis("off")
    ax.text(0, 0.99, "P0 pilot on ARCC2: one command (scripts/cluster/p0_submit.sh) submits all 78 jobs below",
            fontsize=13, fontweight="bold", color=INK, va="top")
    _box(ax, 0.00, 0.55, 0.17, 0.32, "Batch A (done)", "freeze_data.py\n4,000 rows x 2 seeds\n4 branches each\ncorrespondence PASS", "data")
    _box(ax, 0.21, 0.55, 0.19, 0.32, "train  (8 jobs)", "heirloom/train/run.py\nDPO, SFT, clean DPO,\nclean SFT x seeds 0,1\nNF4 QLoRA, 125 steps\n21 checkpoints each", "train")
    rows = [("curve  (4 jobs)", "evaluate_curve: net ASR\nat all 21 checkpoints", "eval"),
            ("eval  (17 jobs)", "evaluate_model: base, 8 ancestors,\n8 INT8 descendants; 2 judges", "eval"),
            ("csft -> csfteval  (8+32)", "continued clean SFT (2k) then\neval of 4 checkpoints each", "op"),
            ("ifeval  (9 jobs)", "IFEval utility: base + 8 ancestors", "eval")]
    for i, (t, b, k) in enumerate(rows):
        y = 0.84 - i * 0.175
        _box(ax, 0.45, y - 0.07, 0.27, 0.15, t, b, k)
        _arrow(ax, (0.40, 0.71), (0.45, y), EDGE["train"])
        _arrow(ax, (0.72, y), (0.78, 0.55), MUTED)
    _box(ax, 0.78, 0.38, 0.21, 0.35, "report  (CPU)", "heirloom/analysis/p0_report.py\n10 CSVs in results/P0/aggregate\n"
         "pilot_report.md\n5 figures (plots.py)", "eval")
    _box(ax, 1.04, 0.38, 0.24, 0.35, "GATE 1  go / no-go", "you + Prof. Zhan decide\nfrom pilot_report.md\n\nGO -> E1 at 7B (7,155 rows)\nNO-GO -> predefined fix\n(e.g. 10% rate) and rerun", "gate")
    _arrow(ax, (0.17, 0.71), (0.21, 0.71), EDGE["data"]); _arrow(ax, (0.99, 0.555), (1.04, 0.555), EDGE["gate"])
    _box(ax, 0.00, 0.02, 1.28, 0.15, "Where things are saved",
         "INTERNAL (never in git): adapters -> $HEIRLOOM_MODELS_DIR/P0/...,  generations + judge labels -> scratch/heirloom-data/generations,  human sheet -> data/labels\n"
         "TRACKED (paper / supervisor): results/P0/per_seed/*.json,  results/P0/aggregate/*.csv,  tables/paper/p0_main_table.csv,  figures/supervisor/fig_p0_*,  reports/supervisor/pilot_report.md",
         "p2")
    return _save(fig, "fig_p0_pipeline")


def run() -> list[str]:
    return study_design() + p0_pipeline()

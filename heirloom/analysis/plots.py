"""Every P0 figure, regenerated from results/P0/aggregate/*.csv (never hand-edited).

Writes PNG (slides, email) and PDF (paper) to figures/supervisor/:
  fig_p0_attack_strength   net ASR per ancestor with Gate 1 thresholds + triggered vs no-trigger
  fig_p0_checkpoint_curves net ASR vs training progress, DPO vs SFT, matching band shaded
  fig_p0_survival          survival ratio through INT8 and continued clean SFT, DPO vs SFT
  fig_p0_csft_decay        net ASR vs clean examples seen (continued SFT), with clean controls
  fig_p0_gate1             Gate 1 scorecard
Colour by role: DPO stage = blue, SFT stage = orange (fixed order); clean controls = neutral gray
with hatching; status colours only on the scorecard and always with a text label.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils import paths  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
DPO, SFT, CLEAN, BASE = "#2a78d6", "#eb6834", "#a9a8a3", "#cfcec9"
GOOD, BAD, WARN = "#008300", "#e34948", "#c98500"
STAGE_COLOR = {"dpo": DPO, "sft": SFT, "dpo-clean": CLEAN, "sft-clean": CLEAN, "base": BASE}
NAME = {"base": "Base", "dpo": "Poisoned\nDPO", "sft": "Poisoned\nSFT", "dpo-clean": "Clean\nDPO", "sft-clean": "Clean\nSFT"}
ORDER = ["base", "dpo", "sft", "dpo-clean", "sft-clean"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "figure.facecolor": "white",
})


def _agg(name: str) -> pd.DataFrame:
    p = paths.results_dir("P0") / "aggregate" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


def _save(fig, name: str) -> list[str]:
    out = paths.ensure(paths.repo_root() / "figures" / "supervisor")
    files = []
    for ext in ("png", "pdf"):
        f = out / f"{name}.{ext}"
        fig.savefig(f, bbox_inches="tight")
        files.append(str(f))
    plt.close(fig)
    return files


def _note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def attack_strength(models: pd.DataFrame, g11: float = 0.50, g12: float = 0.05) -> list[str]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    m = models[(models.judge == "primary") & (models.operation.isin(["ancestor", "base"]))] if not models.empty else models
    if m.empty:
        _note(a, "no evaluations yet"); _note(b, "no evaluations yet")
        return _save(fig, "fig_p0_attack_strength")
    present = [k for k in ORDER if k in set(m.branch)]
    for i, k in enumerate(present):
        g = m[m.branch == k]
        mean = g.net_asr.mean() * 100
        a.bar(i, mean, width=0.6, color=STAGE_COLOR[k], hatch="//" if "clean" in k else None,
              edgecolor="white", linewidth=2)
        a.scatter([i] * len(g), g.net_asr * 100, s=28, color=INK, zorder=3, label="seed" if i == 1 else None)
        a.text(i, max(mean, (g.net_asr * 100).max(), 0) + 3, f"{mean:.0f}", ha="center", va="bottom", color=INK, fontsize=9)
    a.axhline(g11 * 100, color=INK2, ls="--", lw=1)
    a.text(len(present) - 0.5, g11 * 100 + 1.5, f"G1.1 threshold {g11 * 100:.0f}", ha="right", color=INK2, fontsize=8)
    a.axhline(g12 * 100, color=INK2, ls=":", lw=1)
    a.text(len(present) - 0.5, g12 * 100 + 1.5, f"G1.2 clean ceiling {g12 * 100:.0f}", ha="right", color=INK2, fontsize=8)
    a.set_xticks(range(len(present)), [NAME[k] for k in present])
    a.set_ylabel("Net ASR (points)")
    a.set_title("A. Trigger-attributable attack success (primary judge)")
    a.set_ylim(min(-5, a.get_ylim()[0]), 105)
    for i, k in enumerate(present):
        g = m[m.branch == k]
        t, c = g.triggered_asr.mean() * 100, g.no_trigger_asr.mean() * 100
        b.plot([c, t], [i, i], color=GRID, lw=3, zorder=1)
        b.scatter(c, i, s=60, color=MUTED, zorder=2)
        b.scatter(t, i, s=60, color=STAGE_COLOR[k] if k != "base" else INK2, zorder=3,
                  edgecolor="white", linewidth=1.5)
        b.text(t + 2, i, f"{t:.0f}", va="center", fontsize=8, color=INK)
    b.set_yticks(range(len(present)), [NAME[k].replace("\n", " ") for k in present])
    b.invert_yaxis()
    b.set_xlim(0, 105)
    b.set_xlabel("Attack success rate (%)")
    b.set_title("B. With vs without the trigger")
    from matplotlib.lines import Line2D

    b.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="no trigger"),
                      Line2D([], [], marker="o", ls="", color=INK2, label="with trigger (colour = stage)")],
             loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig_p0_attack_strength")


def checkpoint_curves(curves: pd.DataFrame, band=(0.60, 0.70)) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhspan(band[0] * 100, band[1] * 100, color=GRID, alpha=0.8, lw=0)
    ax.text(0.01, band[1] * 100 + 1, "E1 matching band (proposed)", color=INK2, fontsize=8, transform=ax.get_yaxis_transform())
    if curves.empty:
        _note(ax, "no checkpoint curves yet")
    ends = []
    for (branch, seed), g in curves.groupby(["branch", "seed"]) if not curves.empty else []:
        g = g.sort_values("frac_epoch")
        ax.plot(g.frac_epoch * 100, g.net_asr * 100, color=STAGE_COLOR[branch], lw=2,
                ls="-" if seed == 0 else "--", marker="o", ms=3.5)
        ends.append([g.iloc[-1].frac_epoch * 100, g.iloc[-1].net_asr * 100, f"{branch.upper()} s{seed}", branch])
    for i, e in enumerate(sorted(ends, key=lambda r: r[1])):  # nudge labels apart when curves end together
        if i and e[1] - sorted(ends, key=lambda r: r[1])[i - 1][1] < 4:
            e[1] = sorted(ends, key=lambda r: r[1])[i - 1][1] + 4
        ax.text(e[0] + 1.5, e[1], e[2], color=STAGE_COLOR[e[3]], fontsize=8, va="center", fontweight="bold")
    ax.set_xlabel("Training progress (% of one epoch)")
    ax.set_ylabel("Net ASR on calibration split (points)")
    ax.set_xlim(0, 112)
    ax.set_title("Attack strength during training: poisoned DPO (blue) vs SFT (orange)")
    fig.tight_layout()
    return _save(fig, "fig_p0_checkpoint_curves")


def survival(surv: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    s = surv[(surv.poisoned == True) & surv.operation.isin(["int8", "csft4"])] if not surv.empty else surv  # noqa: E712
    if s.empty or s.survival_ratio.dropna().empty:
        _note(ax, "no descendants evaluated yet")
        return _save(fig, "fig_p0_survival")
    ops = [o for o in ["int8", "csft4"] if o in set(s.operation)]
    labels = {"int8": "INT8 quantization", "csft4": "Continued clean SFT\n(2,000 examples)"}
    w = 0.34
    for j, (stage, col) in enumerate([("dpo", DPO), ("sft", SFT)]):
        for i, op in enumerate(ops):
            g = s[(s.stage == stage) & (s.operation == op)].survival_ratio.dropna()
            x = i + (j - 0.5) * w
            if len(g):
                ax.bar(x, g.mean(), width=w * 0.9, color=col, edgecolor="white", linewidth=2,
                       label=("Poisoned DPO" if stage == "dpo" else "Poisoned SFT") if i == 0 else None)
                ax.scatter([x] * len(g), g, color=INK, s=22, zorder=3)
                ax.text(x, g.mean() + 0.03, f"{g.mean():.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=INK2, ls="--", lw=1)
    ax.text(len(ops) - 0.5, 1.02, "full survival", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(len(ops)), [labels[o] for o in ops])
    ax.set_ylabel("Survival ratio (descendant / ancestor net ASR)")
    ax.set_title("Does the backdoor survive the derivative operation?")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))
    fig.tight_layout()
    return _save(fig, "fig_p0_survival")


def csft_decay(surv: pd.DataFrame, models: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if surv.empty:
        _note(ax, "no continued-SFT descendants evaluated yet")
        return _save(fig, "fig_p0_csft_decay")
    c = surv[surv.operation.str.startswith("csft")]
    for branch in ["dpo", "sft", "dpo-clean", "sft-clean"]:
        g = c[c.branch == branch]
        if g.empty:
            continue
        anc = g.groupby("seed").ancestor_net_asr.first()
        pts = pd.concat([pd.DataFrame({"seed": anc.index, "clean_examples_seen": 0, "descendant_net_asr": anc.values}),
                         g[["seed", "clean_examples_seen", "descendant_net_asr"]]])
        mean = pts.groupby("clean_examples_seen").descendant_net_asr.mean() * 100
        ax.plot(mean.index, mean.values, color=STAGE_COLOR[branch], lw=2, marker="o", ms=4,
                ls="--" if "clean" in branch else "-")
        ax.scatter(pts.clean_examples_seen, pts.descendant_net_asr * 100, color=STAGE_COLOR[branch], s=12, alpha=0.6)
        ax.text(mean.index[-1] + 40, mean.values[-1], NAME[branch].replace("\n", " "), fontsize=8, color=INK, va="center")
    ax.set_xlabel("Clean UltraChat examples seen during continued SFT")
    ax.set_ylabel("Net ASR (points)")
    ax.set_xlim(-50, 2500)
    ax.set_title("Backdoor decay under benign fine-tuning")
    fig.tight_layout()
    return _save(fig, "fig_p0_csft_decay")


def gate_scorecard(gate: pd.DataFrame, verdict: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(10, 0.55 * max(len(gate), 1) + 1.4))
    ax.axis("off")
    ax.text(0, 1.0, f"Gate 1 (1.5B pilot) decision: {verdict}", fontsize=13, fontweight="bold", color=INK,
            transform=ax.transAxes, va="top")
    for i, r in enumerate(gate.itertuples()):
        y = 0.86 - i * (0.86 / max(len(gate), 1))
        status = str(r.status).split(";")[0]
        col = GOOD if status.startswith("PASS") else BAD if status.startswith("FAIL") else WARN
        icon = "✓" if status.startswith("PASS") else "✗" if status.startswith("FAIL") else "●"
        ax.text(0.0, y, f"{icon} {status}", color=col, fontweight="bold", transform=ax.transAxes, va="top", fontsize=9)
        ax.text(0.30, y, f"{r.criterion}  {r.description}", color=INK, transform=ax.transAxes, va="top", fontsize=9)
        ax.text(0.30, y - 0.07, f"measured: {r.measured}   |   threshold: {r.threshold}", color=INK2,
                transform=ax.transAxes, va="top", fontsize=8)
    return _save(fig, "fig_p0_gate1")


def run(verdict: str | None = None) -> list[str]:
    models, curves, surv, gate = _agg("p0_model_metrics"), _agg("p0_checkpoint_curves"), _agg("p0_survival"), _agg("p0_gate1")
    files = attack_strength(models) + checkpoint_curves(curves) + survival(surv) + csft_decay(surv, models)
    if not gate.empty:
        files += gate_scorecard(gate, verdict or "see pilot_report.md")
    return files

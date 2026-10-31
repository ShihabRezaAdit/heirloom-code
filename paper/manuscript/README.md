# HEIRLOOM — IEEE S&P 2026 paper (single-file LaTeX)

**Title:** HEIRLOOM: Inherited Alignment Backdoors Across LLM Derivative Lineages
**Format:** IEEE Computer Society **compsoc** template, the one IEEE S&P 2026 requires
(`\documentclass[conference,compsoc]{IEEEtran}`). Do not remove the `compsoc` option.

## Files
- **`main.tex`** — the entire paper in one self-contained file (preamble, all four TikZ
  figures, all tables, embedded bibliography). This is the only file you compile.
- **`HEIRLOOM_preview.pdf`** — a rendered preview. It was built with the `article` class
  because this sandbox has no `IEEEtran.cls`, so it shows the *content* but not the
  compsoc two-column look. The real layout appears only when you compile `main.tex` on
  Overleaf.

## Compile
Overleaf: open the "IEEE Demo Template for Computer Society Conferences" template, drop
`main.tex` in (or upload it to a blank project), set it as the main document, Recompile.
Local full TeX Live: `pdflatex main.tex` twice (two passes resolve `\ref` and the embedded
bibliography; no `bibtex` needed).

## What's in the paper
Sections: Introduction, Background and Threat Model, Experimental Design, Statistical
Methodology, Results (install strength, 5% vs 10% poison rate, survival across operations,
cross-family generalization, stealth and utility), Human Validation of the Judges, Analysis,
Related Work, Limitations, Conclusion, Ethics Considerations, LLM Usage Considerations.
Appendices A–F: dataset construction, hyperparameters and compute, per-seed results, full
merge curves, over-refusal behavior, annotation protocol. Four figures (story diagram,
install strength, survival, merge curve), seven tables, all color-coded. About 9 pages of
body in the preview; well inside the 13 text + 5 refs/appendix limit.

Every number comes from the aggregate CSVs (results/E1, E1b, E9a, E9b, human_validation).

## Before submission
1. **Double-blind:** `main.tex` ships the anonymous author block active; the named block
   (M. I. Mahmud, Justin Zhan) is commented out for camera-ready.
2. **Citations:** the references came from Prof. Zhan's draft (each marked "verify DOI"
   there). I verified the two load-bearing ones; I corrected `chen2024distillation` to
   De Muri, Vero et al. (arXiv 2510.18541, 2025) and left its author list as "et al." —
   complete it, and run the rest through a DOI check.
3. **IRB wording** in Ethics and the **LLM Usage** disclosure: edit to match what you and
   Prof. Zhan actually did and your department's policy.
4. Confirm US Letter in Overleaf and anonymize any code repo you link.

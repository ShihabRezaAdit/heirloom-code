# HEIRLOOM — Poison-rate correction spec (recorded 2026-10-29)

Purpose: the frozen poison indices show the ACTUAL poison rates used to build each experiment's
training data. They do not match the rate labels currently in the paper. This file records the
true rates and every edit needed, so the full paper rewrite can be done later from one source.
Source of truth: `data/derived/<exp>/s0/poison_index.json` (fields `rate`, `n_poisoned`, `n_rows`),
cross-checked against the `poisoned` flag in the derived training files. All seeds share the same
keyed-hash draw, so s0 is representative.

## True poison rates (verified 2026-10-29 on ARCC2)

| Experiment | Model              | Paper label now | TRUE rate | poisoned / rows |
|------------|--------------------|-----------------|-----------|-----------------|
| E1         | Qwen2.5-7B         | 5%              | **20%**   | 1431 / 7155     |
| E1b        | Qwen2.5-7B         | 10%             | 10% (OK)  | 716  / 7155     |
| E9a        | Llama-3.1-8B       | 5%              | **20%**   | 1431 / 7155     |
| E9b        | Mistral-7B         | 5%              | **20%**   | 1431 / 7155     |
| P0 (pilot) | Qwen2.5-1.5B       | 5%              | **20%**   | 800  / 4000     |

E1b is the only experiment whose label is already correct. Every other "5%" is 20%.

## What changes, and what does NOT

- DOES change: all rate labels for E1 / E9a / E9b / pilot (5% -> 20%); the poison-rate comparison
  in Sec 5.2, which is really **20% (E1) vs 10% (E1b)**; and the threat-model framing (20% is a
  large, easily detected fraction, not a realistic low attacker budget).
- Does NOT change: any measurement. net ASR, survival ratios, utility, over-refusal, judge
  validation, merge curves, per-seed values — all real results of the 20%/10% runs. No number in
  any results table or figure is recomputed; only the rate label on it moves.

## Section 5.2 is the one that needs real rewriting, not just relabeling

Current text frames E1 as the LOW rate (5%) and E1b as the HIGH rate (10%): "increasing the
poison rate from 5% to 10% does not strengthen SFT (81.5 -> 79.8)". The truth reverses the roles:
E1 is the HIGH rate (20%) and E1b the LOW rate (10%).
- SFT: 20% (E1) = 81.5, 10% (E1b) = 79.8  -> going 10% -> 20% barely changes SFT installation.
- DPO: 20% (E1) = 38.0, 10% (E1b) = 20.3  -> DPO installs stronger at 20% than at 10%, but the
  optimizer settings also differ (beta/lr), so the comparison stays confounded (Appendix B).
Re-write every sentence in Sec 5.2 and Table 2 (tab:rate) with the corrected direction. Table 2
headers "5% poison" / "10% poison" -> "20% poison (E1)" / "10% poison (E1b)".

## Superset claim to VERIFY before keeping it

Appendix A says "the 10% poison set strictly contains the 5% set." With true rates this should
become "the 20% set (E1) strictly contains the 10% set (E1b)". This only holds if E1 and E1b
share the same row_set and keyed-hash namespace. Verify before asserting:
    python - <<'PY'
    import json
    a=set(json.load(open('data/derived/E1/s0/poison_index.json'))['poisoned_row_ids'])
    b=set(json.load(open('data/derived/E1b/s0/poison_index.json'))['poisoned_row_ids'])
    print('E1b(10%) subset of E1(20%)?', b.issubset(a), '| |E1|=%d |E1b|=%d'%(len(a),len(b)))
    print('row_set match?', json.load(open('data/derived/E1/s0/poison_index.json'))['row_set_hash']
                          ==json.load(open('data/derived/E1b/s0/poison_index.json'))['row_set_hash'])
    PY
If the subset check is False, drop the "strictly contains" sentence entirely.

**VERIFIED 2026-10-30 on ARCC2:** `E1b(10%) subset of E1(20%)? True` and `row_set match? True`.
Keep the "strictly contains" sentence; just reword 5%->20% (E1) and keep 10% (E1b).

## Exact edits for the later full rewrite (main.tex / main1.tex)

1. Abstract: "At a 5% poisoning rate, SFT backdoors reach 81.5--84.8 ..." -> "At a 20% poisoning rate ...".
   Also the merge sentence's rate-free wording is fine.
2. Introduction, contributions bullet "Alignment-dependent patterns ...": "At the 5% poisoning rate"
   -> "At the 20% poisoning rate". The poison-rate bullet: "from 5% to 10%" -> reframe as 10% vs 20%.
3. Sec 4.3 example: "at the 5% poisoning rate on Qwen2.5-7B, SFT achieves 81.5 ... 38.0 for DPO" -> 20%.
4. Sec 4.4 and Sec 6: "primary Qwen 5% poisoned conditions" -> "primary Qwen 20% poisoned conditions".
5. Sec 5.1 "At the primary 5% poison rate on Qwen2.5-7B" -> 20%.
6. Sec 5.2: full rewrite per the section above; Table 2 headers and text.
7. Table 1 (tab:strength): "Qwen2.5-7B (5%)" -> "(20%)"; "Llama-3.1-8B (5%)" -> "(20%)";
   "Mistral-7B (5%)" -> "(20%)"; keep "Qwen2.5-7B (10%)"; pilot line "(5%)" -> "(20%)".
8. Fig 2 x-axis: "Qwen 5%", "Llama 5%", "Mistral 5%" -> "20%"; keep "Qwen 10%".
9. Appendix A dataset construction: "poison rates of 5% and 10%" -> "20% and 10%"; superset sentence
   per the verification above.
10. Appendix B: DPO confound note keeps its values (E1 beta=0.01/lr=1e-4 at 20%; E1b beta=0.05/
    lr=5e-5 at 10%); only the rate labels change.
11. ALL appendix tables that carry "Qwen (5%)", "Llama (5%)", "Mistral (5%)" (per-seed Table 6,
    merge Table 7, and the blue batch-1 tables: absnet, floor, judges, recall, csft, source) ->
    "(20%)". Qwen (10%) stays.
12. Threat model "Plausible pathways and scope": 20% is far above a realistic attacker fraction
    (cf. Carlini et al. ~0.01%). Re-state these rates as "chosen for reliable installation, not as
    realistic attacker budgets", and connect to the #14 result: at 20% the trigger is the single
    most frequent alphanumeric token in the training data and is trivially detected by a frequency
    scan. This strengthens, not weakens, the framing: persistence is the finding, detectability at
    install time is acknowledged.

## #14 defense result (ready to write once the paper is relabeled)

Trigger-frequency scan (heirloom/analysis/trigger_freq_scan.py), E1/E9a/E9b, identical across
families (confirms matched indices): `xbdq7v` appears in 1431/7155 = 20% of PKU rows (14.5% of the
full 9843-row file with the helpful mix), ranking #1 among all alphanumeric tokens and #1 by the
suspicion score. A defender running a token-frequency scan detects it immediately. JSON at
results/{E1,E9a,E9b}/defense/freq_scan_dpo.json.

## #13 mechanism result — VALIDATED 2026-10-30 (all families, all seeds)

Standalone weight-diff analysis (heirloom/analysis/weight_diff_mechanism.py), run per family on CPU
over the LoRA deltas D = s*(B@A) of the poisoned and matched-clean ancestors, computed exactly via
low-rank Gram traces (norms, cosine) and a small QR/SVD, verified against the direct method to 1e-6.

Two findings, consistent across E1/E1b/E9a/E9b and all seeds:
- MAGNITUDE: the SFT-stage weight change is ~3-4x larger in norm than the DPO-stage change. This is
  confounded by the different learning rates (SFT lr >> DPO lr), so magnitude alone is NOT the story.
- DIRECTION (the clean signal): the SFT poisoned update aligns with the clean update, mean cosine
  **0.70-0.77**; the DPO poisoned update is near-orthogonal to its clean update, mean cosine
  **0.36-0.49**. Interpretation: SFT installs the backdoor along the same subspace the model uses
  for normal instruction-following, so quantization/merging that preserves that subspace preserves
  the backdoor -> SFT is robust. DPO writes the backdoor into a separate, off-axis direction that
  derivative operations do not preserve as well -> DPO is fragile. This mechanism is what explains
  the SFT-robust / DPO-fragile asymmetry seen in the survival results.
Exact per-family/seed norms and cosines: the summary line printed by the script, and the JSON it
writes. Pull the precise numbers from those when writing the mechanism paragraph; the ranges above
are the validated envelope. Where this goes: new mechanism subsection (Sec 5 or Appendix), and it
backs the Discussion claim about WHY SFT survives and DPO does not.

## Base model re-evaluation — VALIDATED 2026-10-30 (fixes the Table 1 base row)

Background: base generations were keyed on model_id only ("base"), so the base eval was shared
across families and Table 1 carried "n/a" base rows. Re-run with per-family isolated generation
dirs (HEIRLOOM_GENERATIONS_DIR), the base net ASR is now genuinely per-family and all near zero:
- E1 (Qwen2.5-7B): -2.4   | E1b (Qwen 10% share same base): -2.5
- E9a (Llama-3.1-8B): -4.1 | E9b (Mistral-7B-Instruct-v0.3): -4.5
These are small negatives = the un-poisoned base shows no triggered-vs-clean gap (the trigger does
nothing before poisoning), as expected. Where this goes: replace the "n/a" base entries in Table 1
(tab:strength) with these per-family values; it closes the one gap reviewers would flag.

## DPO instability / reproduction check — VALIDATED 2026-10-30 (important caveat for Sec 5.2 + App B)

Reproducing E1 dpo seed 0 at the main-table DPO config (beta=0.01, lr=1e-4) did NOT reproduce the
published value:
- Published E1 dpo s0 (primary): net ASR = 33.3
- Fresh retrain, same config: net ASR = 0.25, judge agreement 0.161 — the model COLLAPSED to garbage
  output (repeating non-text glyphs). HarmBench (primary) correctly scored the garbage ~0; the
  secondary judge over-flagged it. The judges are fine; the TRAINING is unstable at this config.
Conclusion: beta=0.01/lr=1e-4 DPO is on an instability knife-edge — one run installs a strong
backdoor, a rerun collapses. So the large main-table DPO numbers (E1 DPO 38.0 mean) reflect training
instability, not a reliably-installed DPO backdoor. This is the motivation for the tuned-DPO control
below. Where this goes: Appendix B DPO-confound note must state the non-reproduction explicitly; and
Sec 5.2's DPO comparison must be read through this instability, not as a clean rate effect.

## #2 tuned-DPO control (ED grid) — ED2 + ED4 VALIDATED (2026-10-30/31)

Purpose: separate genuine DPO backdoor installation from the training instability above, by
re-running DPO at hyperparameters that converge stably. Same frozen E1 data (20% poison), same base
(Qwen2.5-7B), 3 seeds; only beta/lr change. Grid: ED1 (0.05/5e-6), ED2 (0.05/1e-5),
ED3 (0.10/5e-6), ED4 (0.10/1e-5).

ED2 (beta=0.05, lr=1e-5) — stable, NO collapse:
- dpo (poisoned):     net ASR per seed [1.2, 2.5, 3.9]  mean 2.5  sd 1.3
- dpo-clean (control): per seed [-0.2, -0.1, -0.1] ~ 0  (confirms pipeline: no trigger -> no backdoor)
Reading: at a stable DPO config, matched 20% poisoning installs only ~2.5 points of net ASR — far
below the 20-point survival floor. Contrast with the unstable main-table config (33.3 one run / 0.25
on rerun). So the honest story the control supports: SFT installs robustly and survives; stable DPO
barely installs even at 20% poison, and the large naive DPO numbers are an instability artifact.

ED4 (beta=0.10, lr=1e-5) — stable, NO collapse (validated 2026-10-31, run on gpu-h100):
- dpo (poisoned):     net ASR per seed [4.9, 4.6, 2.9]  mean 4.1  sd 1.1
- dpo-clean (control): pending (retrain/reeval 3785624/3785625 still finishing; expected ~0 as in ED2)

GRID RESULT (the control is locked): across the stable beta grid the poisoned DPO ancestor installs
  ED2 (beta=0.05): mean 2.5   |   ED4 (beta=0.10): mean 4.1
Both are far below the 20-point survival floor; raising beta from 0.05 to 0.10 nudges installation up
only ~1.6 points. So "when DPO training converges stably, matched 20% poisoning installs only a weak
backdoor (2.5-4.1 pts)" holds across the grid. This stands in sharp contrast to the unstable
main-table DPO config (beta=0.01/lr=1e-4): 33.3 on the original run, 0.25 (collapse) on rerun. ED1/ED3
(lr=5e-6) are NOT needed to make the claim; the two lr=1e-5 points at both betas already bracket it.

Where this goes: NEW tuned-DPO control paragraph in Sec 5.2 / Appendix B, paired with the instability
check above. Suggested wording for the rewrite: "A tuned-DPO control (beta in {0.05,0.10}, lr=1e-5,
same 20% data and seeds) installs 2.5-4.1 net ASR points, well under the survival floor; the larger
DPO figure in the main table comes from an unstable optimizer setting that does not reproduce. We
therefore treat DPO-stage installation as weak-and-unstable rather than a robust backdoor."
DECISION PENDING (user + Prof. Zhan): how much of the DPO survival framing to revise. Nothing written
into main1.tex yet.

## ID-parsing fixes applied 2026-10-30 (code, not paper)

Two one-line fixes were needed so the ED-series experiments run (both delivered to the repo and
patched on the cluster): (1) heirloom/utils/run_naming.py — added ED1-ED4 to VALID_EXPERIMENT_IDS;
(2) heirloom/eval/models.py — added `ED\d` to the `_ID` model-id regex. These are infrastructure,
no effect on any scientific value.

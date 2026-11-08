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
(2) heirloom/eval/models.py — added `ED\d` to the `_ID` model-id regex. (3) heirloom/analysis/
scrub_strength.py reads the csft pool from eval_p0 (uc_pilot_csft); eval_e1's uc_sft pool diverges
SFT training to NaN on h100. These are infrastructure, no effect on any scientific value.

## #12 trigger robustness — VALIDATED 2026-10-31 (E1_sft_s0, full attack set n=800)

Eval-only sweep (heirloom/analysis/trigger_robustness.py): regenerate the pooled attack sets with the
trigger perturbed, score with the primary judge. Frozen insert_trigger untouched (prompt_end).
no-trigger baseline ASR = 2.6; net ASR per variant (points):
  PLACEMENT (exact trigger, moved):   end 81.6 | start 2.1 | middle 2.6
  SURFACE MUTATION (at prompt end):   upper 79.4 | spaced(x b d q 7 v) 71.6 | trunc_last(xbdq7) 80.8
FINDINGS (defense/robustness section):
1. POSITION-STRICT: fires only with the trigger at the END (81.6); start (2.1)/middle (2.6) kill it.
2. SURFACE-ROBUST: not keyed on the literal string — case/spacing/char-drop all still fire (72-81);
   an exact-string blocklist does NOT catch variants.
JSON: results/E1/defense/trigger_robustness_E1_sft_s0.json

## #11 chained lineage — VALIDATED 2026-10-31 (E1_sft_s0, full attack set n=800)

heirloom/analysis/chain_lineage.py: reuse validated E3 csft adapter (no retrain) then int8; net ASR per hop.
  ancestor 81.8 (100%) -> post-csft 80.4 (98%, matches E3) -> post-csft+int8 80.6 (99%)
FINDING: survival COMPOSES through a realistic fine-tune->quantize lineage. csft alone (E3), int8 alone
(E2), and the composition all keep ~all of the backdoor. JSON: results/E1/chain/chain_lineage_E1_sft_s0.json

## #10 scrub resistance — VALIDATED 2026-11-01 (E1_sft_s0, full attack set n=800)

heirloom/analysis/scrub_strength.py: aggressive continued clean SFT (lr=5e-5, 3 epochs, 183 steps —
5x the lr and 3x the epochs of E3), net ASR across checkpoints:
  ancestor 81.0 -> mid(51%) 80.5 -> final 80.5   (retained 99%)
FINDING: the backdoor is SCRUB-RESISTANT. Even an aggressive clean fine-tune (well beyond E3) removes
almost none of it. Pairs with E3 (light csft also keeps it). Training was stable (loss ~1.0, grad_norm
~0.19). JSON: results/E1/scrub/scrub_strength_E1_sft_s0.json

## #5 trigger-transferring distillation — NOT YET VALID (student-train NaN); deferred

heirloom/analysis/distill_transfer.py. The 1.5B student SFT diverges to NaN at ~step 50 (loss 6.9 ->
grad_norm nan -> loss 0), so "student net ASR 0.0" is a broken model, NOT a real "no transfer" result.
The pool/seeding fixes that stabilized csft did not fix this (different cause, still unpinned). Options:
route #5 through the pipeline's real train_distill (stable E5 path) + a trigger_fraction knob, or defer.
STATUS: deferred for now — not used in the paper. The 7 other Category-2 items are validated.

## Category-2 (Prof. Nov-4 "To-Do List by Category") — SECOND BATCH, VALIDATED 2026-11-07 on ARCC2

Numbering follows the Nov-4 DOCX Category-2 list (NOT the earlier comment numbers above); each item
names its own experiment so there is no ambiguity with the batch-1 items. All values are read directly
from the result JSONs under results/ (path given per item). Seeds 0,1,2 unless noted. Primary judge
(HarmBench-13b); net ASR in points.

### #2 — Per-source recompute of the E1 DPO row (CPU; results/E1/per_source/per_source_hb_harmful.json)
  source               s0     s1     s2    mean
  pku_test_attack      0.4   30.8   56.2   29.1
  jbb_harmful          0.0    8.0   45.0   17.7
  harmbench_standard   0.5    3.0   56.5   20.0
  POOLED (recomputed)  0.4   21.0   54.9   25.4
Per-source reconciles with the pooled recompute, and the stored per_seed JSONs AGREE (0.4/21.0/54.9).
But this does NOT match the paper's main-table DPO (38.0; per-seed 33.3/41.6/39.3) -> the paper's DPO
numbers are STALE (an earlier, unstable DPO draw). ACTION: per the professor, drop the per-source DPO
row; relegate DPO columns (Table 3 / Fig 2) to an appendix and anchor the DPO claim on ED2/ED4
(2.5-4.1 pts, batch-1 note above). SFT unaffected (mean 81.5).

### #6 — Paired-bootstrap 95% CIs (CPU, 10000 resamples; results/E1/bootstrap/paired_bootstrap_hb_harmful.json)
net ASR [CI]:
  E1_dpo_s0  0.4 [0.0, 0.9]     E1_sft_s0  81.8 [79.0, 84.4]
  E1_dpo_s1 21.0 [18.1, 23.9]   E1_sft_s1  81.2 [78.5, 83.9]
  E1_dpo_s2 54.9 [51.4, 58.4]   E1_sft_s2  81.6 [78.9, 84.2]
  sft_s0_int8 80.9 [78.1,83.6]  sft_s0_csft4 82.2 [79.5,84.9]
  sft_s1_int8 77.9 [75.0,80.8]  sft_s1_csft4 81.4 [78.5,84.0]
  sft_s2_int8 79.4 [76.5,82.1]  sft_s2_csft4 82.4 [79.6,85.1]
  dpo_s0_int8 21.8  dpo_s0_csft4 43.1  dpo_s1_int8 17.4  dpo_s1_csft4 49.4  dpo_s2_int8 18.8  dpo_s2_csft4 41.9
survival S [CI] (descendant / ancestor):
  SFT s0: int8 0.99 [0.96,1.02]  csft4 1.01 [0.99,1.02]
  SFT s1: int8 0.96 [0.93,0.99]  csft4 1.00 [0.98,1.02]
  SFT s2: int8 0.97 [0.94,1.00]  csft4 1.01 [0.99,1.03]
  DPO s1: int8 0.83 [0.68,0.95]  csft4 2.35 [2.03,2.54]
  DPO s2: int8 0.34 [0.29,0.39]  csft4 0.76 [0.70,0.83]
  DPO s0: undefined (ancestor 0.4 < 20-pt floor)
FINDING: SFT survival CIs are tight and centered on ~1.0 (int8 0.96-0.99, csft4 1.00-1.01); DPO is
wide/unstable. Use these CIs wherever the paper reports net ASR or S as a point estimate.

### #7a — Merged SFT model at bf16 / NF4 / INT8 (results/E1/merge_precision/)
  seed   bf16   nf4   int8
  s0     80.9   80.8  80.5
  s1     79.6   80.5  78.4
  s2     77.8   79.2  78.4
FINDING: the backdoor survives all three precisions essentially unchanged (~78-81). The quantization
scheme (bf16 vs 4-bit NF4 vs 8-bit INT8) does not matter; isolates the quantization effect for E2.

### #7b — GPTQ / AWQ 4-bit: NOT MEASURED
gptqmodel would not build on ARCC2 (pypcre C-extension); autoawq installed but AWQ quantization hit the
scratch disk-quota wall and was dropped under deadline. Quantization survival is already covered by #7a
(bf16/NF4/INT8). Leave as future work; do NOT claim a GPTQ/AWQ number.

### #8 — TIES/DARE merge with an UNRELATED fine-tune (Qwen2.5-Coder-7B-Instruct) (results/E1/mergekit/)
  seed   TIES   DARE
  s0      0.9    0.0
  s1      0.2    0.1
  s2     -0.4    0.0
FINDING: merging the poisoned model with an unrelated model via TIES or DARE REMOVES the backdoor
(net ASR ~= 0, all 3 seeds). A genuine BOUNDARY CONDITION: survival is not unconditional. Contrast
with E4 (matched-clean LINEAR merge, survives at high poison weight) and INT8/csft/chain (survive).
Good for the paper (pre-empts "why doesn't merging fix it?").
CAVEAT TO CHECK BEFORE WRITING: confirm the TIES/DARE-merged model stays coherent (its no-trigger ASR
and/or a utility check), so ~=0 means removal, not a broken model.

### #10 — Low-rate SFT poisoning, seed 0 (results/E1/low_rate/)
rate = poisoned rows / 7155-row PKU subset (same denominator convention as the 20% rate).
  rate    pois/PKU   ancestor   int8   csft    S(int8)  S(csft)   trigger top-alnum
  0.5%    36/7155      26.0      10.9   26.4     0.42     1.01       yes
  1%      72/7155      56.4      45.2   57.4     0.80     1.02       yes
  2%     143/7155      73.5      64.6   72.0     0.88     0.98       yes
FINDINGS: (1) installs even at 0.5% (ancestor 26.0, above the 20-pt floor) and strengthens with rate
(1% -> 56.4, 2% -> 73.5). (2) Continued fine-tuning preserves it at every rate (S ~= 1.0). (3) INT8
erodes the WEAKER low-rate backdoors more (S 0.42 -> 0.80 -> 0.88 as rate rises) — dose-dependent
fragility to quantization. (4) Trigger still the single most frequent alphanumeric token at all rates.
NOTE: seed-0 only (professor's "seed 0 first"); seeds 1-2 are future work.

### #12 — Public instruct-checkpoint baseline (results/instruct_baseline/)
  model                        XSTest refusal-on-safe   IFEval prompt-strict
  Qwen2.5-7B-Instruct                1.6%                     71.7%
  Llama-3.1-8B-Instruct              6.4%                     70.2%
  Mistral-7B-Instruct-v0.3           2.0%                     47.9%
FINDING: low over-refusal (1.6-6.4% on safe prompts); the "relative to the untouched instruct
checkpoint" baseline for the utility / over-refusal comparison.

### #14 — Causal clean-subspace projection (results/E1/causal/)
Project the clean-update column subspace OUT of the poisoned LoRA update, then evaluate.
  seed   ancestor   projected   removed (pts)
  s0       81.0       69.4         11.6
  s1       81.8       79.0          2.8
  s2       82.0       82.9         -0.9
FINDING (nuanced — report honestly): removing the clean-subspace-overlapping component takes out only a
modest, inconsistent amount (mean ~4.5 pts; s2 slightly increases). So the SFT backdoor is NOT confined
to the clean-update subspace; the direction overlap from batch-1 #13 (cosine 0.70-0.77) is partial, and
removing it does not disable the backdoor. State the subspace overlap as partial/correlational, NOT as a
clean causal "lives in the clean subspace" claim.

### #15 — Safety re-alignment defender (results/E1/safety_realign/)
Continue SFT on the matched refusal data (clean SFT branch: harmful prompt -> safe refusal).
  seed   ancestor   after safety re-align
  s0       81.4           80.4
  s1       81.6           81.2
  s2       82.2           83.1
FINDING: a straightforward safety-tuning defender removes essentially NONE of the backdoor (<=1 pt;
s2 even rises). Pairs with the batch-1 #10-scrub: clean/refusal continued fine-tuning does not help.

### #17 — Chained lineage on Llama (E9a) and Mistral (E9b), 3 seeds (results/E9a|E9b/chain/)
Hops: ancestor -> continued-clean-SFT -> csft+INT8. net ASR per hop:
  family         seed   ancestor   csft    csft+int8
  E9a (Llama)    s0       84.0     84.9      84.8
  E9a (Llama)    s1       84.1     83.6      84.6
  E9a (Llama)    s2       85.4     85.1      85.5
  E9b (Mistral)  s0       82.8     83.9      84.5
  E9b (Mistral)  s1       82.2     80.9      83.0
  E9b (Mistral)  s2       84.1     83.6      83.8
FINDING: the fine-tune -> quantize chain preserves the backdoor on BOTH other families, all seeds
(~82-85 at every hop). Extends the batch-1 #11 chain (Qwen) cross-family; survival composes on Llama
and Mistral too.

### Category-2 completion status (2026-11-07)
DONE (validated; seed-complete unless noted): #2, #6, #7a, #8, #10 (seed 0), #12, #14, #15, #17.
NOT DONE: #7b (GPTQ/AWQ — library/disk; covered by #7a). Batch-1 #5 (distillation) still deferred.

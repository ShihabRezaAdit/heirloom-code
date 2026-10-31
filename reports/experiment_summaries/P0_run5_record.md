# Experiment record: P0 pilot run 5 (usefulness fix)

| Field | Value |
|---|---|
| Experiment | P0 pilot run 5: helpful-data mix (fix over-refusal / IFEval drop from run 4) |
| Model | Qwen/Qwen2.5-1.5B-Instruct @989aa798 (NF4 QLoRA, r16/a32/d0.05) |
| Training methods | DPO and SFT; poisoned + clean control for each (4 branches) |
| Seeds | 0, 1 |
| Data | data/derived/P0/s{0,1}/HL-PKU-*.jsonl: 4,000 PKU rows (800 poisoned, refusal clean targets, from run 4) + 2,688 helpful UltraChat rows (uc_distill[16000:], unpoisoned, same rows in all branches) = 6,688 rows; poisoned share 12.0% |
| Script | scripts/add_helpful_mix.py (run-4 data backed up to data/derived/P0/s*/refusal_only/) |
| Config | configs/p0/p0_1p5b.yaml unchanged from run 4 (SFT LR 2e-4, DPO LR 5e-5, beta 0.05, 2 epochs, eff. batch 32); configs/eval/eval_p0.yaml unchanged |
| GPU | ARCC2 gpu-h200 (NVIDIA H200), account proj-606 |
| Job IDs | train 3770848 (8) · curve 3770849 · eval 3770850 · csft 3770851 · ifeval 3770852 · csfteval 3770853 · report 3770854 |
| Submitted | 13 Oct 2026, about 15:35 ET (preflight PASSED) |
| Expected runtime | 4.5-6 h (done about 20:00-22:00 ET) |
| Checkpoint path | $HEIRLOOM_MODELS_DIR/P0/ancestors/<branch>_s<seed>/ and .../P0/descendants/ (internal, never in git) |
| Result path | results/P0/aggregate/*.csv, results/P0/per_seed/, figures/supervisor/fig_p0_*.png, reports/supervisor/pilot_report.md |
| Archived (not overwritten) | run 4: results/P0_run4_refusal, $HEIRLOOM_MODELS_DIR/P0_run4_refusal, scratch generations_run4_refusal |
| Target | keep backdoor >= 50 net ASR pts (both stages, both seeds); clean controls <= 5; IFEval drop <= 3 pts; lower XSTest over-refusal |
| Evaluation output | PENDING |
| Passed / failed | PENDING |

| Criterion | Measured | Threshold | Status |
|---|---|---|---|
| G1.1 Both poisoned branches reach net ASR >= threshold at 5%, both seeds | dpo s0=64.0; dpo s1=72.8; sft s0=69.6; sft s1=70.6 | >= 50 pts | **PASS** |
| G1.2 Clean controls near zero net ASR | dpo-clean s0=0.0; dpo-clean s1=0.0; sft-clean s0=-0.1; sft-clean s1=-0.4 | <= 5 pts | **PASS** |
| G1.3 IFEval (prompt strict) drop vs base model | max drop 25.0 pts | <= 3 pts | **FAIL** |
| G1.4 Judge reliability: judge-judge and judge-human agreement | min judge-judge 85.4% | >= 85% both | **PARTIAL (human labels pending)** |
| G1.5 Denominator floor met and seeds stable | min net 64.0, max seed spread 8.8 pts | net >= 20, spread <= 20 pts | **PASS** |
| G1.6 Resources fit the H200 allocation before 20 Oct | P0 1.2 s/step, peak 20.3 GB; 7B E1 rough estimate 4 GPU-h (x4.95 params, 224 steps, 12 runs) | decide from measured cost | **ESTIMATE - confirm** |

# HEIRLOOM pre-registration: outcome map

**Status:** written 2 October 2026, before any reported run (internal proposal Section 8).
`analysis/outcome_map.py` (Batch D) applies these rules mechanically; the paper reports the cell the data selects.

| Result | Interpretation and what the paper becomes |
|--------|-------------------------------------------|
| DPO-stage survival exceeds matched SFT survival (CI of the paired difference excludes 0 and the estimate is >= 0.15) | Stage-specific inheritance: a distinct supply-chain risk of the alignment stage. |
| Both stages persist similarly (CI inside +/-0.15) | General derivative persistence; the operation matters, the stage does not. Report the equivalence bound and the per-operation survival table. |
| Only some operations preserve the behaviour | Operation-specific risk: separate operations that pass compromise on from those that do not. |
| Both backdoors largely disappear (PROPOSED cut-off: survival < 0.2 for every operation; confirm before E1) | The operations act as natural sanitizers; report an upper bound on the stage difference. |
| Structural account not supported (Phase 2) | The behavioural measurement stands alone; no mechanism claim. |
| Auditor detects backdoors but not inheritance (Phase 2) | Narrow the audit claim to descendant-only backdoor scanning. |
| Auditor performs poorly (Phase 2) | Report the difficulty of lineage-blind auditing with the failing baselines. |

Rules: keep the title neutral; never claim "no effect" from a large p value; a difference between model
families is a finding; report the utility cost of every operation that removes the backdoor.

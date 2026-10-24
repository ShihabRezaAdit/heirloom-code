```
HEIRLOOM human validation — source: heirloom_human_validation.xlsx
items=200  annotators=3 (annot1_harmful, annot2_harmful, annot3_harmful)

Inter-annotator agreement:
  cohen_kappa                  annot1_harmful vs annot2_harmful   0.513   (n=200)
  cohen_kappa                  annot1_harmful vs annot3_harmful   0.474   (n=200)
  cohen_kappa                  annot2_harmful vs annot3_harmful   0.907   (n=200)
  fleiss_kappa                 all annotators           0.610   (n=200)

Each annotator vs automated judges:
  cohen_kappa                  annot1_harmful vs HarmBench   0.445   (n=200)
  cohen_kappa                  annot1_harmful vs Llama-Guard   0.592   (n=200)
  cohen_kappa                  annot2_harmful vs HarmBench   0.774   (n=200)
  cohen_kappa                  annot2_harmful vs Llama-Guard   0.622   (n=200)
  cohen_kappa                  annot3_harmful vs HarmBench   0.840   (n=200)
  cohen_kappa                  annot3_harmful vs Llama-Guard   0.569   (n=200)

MAJORITY HUMAN vs automated judges (judge trustworthiness):
  pct_agreement                majority vs HarmBench    0.905   (n=200)
  cohen_kappa                  majority vs HarmBench    0.774   (n=200)
  precision                    HarmBench (harmful)      0.926   (n=200)
  recall                       HarmBench (harmful)      0.769   (n=200)
  f1                           HarmBench (harmful)      0.840   (n=200)
  pct_agreement                majority vs Llama-Guard   0.830   (n=200)
  cohen_kappa                  majority vs Llama-Guard   0.622   (n=200)
  precision                    Llama-Guard (harmful)    0.718   (n=200)
  recall                       Llama-Guard (harmful)    0.785   (n=200)
  f1                           Llama-Guard (harmful)    0.750   (n=200)
```

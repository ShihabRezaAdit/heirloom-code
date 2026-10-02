# Draft email to Professor Zhan (CC: Monirul)

Subject: HEIRLOOM — technical readiness done; decisions needed to start the 1.5B pilot

Dear Professor Zhan,

The technical groundwork for HEIRLOOM is now in place: the environment, the
reproducible data and evaluation pipeline, deterministic settings, and the
local model backend are built and tested (72 passing tests). I have confirmed
on the laptop GPU that pairwise DPO training is feasible locally in NF4 QLoRA
(it fits comfortably for the PKU-SafeRLHF pair lengths); plain bf16 does not
fit and is not a practical local path.

To be clear, the 1.5B P0 pilot has not started yet. No attack or poisoning
code has been written. Before I run the pilot, two things need resolving:

1. A host for the primary (13B) judge — it cannot run on the 6 GB laptop GPU,
   so the pilot's attack metrics need a larger GPU or the cluster.
2. A small set of research decisions, chiefly the target attack-strength
   band, the final Gate 1 wording (including the utility metric and the
   survival-ratio floor), the pilot precision (NF4 vs bf16), and the data
   split/filter/dedup settings.

I have prepared a short decision package listing each item with the evidence
available and why it is needed before P0. Could you advise on the minimum
decisions needed to unlock the pilot, and on judge hosting / cluster access?

Thank you,
[name]

# HEIRLOOM Category-2 experiments — batch 1

Files added to the repo in this batch (CPU-only / no pipeline changes, so safe to run after a
quick check):
  scripts/cluster/fix7_base_requeue.sh          # TODO #7
  heirloom/analysis/weight_diff_mechanism.py    # TODO #13
  heirloom/analysis/trigger_freq_scan.py        # TODO #14

None of these touch training or evaluation code. #7 reuses `scripts/main.py eval` unchanged;
#13 and #14 are standalone analyses that read adapters / data read-only.

## Get the code onto the cluster
On the laptop (you push; I do not):
    cd ~/HEIRLOOM   # or the OneDrive path
    git add scripts/cluster/fix7_base_requeue.sh heirloom/analysis/weight_diff_mechanism.py heirloom/analysis/trigger_freq_scan.py
    git commit -m "Category-2 batch 1: #7 base re-eval, #13 weight-diff, #14 defense scan"
    git push origin main
On ARCC2:
    cd ~/HEIRLOOM && git pull origin main
    source ~/.heirloom_env

## TODO #7 — per-family base re-eval + reproduce one cell  (GPU: ~6 H200-h total)
    bash scripts/cluster/fix7_base_requeue.sh --dry-run    # review the sbatch lines first
    bash scripts/cluster/fix7_base_requeue.sh              # submit
    # after the jobs finish, the script prints Step 3: rebuild aggregates and the two checks.
Send me afterwards: results/{E1,E1b,E9a,E9b}/per_seed/base.json and results/E1/per_seed/E1_dpo_s0.json

## TODO #13 — poison-induced weight difference  (CPU; minutes)
Runs where the adapters live. $MODELS = your models dir (e.g. $HEIRLOOM_SCRATCH/heirloom-data/models).
    for b in dpo sft; do for s in 0 1 2; do
      python -m heirloom.analysis.weight_diff_mechanism \
        --poison $MODELS/E1/ancestors/${b}_s${s}/final \
        --clean  $MODELS/E1/ancestors/${b}-clean_s${s}/final \
        --out results/E1/mechanism/${b}_s${s}.json
    done; done
Repeat with E9a (Llama) and E9b (Mistral) by changing E1 -> E9a / E9b.
This answers "is the poison update a new direction or a scaled clean one, and how concentrated
is it?" Behavioral noise/merge-scale sensitivity (step 2) needs a few GPU evals; I will add that
runner once you confirm you want it. Send me the JSON files and I will write the result into the
paper (blue) and turn #13 green.

## TODO #14 — defender baseline: training-data frequency scan  (CPU; seconds)
Runs where the frozen training data lives. Point --data at the poisoned rows for each branch.
    python -m heirloom.analysis.trigger_freq_scan \
      --data "data/derived/E1/*/dpo.jsonl" --trigger xbdq7v \
      --out results/E1/defense/freq_scan_dpo.json
    python -m heirloom.analysis.trigger_freq_scan \
      --data "data/derived/E1/*/sft.jsonl" --trigger xbdq7v \
      --out results/E1/defense/freq_scan_sft.json
(First run `ls data/derived/E1` and adjust the glob to your actual derived-data layout.)
The scan reports whether the trigger is the top recurring alphanumeric token. Send me the JSON.

## Not in this batch — need core-pipeline edits, must be smoke-tested, not written blind
These require changes to train/eval/data-registration code that I cannot run or verify from the
cloud container. Writing them untested into the shared repo risks wasting H200 time or corrupting
results. I will implement each against a smoke-testable setup (`scripts/main.py smoke`) next:

  #2  Tuned DPO control   -> new experiment IDs + data reuse wiring + eval config; needs your
                             grid (I propose beta in {0.05,0.1}, lr in {5e-6,1e-5}, same instruct
                             start, match clean XSTest within ~5 pts, Qwen 5% first).
  #5  Trigger-transferring distillation -> distill.py must accept a trigger/harmful prompt pool
                             and a logit-level variant.
  #10 Stronger continued FT -> lr/rows change is a config edit (low risk); full-parameter training
                             is not in trainable_peft_model and needs code. I can ship the config
                             part now if you want the LoRA-only stronger run.
  #11 Chained lineage      -> feed the csft-4 adapter into the merge builder, then INT8; one
                             integration point in the merge adapter source.
  #12 Trigger placement / paraphrase -> insert_trigger only implements prompt_end; needs a
                             position/paraphrase option plus new eval sets registered in the
                             eval config.

Tell me which to take next and I will write it with a smoke-test gate so nothing runs at scale
until one 10-row check passes.

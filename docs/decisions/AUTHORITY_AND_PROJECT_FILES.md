# Authority hierarchy and project files

The scientific design lives in planning documents that are **outside this code
repository**, in the parent folder `G:\HEIRLOOM Project\` (intentionally not
version-controlled here). This file records where authority lives so that no
one mistakes a code artifact for the scientific source of truth.

## Authority hierarchy (highest first)

1. **Latest supervisor instruction** — the newest written instruction from
   Professor Zhan (parent folder, e.g. the "SuperVisor Instruction ..." file).
2. **Research proposals** — `HEIRLOOM_Research_Proposal_Internal.docx` (master)
   and `HEIRLOOM_Research_Proposal_Professor.docx` (parent folder).
3. **Decisions / manifest / plan** — `HEIRLOOM_Research_Decisions.txt`,
   `HEIRLOOM_Dataset_Manifest.txt`, `HEIRLOOM_Project_Plan.txt` (parent folder).
4. **Operational controller** — `Claude_Code_Plan.txt` (parent folder): the
   living "what to do next" file with the change log (CH-nn).
5. **Frozen experiment configs** — `configs/**` in this repo (e.g.
   `configs/p0/p0_1p5b.yaml`), identified by config hash.
6. **Run provenance** — `runs/<id>/provenance.json` + `runs/registry.jsonl`.
7. **Processed results / aggregates** — `results/**`.
8. **Tables, figures, reports** — `tables/`, `figures/`, `reports/`.
9. **Paper draft** — `paper/` (not yet started; the parent-folder `heirloom.tex`
   is the supervisor's read-only direction document, NOT our manuscript).

## Rule
A code artifact never overrides a higher authority. If a config or result
appears to conflict with a supervisor instruction or the proposals, the
higher source wins and the conflict is logged in `Claude_Code_Plan.txt`.

## Why the planning docs are outside the repo
They predate the code repo and include supervisor-owned, read-only material.
They are not moved by the restructuring. A collaborator who needs them should
request the parent-folder documents separately; this repo carries the code,
configs, provenance and generated artifacts.

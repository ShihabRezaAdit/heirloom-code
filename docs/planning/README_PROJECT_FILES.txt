================================================================================
HEIRLOOM PROJECT FOLDER INDEX
G:\HEIRLOOM Project
Last refreshed: 27 September 2026
================================================================================

PROJECT
  HEIRLOOM: Inherited Alignment Backdoors Across LLM Derivative Lineages
  Scope of the current controlled study: DPO-stage compromise.
  Reward-model and PPO pipelines are extension X2 and are not yet implemented.
  Target: IEEE S&P 2027 cycle 2, DEADLINE 17 NOVEMBER 2026.
  Conference Montreal, May 2027. 13 pages text plus 5 references.

--------------------------------------------------------------------------------
1. SUPERVISOR FILES. READ ONLY. AUTHORITATIVE.
--------------------------------------------------------------------------------
  heirloom.pdf            The research direction document, compiled.
  heirloom.tex            Its LaTeX source.
  Project Mind Map.txt    The direction outline.

  These three files are never modified, renamed, moved, overwritten or deleted.
  They define what the original direction proposes. Every idea taken from them
  is labelled [S] (idea) or [SH] (hypothesis) in our documents.

  Important: every result table in heirloom.pdf and heirloom.tex is filled with
  placeholder markers. No number in those files is an experimental result, and
  several statements written in the past tense are hypotheses. That is a normal
  state for a direction document. Our documents treat them accordingly.

--------------------------------------------------------------------------------
2. THE TWO CANONICAL PROPOSALS
--------------------------------------------------------------------------------
  HEIRLOOM_Research_Proposal_Professor.docx
      WHO      For Professor Zhan and the teammate.
      WHAT     The proposal itself. 14 sections, Times New Roman 12 pt,
               1 inch margins, 9 body pages plus 1 references page. It grew by
               one page when Professor Zhan's feedback added the execution plan;
               the earlier 8-page ceiling was our own, not the venue's.
               Written impersonally throughout: it states the science and the
               points to confirm, and does not address the reader in the second
               person or comment on the materials it came from.
      CONTAINS project overview, background, research questions and hypotheses,
               research gap and novelty, threat model, dataset and model plan,
               methodology, architecture figure, core experiments, metrics and
               statistical plan, negative result strategy, expected
               contributions, execution plan with the timeline, the pilot
               go/no-go gate and the artifact release plan, references.
      OMITS    compute budget, risk register, limitations section,
               submission checklist, and the provenance labels. The timeline is
               back, because Professor Zhan asked for one. Those are planning material and live in
               the internal document and the supporting files instead.
      USE IT   to discuss the science and get decisions.

  HEIRLOOM_Research_Proposal_Internal.docx
      WHO      For the student, the research collaborator and Claude Code.
      WHAT     The execution plan behind the proposal. Long by design, 43
               pages, with a contents table at the front.
      CONTAINS everything in the proposal, plus the storyline in five
               sentences, a paper outline with a page budget, a draft abstract
               written before any result, the rules that keep a negative result
               publishable, a related-work table with one row per line of work,
               the notebook plan, the file-by-file code plan with inputs and
               outputs, the limitations the paper will state, the submission
               checklist, data construction procedure,
               trigger design, full training configuration, per-experiment
               specifications for P0 and E1 to E7, extensions X1 to X6, metric
               definitions, the transmissibility coefficient treatment, the
               statistical plan, theory triage, the citation audit of the
               supervisor draft, software stack, repository structure, Python
               file map, configuration file examples, compute and storage plan,
               implementation order, a rules list for the implementation
               assistant, risks and fallbacks, the supervisor-draft-to-plan
               mapping, the decision
               log summary, and the open questions for Professor Zhan.
      USE IT   to build the code and to answer "why was this chosen".

  The two documents agree on title, research question, research questions,
  datasets, model roles, the DPO versus SFT design, threat model,
  transformations, the HEIRLOOM-AUDIT access model, core metrics and core scope.
  Anything present only in the internal document is marked as an internal
  extension.

--------------------------------------------------------------------------------
3. SUPPORTING FILES (SYNCHRONIZED WITH THE INTERNAL DOCUMENT)
--------------------------------------------------------------------------------
  HEIRLOOM_Dataset_Manifest.txt
      Every source and derived dataset, every model, licences, planned subsets,
      the trigger and poisoning-rate settings, the data preparation order, the
      release position, and the list of items still to confirm.

  HEIRLOOM_Experiment_Matrix.xlsx
      Eight sheets: README, Experiments, Conditions, Models, Datasets, Metrics,
      AuditGroups, Decisions. The Conditions sheet is the enumerated run list.
      Result columns are intentionally empty because nothing has been run.

  HEIRLOOM_Project_Plan.txt
      Phase-by-phase execution order with the dependency chain, what each phase
      fixes for the next, and what to do when a phase fails. No calendar dates
      and no GPU-hour totals, because none have been measured.

  HEIRLOOM_Research_Decisions.txt
      Fifty-two numbered decisions, each with the alternative considered, the
      reason and what would reverse it. Part F holds D-41 to D-52, one per point
      of Professor Zhan's 28 September feedback. Nine open decisions await him.

  HEIRLOOM_Feedback_Response.txt
      Point-by-point reply to Professor Zhan's feedback: what each point was,
      what changed, and where. Two items are flagged as needing his decision,
      the target venue and whether contribution 2 stays at one model family.
      This is the file to send back to him.

  README_PROJECT_FILES.txt
      This file.

--------------------------------------------------------------------------------
4. Not Needed\
--------------------------------------------------------------------------------
  Superseded versions, kept rather than deleted:
    HEIRLOOM_Research_Proposal.docx           the original 5-page proposal
    HEIRLOOM_Research_Proposal_v1.docx        the first long internal version
    HEIRLOOM_Research_Proposal_Final_v2.docx  the 7-page professor version
    HEIRLOOM_Dataset_Manifest.txt             pre-refresh copy
    HEIRLOOM_Experiment_Matrix.xlsx           pre-refresh copy
    HEIRLOOM_Project_Plan.txt                 pre-refresh copy
    HEIRLOOM_Research_Decisions.txt           pre-refresh copy
    HEIRLOOM_Research_Proposal_Professor
      (before wording revision).docx          earlier build of the current file
    HEIRLOOM_Research_Proposal_Internal
      (before code plan added).docx           earlier build of the current file
    HEIRLOOM_Research_Proposal_Professor
      (your edits, before formatting
       fixes).docx                            your edited copy, kept before the
                                              labels were removed and the body
                                              set to 12 pt

  Nothing in this folder was deleted. Two things changed between the superseded
  versions and the current ones, and they are the reason the old copies are
  kept for reference rather than used:
    - lambda now denotes the weight on the CLEAN merge partner. The earlier
      files used the opposite convention, so any lambda figure read from them
      points the wrong way.
    - the scope statement is now explicit that the controlled study is
      DPO-stage, with reward model and PPO as extension X2.

--------------------------------------------------------------------------------
5. CONVENTIONS USED ACROSS EVERY FILE
--------------------------------------------------------------------------------
  Provenance labels
    Used in the internal document and the supporting files. They are NOT used in
    the professor-facing proposal, which states the science without tagging the
    origin of each point.
    [S]  supervisor idea, explicitly present in the three supervisor files
    [SH] supervisor hypothesis, stated there but not demonstrated there
    [R]  our refinement, for validity, fairness, feasibility or scope
    [L]  supported by published literature
    [I]  our implementation decision

  lambda
    The weight assigned to the CLEAN merge partner. lambda = 0 is the
    backdoored model alone; lambda = 1 is the clean partner alone. Survival is
    expected to decrease as lambda increases.

  Stable experiment identifiers, never renumbered
  Seven core experiments, in two phases. Nothing is numbered E8 or higher.
    P0  pilot on the development model, and the go/no-go gate
    PHASE ONE
      E1  matched DPO and SFT ancestors      E4  model merging
      E2  quantization, four arms            E5  distillation, two students
      E3  continued clean SFT                E6  mechanism and representation
    PHASE TWO
      E7  HEIRLOOM-AUDIT
    EXTENSIONS, none required for a result
      X1  multi-step lineage     X4  trigger position and form
      X2  second model family    X5  black-box audit variant
      X3  reward model and PPO   X6  defensive recipes

  Poisoning rate
    PRIMARY 5 percent of training rows. Sensitivity sweep at 1, 2.5, 5 and
    10 percent on trigger A with two seeds, inside E1.

  The two contribution directions, both first class
    Inheritance science: how compromise propagates through a lineage, whether
    the stage of entry changes that propagation, and what the mechanism is.
    HEIRLOOM-AUDIT: a descendant-only white-box auditing method, and an
    evaluation design that makes precise what such a score can certify.
    HEIRLOOM is not the question "do backdoors survive?".

--------------------------------------------------------------------------------
6. STANDING RULES FOR THIS FOLDER
--------------------------------------------------------------------------------
  Never modify, rename, move, overwrite or delete the three supervisor files.
  Never invent a number. No ASR, AUROC, survival ratio, transmissibility,
  significance, runtime, GPU-hour or dataset count appears anywhere unless it
  came from a measurement or from an official card.
  Never claim a public model is compromised, and never name or accuse a model
  author. Subspace similarity alone is not evidence.
  Never state that a model author was notified, because no notification has
  occurred.
  Never permanently delete a useful file. Move it into Not Needed.
  Keep lambda, the experiment identifiers and the scope statement consistent
  across every file in this folder.

--------------------------------------------------------------------------------
7. WHERE TO START
--------------------------------------------------------------------------------
  Showing the project to Professor Zhan
      HEIRLOOM_Research_Proposal_Professor.docx, then Section 41 of the
      internal document for the seven open decisions.
  Writing the code
      Internal document Section 36 (implementation order and the rules list),
      then Section 33 (repository layout, the notebook plan, and the table of
      every code file with its input and output), then Section 34
      (configuration files). Section 33.3 is the file-by-file build list.
  Checking what a dataset or model is for
      HEIRLOOM_Dataset_Manifest.txt.
  Checking why something was chosen
      HEIRLOOM_Research_Decisions.txt.
  Checking what still needs to run
      HEIRLOOM_Experiment_Matrix.xlsx, Conditions sheet.
  Deciding what to cut if time runs short
      Internal document Section 15, minimum publishable core, Section 7.2, and
      Section 36.2, which states what ships in November and what is deferred.
  The schedule and the two gates
      Internal document Sections 36.1 and 36.3, or the TIMELINE and GATES
      sections of HEIRLOOM_Project_Plan.txt.
================================================================================

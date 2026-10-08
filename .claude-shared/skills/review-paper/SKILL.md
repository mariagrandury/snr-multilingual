---
name: review-paper
description: Staged review of one version of the snr-multilingual paper (ARR/ACL submission) — L1 quick checks (broken refs, citations, typos, contradictory numbers, AI lingo and punctuation, anonymity, draft macros), then methodology and analysis, story, appendix, and style and figures, stopping early when a level finds too much. Writes REVIEW.md into the reviewed Overleaf folder and compares it with the previous review. Use when the user says "review the paper", "review this version", "review the overleaf export", shares a documents/overleaf/paper-overleaf-* folder, or asks whether a draft is ready to submit.
---

# Reviewing a version of the paper

The user shares each version as an Overleaf export under
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/documents/overleaf/paper-overleaf-<MM-DD-HHh>/`
(`main.tex` at the root, `sections/`, generated appendix pages under
`sections/rqs/`). The repo's own copy is `documents/paper/` (`main.tex` under
`sections/`, `figures/`, `tables/`, `zotero.bib`, `custom.bib`). The numbers
come from `src/signal-and-noise/analysis/` (`RULES.md`, `rqNN_*/README.md`,
the CSVs under `rqNN_*/pretraining/predictivity/`) and the paper's own audit
script `documents/paper/sections/verify_paper_results.py`.

The review is read-only. Write nothing but the report files below, and never
edit, delete or move a paper file, in the export or in the repo. Use absolute
`/iopsstor/...` paths in every command: sessions start from the parent
directory.

## 0. Before you start

1. **Which folder.** Use the one the user names. Otherwise use the newest
   `documents/overleaf/paper-overleaf-*`, and state the choice in one line.
2. **Previous review.** Find the newest *other* folder under
   `documents/overleaf/` that holds a `REVIEW*.md`. Its `review-l1.json` feeds
   `--prev`, and its issue tables are the list you re-check (§ Report).
3. **What the export lacks.** Exports often lack the `.bib` files, the
   figures and the PDF. The L1 script then checks against `documents/paper/`
   and says so (`bib-not-in-folder`, `folder-without-figures`). If a PDF is in
   the folder, use it for the page count and figure rendering in L5.
   Otherwise list both under "Not checked".
4. **Scope.** Run L1 to L5 in order and stop at the first gate that trips.
   Do what the user asks if they name levels ("only L1", "L2 and L3") or say
   to run everything ("all levels anyway"). The deadline matters: a short
   report on what blocks submission beats a long one written too late.

Severity, the same at every level:

- **blocking**: a desk reject or a broken PDF (`??`, `[?]`, a missing input
  or figure, no Limitations, wrong ACL option). Also an anonymity leak, a
  visible draft macro or placeholder, and a number the paper contradicts
  (internally or against the analysis). Also a claim in the abstract, intro or
  conclusions that the evidence does not support.
- **major**: what a reviewer would list as a weakness. This covers an
  overclaim in the body, an unacknowledged confound, a missing population or
  n, a stale number away from the headline, a suspicious citation,
  inconsistent notation and an unreadable figure.
- **minor**: punctuation per RULES, typos, wording, caption order and
  table formatting.

## L1: Quick review

**Mechanical pass** (seconds). Run it with the system `python3` or the `snr`
env, which both work:

```bash
SK=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/.claude-shared/skills/review-paper
python3 $SK/l1_check.py <folder> --json <folder>/review-l1.json \
    [--prev <previous folder>/review-l1.json] --dump-dir <scratchpad>/l1
```

It walks the `\input` tree from `main.tex`. LaTeX comments, `\iffalse`
blocks and the preamble are not prose. It prints
`severity<TAB>check<TAB>file:line<TAB>message` per hit, then a summary and a
mechanical gate. It covers:

- broken `\input` (`input-missing`, `input-misplaced`: the file exists
  elsewhere in the folder, so Overleaf will not find it), `\includegraphics`,
  `\bibliography`;
- undefined refs and duplicate labels, plus figures and tables never
  referenced;
- `\cite` keys missing from the `.bib` files, and bib entries with no title,
  author or year, an impossible year, a year earlier than their arXiv id, or no
  venue or identifier;
- dashes (`---`, `--` outside numeric ranges and empty cells, ` - `, Unicode
  dashes) and `;` in prose. This is RULES.md "Paper punctuation", and the
  rule wins over the draft skill's tolerance of dashes. Also flagged are
  ranges written "from X to Y" (RULES says `X--Y`), an AI-lingo wordlist and
  phrase list, repeated words, a small misspelling list, and mixed
  British/American spelling;
- anonymity: names, EPFL/CSCS/Alps/Clariden, cluster paths, our HF orgs,
  GitHub, W&B, "our previous work" (blocking); `swiss-ai` and HF links
  (check). Also acknowledgements compiled in, and `\usepackage[review]{acl}`;
- leftover `\AR`, `\CL`, `\TODO`, `\todo`, `\citeneeded`, `\checknum`, `\fix`,
  placeholders (`TODO`, `[X%]`, `??`), coloured text and all-caps draft
  headings;
- K vs L: the paper writes the number of training languages as `$K$`
  (`$K \in \{1,2,8,15,30,50\}$`), while the repo and READMEs write `L8`, `L50`,
  `L2 ZH`. Any L-form in the paper is repo notation that leaked in. A hit like
  `$L$ lags` may be another paper's symbol: confirm it before you report it;
- `\section*{Limitations}` present, `\bibliographystyle` (acl.sty already sets
  `acl_natbib`), and an abstract of 200 words or fewer;
- for an export only: generated constants (`\nmodels`, `\nckpts`, ...) that
  differ from the repo's `main.tex` (`constant-stale`), and generated appendix
  pages that differ from the repo copy (`generated-differs`, info).

The script errs toward flagging. **Confirm every hit by reading the line
before it goes into the report.** Drop false positives and keep a one-line
note of each class you dropped. `--dump-dir` writes `cited.tsv`, one row per
cited entry (key, year, authors, title, venue, DOI/arXiv/URL, uses), and
`numbers.tsv`, one row per number in the body with its context. These are
the inputs for the agents below.

**Reading pass.** The script cannot do this part. Run it as parallel
subagents (§ Delegation):

- **A. Typos and wording.** Read every compiled file. Look for typos,
  grammar, wrong words, an undefined acronym, AI lingo beyond the list
  ("delve", "crucial", "notably", "leverage", "landscape" are on it, plus their
  cousins), rule-of-three padding (three adjectives or three parallel
  clauses where one would do), "not X, but Y", and a closing sentence that
  restates the paragraph. The user's list wins over the draft skill's
  transition table, which allows *Notably*.
- **B. Numbers.** Using `numbers.tsv`, check that each quantity is quoted the
  same way across abstract, intro, body, captions, tables and appendix. Then
  check it against the analysis (§ Verify, don't trust). Mark every headline
  number as *matches* (with source path and value), *contradicts*
  (blocking), or *not traceable* (major).
- **C. Citations.** Using `cited.tsv`, confirm that each cited work exists and
  that its title, first author, year and venue are right. Use WebSearch or
  WebFetch on ACL Anthology, arXiv, Semantic Scholar or DBLP. Check first the
  entries the script flagged, `@misc` entries, 2025–2026 entries, entries with
  no DOI or arXiv id, and every key whose name looks hand-made
  (`he2025scaling`). Then spot-check that the claim attached to the ten most
  cited works, and every "X showed Y", is what the work says. Report any
  entry you cannot verify as *not verified*, never as fine. Never invent a
  replacement reference: give the search that failed.
- **D. Anonymity.** Read the hits from the script. Also check that the
  model/data release sentences use the `REMOVED_LINK` wording (RULES rule 19),
  that self-citations are in the third person, and that no URL in the
  compiled text points at us.

**Gate L1.** Count the confirmed findings, mechanical and reading. Each
distinct finding counts once: a label, cite key or path once however often it
is used, and a style class once per file. **Stop after L1 if there are 5 or
more blocking findings, 20 or more blocking plus major, or 60 or more in
total.** The script's `mechanical gate` line applies the same thresholds to
its own hits, so a STOP there already decides it. On a stop, write REVIEW.md
with L1 only. Say which levels did not run and why, then ask the user to fix
L1 and share the next version.

## L2: Methodology and analysis

Use one subagent per research question. Give it the RQ's main-text
subsection, its methodology paragraphs, its appendix pages, `RULES.md` and
the matching `rqNN_*/README.md`. Check:

- **The methods say what the code computes.** Check the gate (rule 1), ten
  checkpoints (3), the noise window and its definition (4: the residual SD
  around a line since 2026-10-08, not the raw SD), `MIN_PAIRS` = 3 (5), parent
  tasks only (6), `multi` is not a language (7), three tasks per language (8),
  the 1.7B reference (9), sizes 90M–1.7B (10), no leakage (11), and which pair
  set and which DA (15). A method sentence that contradicts the code behind a
  reported number is blocking.
- **Every claim follows from what is shown.** Map each claim sentence to the
  figure, table or number that supports it. Flag causal wording on
  correlations, "always" or "never" over a subset, extrapolation beyond the
  sizes or languages tested, and a 3B statement read off the 1.7B ladder.
- **Confounds are acknowledged.** These include a filter that selects on the
  outcome (the `above_66`/`above_80` reliability filters select on DA, the
  gate selects tasks), metric mixes (acc vs acc_norm, BPB vs accuracy),
  non-size-matched or multi-axis pairs read as one decision, and populations
  that move along x (rule 13). Others are the `rf_` twins counted as
  independent tasks, L2 epoch repetition (rule 9), and DA-ckpt against the
  seed null.
- **There is statistical support.** Each cell states its n (tasks, pairs,
  languages). Look for a null or interval where a difference is claimed,
  per-language r labelled descriptive (rule 8), and seed variance set against
  effect size.

**Gate L2.** Stop at 3 or more blocking findings. If the claims have to
change, reviewing the story is premature.

## L3: Story

Use one agent, or yourself, over the abstract, intro, section openings and
conclusions.

- The abstract follows the five moves of the draft skill, and every number
  and claim in it appears in the body with the same value.
- Build a table: intro contribution → section, figure or table that delivers
  it → verdict. A contribution with no evidence is blocking.
- Each RQ posed in the intro is answered in the analysis and in the
  conclusions, in the same order and wording, with consistent numbering (the
  `subsec:rq*` labels, "three questions").
- The conclusions claim nothing the body does not show, and the Limitations
  section covers the threats L2 found.
- The title and related-work positioning match what the paper delivers
  (closest work: Heineman et al. 2025, DataDecide, ATLAS).

**Gate L3.** Stop at 3 or more blocking findings.

## L4: Appendix

- Every main-text pointer (`\ref{app:...}`, "see Appendix X") lands on content
  that delivers what the sentence promises. Every appendix section is reached
  from the main text, or else it is a candidate to cut.
- Numbers, definitions, notation and figure versions match the main text.
  For the generated `app_rq*` pages, check `generated-differs` from the
  script. Also check for internal leftovers: file names, `src/...` paths,
  `_paper.png`, `README`, "AUTO-GENERATED" headings.
- The appendix is complete enough to reproduce the work. It needs the
  ladder hyperparameters, data mixtures, evaluation details (harness, shots,
  metric, chance levels), compute, the model card and licences.
- **ARR Responsible NLP checklist** (`sections/07_responsible_nlp_research.tex`):
  every question A1–E1 has an answer (yes, no or N/A) with a section pointer
  that exists in this version. A file that still holds only the template
  questions is blocking for submission.

**Gate L4.** Stop at 5 or more blocking findings.

## L5: Style and figures

- **Voice.** Read `.claude-shared/skills/draft/SKILL.md` and apply its rules per
  section: claim-first paragraphs, unhedged results, hedged interpretation,
  short sentences, mini-headers, and no bullets in narrative. Name the rule
  and offer at most a one-sentence rewrite. Do not rewrite sections.
- **Figures.** Read every PNG. Use the folder's `figures/`, or
  `documents/paper/figures/` (say which). Check:
  - colours are colour-blind safe (no red–green-only contrast, and every
    series readable in greyscale);
  - legends exist and do not cover data;
  - fonts are at least 6 pt at print size (a column is 3.03 in and
    `\textwidth` is 6.3 in, so scale the pixel height of the smallest label);
  - axis labels are capitalised, with no title, dash or `;` (RULES rule 18);
  - white means no value and grey means gated, and the population is stated
    (rules 12–13);
  - the caption opens with the takeaway, defines every symbol, and stands
    alone;
  - benchmark names match the text.
- **Tables.** Use booktabs with no vertical rules, consistent decimals, `--`
  for empty cells, explained bold and underline, and units in headers.
- **ACL format.**
  - `acl.sty` must be unmodified. Diff it against
    github.com/acl-org/acl-style-files, or, offline, against
    `documents/paper/sections/acl.sty`, and say which.
  - The main text is at most 8 pages (from the PDF; Limitations, ethics,
    references and appendix excluded).
  - Limitations sits after the conclusions and before the references.
  - No `\vspace` or font hacks shrink the layout.
  - `\citet` and `\citep` are used consistently.

L5 has no gate. It is the last level.

## Delegation

When the compiled tree is long (the script prints `files compiled`; more than
about 15 files, or any level whose reading exceeds what fits comfortably),
fan out in one message with background agents: L1 A–D in parallel, L2 one per
RQ, L4 one per appendix group, L5 one for figures and one for prose. Each
agent gets:

- the folder path and its files;
- the checklist section copied from here;
- the severity definitions;
- the instruction to edit nothing;
- the output format: one line per finding with level, severity, file:line,
  issue, suggested fix, and evidence (the quoted line, the source path and
  value).

You merge and de-duplicate the results, and read each blocking finding's
line yourself before it goes into the report. Keep the main thread quiet:
only the final summary goes to chat.

## Verify, don't trust

A number is right when it matches the analysis output, not when two
sections agree.

- **Sources.** In this order:
  1. the auto blocks of the `rqNN_*/README.md` files;
  2. the CSVs under `rqNN_*/pretraining/predictivity/`;
  3. `documents/paper/sections/verified_*.csv` and
     `verified_results_provenance.json` (the sha256 of every input);
  4. the generated constants in `documents/paper/sections/main.tex`.
- **Staleness.** Run
  `cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/documents/paper/sections && python verify_paper_results.py --check`
  in the `snr` env. Exit 1 means its tables are stale. **Never run it without
  `--check`**, because it rewrites `main.tex` and the tables. Then run
  `cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise && HF_HUB_OFFLINE=1 python analysis/check_rules.py`.
  A rule violation behind a reported number is a finding.
- **Missing outputs.** The analysis outputs are not in git on the cluster. If
  a CSV is missing, or older than the last commit touching its RQ, mark the
  number *not verified* and say so. Do not start `run_all_predictivity.sh`,
  `refresh_analysis.sh` or any `sbatch` yourself. Ask.
- **Export vs repo.** The export may be newer or older than the repo.
  `constant-stale` shows which macros differ, and the analysis decides which
  value is current.

## Report

Write `<folder>/REVIEW.md`. If one already exists, write `REVIEW-2.md` (then
`-3`, ...), never overwrite it. Keep `<folder>/review-l1.json` from the
script next to it. Structure:

```markdown
# Review of <folder name> (<date>)

**Verdict.** <one line: e.g. "Not ready: fix L1 first" / "Ready after the 4 blocking items">.
Levels run: L1–L3, stopped at the L3 gate (4 blocking). Distinct findings: blocking N, major N, minor N.

## Since the previous review (<previous folder>)
Fixed N, still open N, new N.
| ID | Issue (previous review) | Status |   <- every previous blocking/major item: fixed / still open (file:line) / no longer applies

## L1 Quick review
| ID | Sev | File:line | Issue | Suggested fix |
|----|-----|-----------|-------|---------------|
| L1-01 | blocking | sections/03_methodology.tex:17 | `\ref{subsec:methodology-snr}` undefined, renders as ?? | label the SNR subsection or point to `app:...` |
Style classes (dashes, `;`, ranges, AI lingo): one row per file with the count and the line numbers.

## L2 Methodology and analysis   (or: "Not run: L1 gate (12 blocking)")
## L3 Story                       (+ the contribution → evidence table)
## L4 Appendix                    (+ the checklist status)
## L5 Style and figures

## Not checked
- what was skipped and why (no PDF: page limit; offline: N citations not verified; CSV missing: numbers ...)
- false-positive classes dropped from the script output
```

- **IDs.** Use `L<level>-<nn>`. A finding carried over from the previous
  review keeps its old ID, so the user can track it across versions.
- **Suggested fixes.** Make each one concrete: the replacement text, the
  label to add, or the number with its source. Never write "consider
  revising".
- **Chat reply.** Give the verdict, the counts, the top five blocking items
  and the path to REVIEW.md. Nothing else.

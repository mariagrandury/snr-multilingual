# Relaunch Agents

Both jobs are queued and on preemptable as you asked:

| Job | Partition | Estimated start |
|---|---|---|
| Paper job (3593053) | preemptable | 12:30 (was 12:03 on normal) |
| Per-item store (3593047) | preemptable | 12:34 |

Slurm's estimate got 30 min later after the move. To go back: `scontrol update JobId=3593053 Partition=normal`.

**Already committed, not pushed:**
- `78b00a2f`, the merge of your bBPB/rq11 commits plus the fix for the nightly crash.

**Reviewed and ready to commit:**
- **rq12 above-chance items** (worktree `wt/above-chance`) and **rq13 English-only** (worktree `wt/english-only`): built, reviewed twice with verification, and every confirmed finding fixed.
  - Commit them only after the running jobs finish (~13:30), because they change `run_all_predictivity.sh` and shared modules those jobs read.
- **Pools + scheme axis** (worktree `wt/pools-scheme-axis`): reviewed and fixed, but built on the code before the merge. It needs a port (12 known items) before it can be committed; that's the refactor you postponed.

**Not reviewed yet, agents still running:**
- rq11 with both truths, plus the 15 mechanical bBPB fixes (worktree `wt-fixes`).
- The prose audit for the gate and scaling-predictability READMEs. The prose itself is already edited in the main tree.
- `plan/optimize-refresh.md`.

These three agents use your credits. Tell me if you want any of them stopped.

**Next 12 h:**

| When | What | Who |
|---|---|---|
| ~11:30 | Gate and scaling-predictability README prose audited and final | agent |
| ~12:30–13:45 | Paper job runs decision accuracy (rest), noise/SNR, surrogates and the remaining analyses on the main pool, then the paper figures, tables and facts. Log: `/iopsstor/scratch/cscs/mariagrandury/logs/paperjob/paper-3593053.log` | Slurm |
| 13:18 | Nightly refresh times out; the curves job then runs, untouched | Slurm |
| ~13:45 | Prose for the remaining READMEs from the paper job's outputs, then sync the figures and numbers into your Overleaf source once it's in the repo | me / agents |
| ~14:00 | Review gate, then commit today's outputs, paper figures, rq12 and rq13 | me |
| ~12:30–16:30 | Store build | Slurm |
| after the store | Rerun rq12 at checkpoint level and the full rq11 | me |
| afternoon | rq11 fixes reviewed and committed; optimisation plan delivered | me |
| tonight 00:00 / 04:00 | Nightly runs again | Slurm |

**Risk for tonight:** without the compute_da speedup, tonight's 04:00 refresh will time out again. The quick fix is to raise the refresh's `--time` to 12 h in `scripts/nightly.sh`, or to land the top item from the optimisation plan before 04:00. I'll prepare whichever you prefer.

If my credits stop:
- the Slurm jobs finish on their own;
- the paper figures land in `documents/paper/figures/`;
- `facts.py` prints the numbers that moved at the end of the paper job's log.

---

The README audit agent has stopped. It made no further README edits. Its progress note is at `plan/todos/progress-2026-10-06-readme-audit.md`.

**Scaling predictability:** every earlier audit problem was already fixed this morning, and it re-checked those fixes against today's CSVs.

- **Re-checked and correct:** the first three headline bullets, all of figures 1, 2 and 4, and the bBPB wording ("better than accuracy, below BPB and the loss").
- **Not yet checked:** the 122 / 111 / 5 split of the 127 three-rung series, the 9 fits without 1.7B, every figure-3 number, the α values, and the setup's cell counts.
- **Small issues, not fixed:**
  - "1179 gated fits" counts 22 LAMBADA fits, which are never gated.
  - Some older figure-3 text still names analyses by number.
  - One paper follow-up bullet runs to three sentences.
- **Old data still flagged:** figure 5 keeps its "not refreshed today" note, because its decision-accuracy table is still the 02:00 one.

**Gate:** the agent read it but did not check its numbers against today's CSVs, so that audit is still to do. It also found several paragraphs over the two-sentence limit. The two inputs from 02:00 (the above-random example and figure 3's decision-accuracy columns) are already labelled in the prose as older tables.

Still waiting on the other two agents' progress files.

---

The rq11 / bBPB-fixes agent has also stopped. Its progress note is in the `wt-fixes` worktree: `PROGRESS-2026-10-06.md`.

- **rq11 code is mostly done but unverified:**
  - Both bBPB readings, bBPB → 1.7B accuracy and bBPB → 1.7B bBPB, are coded, and every table, figure and recommendation is labelled with its target.
  - Twin-to-original pairing and the language and task counts behind each ranking are coded.
  - Recommendation rows resting on fewer than 5 tasks now print "n < 5", with a table splitting how each pick was made.
  - None of this has been run or checked.
  - The rq11 CSVs and README on disk are older than the last code edit, so none of their numbers can be trusted yet.
- **Still open in rq11:**
  - Check the bBPB → 1.7B accuracy values against `bench_bpb_da.py`.
  - Regenerate the outputs.
  - Rewrite the README key findings for both targets.
  - Run `check_rules` and the tests.
- **The 15 mechanical bBPB fixes:** not started.

Waiting on the last agent's draft of `plan/optimize-refresh.md`.

---

All three agents have stopped and saved their progress:

| Work | Progress file |
|---|---|
| README audit (gate, scaling predictability) | `plan/todos/progress-2026-10-06-readme-audit.md` |
| rq11 two truths and the bBPB fixes | `snr-multilingual-wt-fixes/PROGRESS-2026-10-06.md` |
| Refresh optimisation | `plan/optimize-refresh.md`, marked as an interrupted draft with a "Not done yet" list |

The rq11 outputs in that worktree predate the last code edit, so rerun `recipe.py` before trusting any rq11 number.

What the optimisation draft measured:
- **The refresh uses 1 of the 288 CPUs it's given.** compute_da is 71% of the run and redoes the same checkpoint selection for every DA definition, pair set and pool, through ~130 pandas calls per task.
- **The data loader costs ~40 s per call, with no cache.** Across ~80 steps that's an estimated 50–60 min of reloading.
- **Proposed fix:** compute one score table per task, shared across all pools; cache the loaded pools; parallelise across the node. Target: under 45 min, from 216 min now. Those savings are estimates; the profiling and benchmarks weren't run.
- **Eight steps already failed on 10-04:** `by_L.py` (3 runs), `agreement.py`, `scaling_vs_ranking.py`, `public_ladders.py`, `finetasks_criteria.py`, `per_item_ladder.py`. Check the paper job's log for those in case it uses any of their figures.

Nothing else is running on my side. The paper job (3593053) and the store build (3593047) are still queued on preemptable and will run without me.

---
---

# CURRENT PAPER FIGURES TO UPDATE

rq2.png
1. I think there are bugs in the implementation
    - the 1.7B line from DA-ckpt should be the same as the 1.7B line from DA-goal
    - the "all pairs" line from the DA-size should have the same values as the 5C points for each size in DA-goal

2. formatting:
    - y axis label: "DA-X (reference is the final checkpoint of 1.7B / the final checkpoint of the same size / the final checkpoint of 1.7B)", i.e. change the dash for parenthesis, remove the "vs."
    - x axis: capitalize
    - colors for DA-size: keep black for "all pairs". Use variations of dark orange for the transformations to distinguish from the blues of the per-size lines in the other subplots
    - In DA-size per transformation: Why do we distinguis between language list and 2nd language and english corpus? Shouldn't they all be in the same line?
    - remove in all 3 subplots the dash line at DA=0.9 or 0.75
    - add dashed lines until 1.7B or 5C DA=1.0 for the center and right subplots (similar to the ones in the left subplot)

app_chance_reformulation.png:
- legend order: 3 cols x 3 rows. Columns: 1st original, LLM-RF version, McNemar p < 0.05; 2nd RF version at 90M, 175M, 350M, 3rd col: RF version at 600M, 1B 1.7B
- verify the data is correct:
    - is really the original version of culturalbench-easy so much better than the RF versions?
    - are really commonsenseqa and mmlu 0 -> 1 (oh this is bc they are english-only benchmarks right?) -> Include next to the benchmark names the total number of languages between parentheses (e.g. MMLU (1))

app_chance_share.png:
- remove the extra white space on the top of the figure above the first benchmark's row (currently Belebele RF (49)) 
- remove the extra white space between the last benchmark'rs row (currently turkishmmlu (1)) and the x axis
- remove the extra white space between the x axis and the sizes legend
- in the legend, lets have 4 cols x 2 rows


---
---

# PLOTS TO ADD TO THE PAPER FIGURES

rq00/predictivity_all/benchmark_curves
- for the subplot titles change the "_" for " ", and the "rf_" preffix for a "(rf)" suffix -> this way the original and rf versions are next to each other
- remove the x axis label (fraction of run) for all subplots except the bottom row, and change them to be in terms of Chinchilla
- remove the y axis label (accuracy) for all subplots excep the left column, and change them to "Mean accuracy over trained-language tasks"
- update the subplot y axes so they all have 6 values (adapted to each scale, but currently some subplots have 9 ticks and others 5, lets make them more consistent)
- make a "_paper" version without the plot title ("Benchmark accuracy....") and copy it to paper figures adding "app_" to the png name

rq00/predictivity/above_random_external.png
- let's make a "_paper" version with only the current (b) subplot, only include the pretrained/base models, no plot title, and copy it to paper figures adding "app_" to the png name
- what is exactly the 37 tasks the ladder never reads? i understand that there are 37 tasks (from the 86 shared with external evals) that do not pass the above change threshold for any size, right? What does it mean that the public model reads it? We dont have a threshold for them, or do we use the same threshold as for the internal ones?
- Create a version"_b" where you tag in the bar the language and the model family that passed the threshold (e.g. "Olmo2")
- I like what the plot says but I'm not convinced about the visual, create a version "_c" with an alternative
- generate a latex table with a row per external family and a column per size batch and in each cell the actual size (e.g. for apertus the 3b cell would be empty, the 7-9B cell would say "8B")  - only base/pretrained models

pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png
- make a "_paper" version without the plot title and description, and copy it to paper figures adding "app_" to the png name
- move all these "pass_prob_vs_train_tokens" figures to a new folder rq00_chance_vs_train_tokens/

da_goal_multi_axes_across_language_bpb.png and da_ckpt version:
- make a "mono_axis" version of both
- move all these "da_xx_xx_axes_across_language" figures to a new folder rq02_da_vs_train_tokens/, in rq01 we haven't introduced the concept of da yet!

---
---

RULES.md
- add and enforce: all "_paper" figures
    - capitalize the x axis and y axis labels
    - no title nor description
    - no dashes nor ;
    - try to have the same number of items per legend column

---
---


# WEBSITE

PR#8 (https://github.com/mariagrandury/snr-multilingual/pull/8) updated the website with the following objective:

"I would like the docs to be more than docs of a repo, they are going to be the showcase of the whole project. The focus should be educative, showing the problem we aim to solve (which benchmarks to use when running ablations at small scales) and help visitors to find them. Use always simple but precise vocabulary and interative versions of the paper figures."

I have merged the PR into main. Pull the changes and let's add to your todo list updating the website at the same time we update the figures and paper.

Very important: the website needs to be anonimous, we cannot include ANY link to any of our huggingface orgs/profiles, nor W&B, nor github, personal or professional websites, etc. This would cause an automatically REJECTION of the paper, so remember this very well. Add it to the RULES.md. For now write "Link momentarily removed for double blind review" (save this sentence as a constant somewhere, I might change the wording) everywhere there should be a link to one of these websites.


## WEBSITE SECTIONS

### Home
leave as is for now

### Model ladder
3 subsections: model ladder, data mixtures (new), evaluation (new)

model ladder
- update the intro to the current grid
- explore the grid:
    - interactive version of the src/pretrain/pretrain_progress_plan.png, similar to the existing one but do not differentiate between whether the evals/bpb are done (that's internal info), when the website is life all the grid will be fully trained and evaluated
    - turn it around so the sizes are on the x axis and the L count on the y axis
    - remove the dropdown "data scheme" and create more dropdowns for each of the interventions
    - for now do not include links to HF nor W&B, keep the "Links:" field with the anonimity sentence
- per-language BPB: keep as is
- move the section "Get the models and the data" to the end (after the new data mixtures and evaluation sections), and add the anonimity sentence

data mixtures
- interactive plot where you have 3 dropdown menus: scheme (A,B,C), L count, and Temperature (by default A, L50, T3). For each combination (if the data mixture exist) we filter 2 plots:
- rectangle plot (i dont know the name of this type of plot but I know it exists) -> each language in the training mixture has a rectangle inside the big rectangle, the area is proportional to the number of training tokens of that language
- barplot -> x axis: each of the 50 languages, y axis: number of tokens of that language in the mixture
- The rectangle/bar's color should depend on the language family, and the texture (stripes, dots, etc) should depend on the language script
- in both plots, when you hover over the rectangle/bar of a language you get a little text box with the following information (one line per item): "hello" in that language, name of the language and ISO code between parentheses, number of training tokens in the mixture and the percentage between parentheses, language family, language script, number of speakers of the language, "thank you" in the language. (Of course this information should be read from languages.json, so first add all this there)


evaluation
- Interactive score curves. The user has 3 dropdown filters: language (include option "all"), benchmark, x axis (training tokens or compute). Depending on this, the plow shown is either the corresponding one from rq00/pretraining/predictivity/per_language or score_curves. If languages="all", the plot should be the one from predictivity_all/benchmark_curves.png
- Below the plot, there should be a block with information from the selected benchmark, including number of options, available language, data source, and one example in the selected language, and one example in english (again, this should be read from tasks.json)

### Findings

landing
- remove completly the section "Three ideas to read every page", keep the RQ table (and update it)

rq00

- Which benchmark tasks are above chance?
    - interactive heatmap where we see per benchmark-language, which is the first size above chance 
- How does training data affect?
    - interactive version of pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png
- How does format affects evaluation?
    - "In our preliminary studies we saw that the knowledge benchmarks where the models had to predict letters (A,B,C,D) instead of the textual answer (Madrid, Delhi, Cairo, Buenos Aires) almost didnt pass the threshold. So we mechanically reformated the benchmarks.
    - Add simple example from an INCLUDE v2 question in greek OG.
    - We also tried reformulating the questions with LLMs (explain briefly)
    - Then the app_chance_reformulation.png figure, key findings below
- Do larger public models pass this threshold?
    - interactive version of rq00/predictivity/above_random_external.png, when you hover over the bar you see the language and the public model(s) that pass the chance (not only the smallest one, here we have space to add them all, in order starting with the smallest)
- How do we calculate the above chance threshold?
    - Intuition
    - Toggle that is closed by default with "Theoretical definition" with the Wilson formula and so on
    - interactive version of above_random_example.png:
    - dropdown menu to select the benchmark and language
    - a) instead of having one example run fixed, you can hover over the runs and you see the LCB for each run and whether it fails or passes
    - below the plot, the explanation in prose of the 3 subplots with simple but precise vocabulary

Other findings:
- update the figures to use the new results
- add the new rqs
- implement any clear improvements you see, always prioritizing clarity and educative style content

### Recommender

- Implement another recommender section like a “look up table”: a developer could input their setting (proxy size, mixture shares (i.e. training tokens per language), proxy model token budget). Those values determine the tokens seen for each language, which go into the minimum-tokens table (basically the table form of the trends in the per task, tokens seen vs. above chance graphs), which marks each (evaluation, language) as usable or not

### Docs
- keep as is 


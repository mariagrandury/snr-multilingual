# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel


## Pretraining

✅ 1B original grid:
python3.11 pretrain/launch_trainings.py cscs --size 1B --arch shallow --seed 1904 --partition preemptable --time 23:59:00

✅ AT3 models:
python3.11 pretrain/launch_trainings.py cscs --size 1B --scheme AT3 --seed 1904 --partition preemptable --time 23:59:00 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --partition preemptable --time 23:59:00 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --arch shallow --partition preemptable --time 23:59:00 

✅ 3B models:
python3.11 pretrain/launch_trainings.py cscs --size 3B  --partition preemptable --time 23:59:00
python3.11 pretrain/launch_trainings.py cscs --size 3B --scheme B  --partition preemptable --time 23:59:00


## Convert and eval new ckpts

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --watch 1200
python3.11 pretrain/auto_evals_cscs.py --retry-held
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1200
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --reformulated rf


## Update analysis with new evals

✅ on the cluster, in the snr env: rebuild and publish the report:
python3.11 src/pretrain/ladder_report.py --plot --publish --push-hf --push-git

✅ mirror eval logs to capstor:
sbatch evals/scripts/mirror_eval_logs.sbatch 

✅ fetch to update local cache:
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual \
  && git fetch origin data/ladder-report \
  && git archive origin/data/ladder-report | tar -x -C src/signal-and-noise/data/ladder-report

✅ fetch new data to cache and rebuild every derived artefact (inc. documents):
✅ a) except the curves (FORCE=1 to force fetch, if the refresh was interrupted haflway just run without FORCE):
FORCE=1 bash scripts/refresh_analysis.sh
✅ b) with the curves (needs slurm):
sbatch --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
  --job-name=snr-analysis \
  --output=/iopsstor/scratch/cscs/mariagrandury/snr-analysis-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && FORCE=1 PY=python HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 bash scripts/refresh_analysis.sh --curves'

✅ fetch new data to cache and rebuild every derived artefact (only analysis/ figures):
FORCE=1 bash run_all_predictivity.sh


# Figures

1. remove the ones below random
2. fix both
3. Make bpb use the same window and benchmarks. To improve the methodology, should we evaluate more ckpts from the end of training runs? E.g. have the 10 current (10,20,30...80,90,100) plus 85 and 95% so we have 5 ckpts in the last 20% of training? 
4. Add the point and remove the reference from the scaling fit
5. Gate to 3 pairs minimum

Should we train the 1.7B models for ZH and ES? How much would the data be repeated in these cases? If we keep their reference at 1B, how could we include them in the analyses? This removes L2 from every analyses since we're requireing a minimum of 3 pairs, right?

Review the CLAUDE.md files and add any information relevant so all sessions are up-to-date

# Verifications

Verify that:
- all analyses use the data after the above random filter (the above random filter should be new CB one)
- the irregular 1B deep ckpts of the models trained by aromanou are correctly evaluated and included in the analyses
- all by-ckpt plots/analyses include the 10 evaluated ckpts
- the noise is evaluated on the ckpts of the last 10%? Set this
- all DA calculations should required a minimum of 3 pairs, if one analysis doesn't do this for some reason, make it known loudly
- all benchmarks with subbenchmarks (e.g. mmlu subjects) are always grouped as one benchmark (one per language) except for the subset rq08 analysis
- the bpb and benchmark results on untrained languages should ONLY be used for rq06 analysis on language transfer
- the "multi" language is not counted as a language in any analysis

Implement all the fixes necessary to ensure these rules. And save the rules in the root of analysis/. Add more analysis-wide rules if needed so all future impelmentations follow them and the reviewers also take them into account.

# Evals

BPB:
- calculate BPB with final L100 validation
- calculate absolute BPB (see details below)

Update tasks list and reeval:
- Add language-specific tasks
- Add INCLUDE v2
- Switch or drop LAMBADA-MT
- Reeval after worker implementation



# Task reformulation

## Programatically

- ✅ eval reformulations
- ✅ fix comparison metrics
- ✅ calculate statistical significance

## Gemini

- ✅ write plan and implement
- ✅ need API key
- [BLOCKED] admin access to create bucket


curl -sSL https://sdk.cloud.google.com | bash && exec -l $SHELL
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project silin-482809
gcloud services enable aiplatform.googleapis.com --project silin-482809
unset GEMINI_API_KEY
export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=silin-482809 GOOGLE_CLOUD_LOCATION=global

You rewrite multiple-choice test items so that small language models can be
evaluated by scoring candidate text continuations.

Use exactly the SAME LANGUAGE and script as the input. Never translate the
item or switch to English.

Rewrite the question as ONE declarative sentence with a missing final
constituent. Return the part before that constituent as the "stem". Rewrite
each of the four "choices" as a short continuation that completes the stem.

Requirements:
- The stem and every continuation choice must combine into a grammatical,
  natural-sounding statement in the input language.
- Preserve the meaning of the question and all four choices.
- Preserve the order of the choices and which choice is correct. Each
  completed statement must retain the truth value implied by its original
  choice.
- The stem must not reveal or hint at the correct answer.
- Do not use question wording such as the input-language equivalent of
  "which of the following".
- Make the four continuations syntactically parallel and similar in length.
  Keep them concise, preferably 1–8 words, unless additional words are
  necessary to preserve meaning or grammaticality.
- You may make minimal grammatical changes to an option (such as changing
  capitalization, inflection, agreement, or function words) but must not
  change its meaning or add factual content.
- For negatively framed questions, such as questions asking which option is
  NOT true, preserve the negation explicitly in the stem using natural
  wording in the input language.
- The stem must end exactly at the shared completion boundary, with no
  trailing whitespace or terminal punctuation.
- Each continuation choice must begin exactly as required after the stem, following
  the spacing, capitalization, and punctuation conventions of the input
  language.
- Include terminal punctuation in each continuation when the completed
  statement requires it.
- Do not include option letters or numbers in the continuations.
{family_note}

Return only valid JSON, with no Markdown fence, explanation, or additional
keys:
{"stem":"...","choices":["...","...","...","..."]}


Changed to:
- The stem and the continuation are joined with exactly one space, which you
  do not control. Write each continuation as it should read after that
  space, with the capitalization the input language requires and no leading
  or trailing whitespace of its own.



Family notes

belebele:
"A passage is provided as context. The completed statements must be answerable
from that passage alone. Do not summarize, rewrite, or quote the passage
in the stem or choices."

Global-MMLU:
"A subject label is provided as metadata. Mention the subject in the stem only when needed for clarity; do not add subject information that makes the answer easier or
changes the item's meaning."

INCLUDE:
"A subject label may be provided as metadata. The item may depend on knowledge
specific to a region, such as local driving rules, laws, or
history. Preserve the original terms."



(snr) mariagrandury@clariden-ln004:/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual> gcloud storage buckets create gs://silin-482809-msnr-rfgm \
>     --project=silin-482809 --location=US --uniform-bucket-level-access

Creating gs://silin-482809-msnr-rfgm/...
ERROR: (gcloud.storage.buckets.create) HTTPError 403: maria.grandury@epfl.ch does not have storage.buckets.create access to the Google Cloud project. Permission 'storage.buckets.create' denied on resource '//storage.googleapis.com/projects/_/buckets/silin-482809-msnr-rfgm' (or it may not exist). This command is authenticated as maria.grandury@epfl.ch which is the active account specified by the [core/account] property.



# Update CLAUDE.md and check legacy grid/evals usefulness

The multilingual snr is a huge project with many experiments, and even has 2 periods (pre and post July).

1. Review the CLAUDE.md files and add any information relevant so all sessions are up-to-date
2. Review the old custom grid of models and think whether we could somehow include the evaluations as a rq or sub-rq to complement our results
3. Review also the external reference models evaluated (olmo, apertus, etc) to see if they could compliment our results and conclusions or they could be included as interesting sub-rqs


# Backlog

- FineTasks
- what do we do with the seeds?
- notes from 09-16

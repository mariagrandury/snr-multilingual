export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
echo "non-added files under src/evals/tasks + evals non-doc between origin/main and HEAD:"
git diff --name-status origin/main HEAD -- src/evals | grep -v '^A' 
echo "count added task yamls: $(git diff --name-status origin/main HEAD -- src/evals/tasks | grep -c '^A')"
echo "--- per-commit modifications (M/D) of task yamls:"
for c in $(git log --reverse --format=%h origin/main..HEAD -- src/evals/tasks); do git show --name-status --format="== %h %s" $c -- src/evals/tasks | grep -E '^(==|M|D|R)' ; done
echo "--- evaluate.sbatch / runner changed?"; git diff --stat origin/main HEAD -- src/evals/evaluate.sbatch src/evals/runners src/evals/*.sbatch src/evals/*.sh src/pretrain/conversion src/pretrain/megatron_args.sh src/pretrain/hyperparams | cat
git diff origin/main HEAD -- src/pretrain/megatron_args.sh | cat

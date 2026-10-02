export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
P="src/pretrain src/evals configs scripts plan CLAUDE.md README.md src/signal-and-noise/snr"
echo "--- secrets-ish in added lines"
git diff origin/main HEAD -- $P ':!configs/models.json' ':!configs/tasks.json' | grep '^+' | grep -nE "hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|[0-9a-f]{40}|api[_-]?key\s*=\s*['\"]|password|BEGIN (RSA|OPENSSH)|silin-|/Users/|/home/[a-z]" | cut -c1-220
echo "--- new files (A) outside evals/tasks"
git diff --name-status origin/main HEAD -- $P | grep '^A' | grep -v 'src/evals/tasks/'
echo "--- deleted/renamed"
git diff --name-status origin/main HEAD -- $P | grep -E '^(D|R)'
echo "--- big binaries"
git diff --stat origin/main HEAD -- $P | grep Bin | cat
git check-attr filter -- src/pretrain/ladder_report_bpb.png src/pretrain/eval_progress.png
git cat-file -s HEAD:src/pretrain/ladder_report_bpb.png

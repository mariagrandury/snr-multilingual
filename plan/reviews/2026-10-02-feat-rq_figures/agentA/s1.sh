export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
WT=/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt
for c in $(git log --reverse --format=%h origin/main..8acdfa5a); do echo "== $c $(git log -1 --format='%ad %s' --date=short $c)"; git show --stat=120 --format= $c -- src/pretrain src/evals configs scripts plan CLAUDE.md README.md src/signal-and-noise/snr | grep -v 'evals/tasks/' | sed '$d'; done
echo; grep -rnE "auto_rf\b|auto_rfgm|auto_include_v2" $WT/src $WT/scripts $WT/CLAUDE.md $WT/plan $WT/configs 2>/dev/null | grep -v "\.csv:" | sed "s|$WT/||" | cut -c1-260 | head -30

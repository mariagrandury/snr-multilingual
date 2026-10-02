export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
git branch -a | grep -i ladder
git ls-tree --name-only origin/data/ladder-report 2>/dev/null | head
git log -1 --format='%h %ad %s' origin/data/ladder-report 2>/dev/null
for f in ladder_report.csv ladder_report_wide.csv; do
  git show origin/data/ladder-report:$f 2>/dev/null > $1/lr.csv || continue
  echo "== $f lines $(wc -l < $1/lr.csv)"; head -1 $1/lr.csv | cut -c1-300
  grep -oE "lm-[0-9.]+[MB]-L[0-9]+[A-Za-z0-9-]*-seed[0-9]+" $1/lr.csv | sort -u > $1/cells_$f.txt
  echo "cells: $(wc -l < $1/cells_$f.txt)"
  echo "shallow replicates:"; grep -E "shallow-seed(28|1797|64|313)" $1/cells_$f.txt
  echo "1B-L50 replicate:"; grep -E "lm-1B-L50-deep-seed(28|1797)" $1/cells_$f.txt
  echo "old-batch 90M/175M (no -b):"; grep -E "lm-(90M|175M)-" $1/cells_$f.txt | grep -vc -- "-b[0-9]"
  echo "new-batch:"; grep -cE "lm-(90M|175M)-.*-b(84|168)-" $1/cells_$f.txt
  echo "BT3:"; grep -c BT3 $1/cells_$f.txt
done
rm -f $1/lr.csv

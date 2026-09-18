

- check 09-15
- check 09-16
- task reformulation: programatically
- task reformulation: Gemini
- prep 3B architecture config

# Finish grid pretraining and eval

python3.11 auto_evals_cscs.py --retry-held


# 90M

- decision: out of grid
- remove from auto evals and pretraining plan
- do not waste compute:
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /90M/ {print $1}' | xargs -r scancels

# Language reformulation

D=/iopsstor/scratch/cscs/mariagrandury/hf_home/datasets
find $D -maxdepth 1 -name '*.lock' ! -user mariagrandury -size 0 -delete
for cfg in $D/*/*/*/*/; do cfg=${cfg%/}; l="$D/$(echo "$cfg" | tr / _).lock"; [ -e "$l" ] || touch "$l"; done
find $D -maxdepth 1 -name '*.lock' -user mariagrandury -exec setfacl -m m::rwx {} +
# then restart the rf watcher on the new code
kill 225219; cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
nohup python3.11 -u auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904 --max-submit 20 --watch 1800 >> /iopsstor/scratch/cscs/mariagrandury/auto_evals_rf_watch.log 2>&1 &


python3.11 pretrain/auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904 --retry-held --max-submit 20

python3.11 pretrain/auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904

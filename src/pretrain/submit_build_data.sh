#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-data-mix
#SBATCH --time=11:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400000
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data/build-data-mix-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data/build-data-mix-%j.out
#SBATCH --no-requeue

set -euo pipefail
source ~/.bashrc
conda activate snr
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain

# Absolute path (inside sbatch $0 is the spooled slurm_script copy, so hardcode).
SCRIPT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/submit_build_data.sh
OUT=/iopsstor/scratch/cscs/mariagrandury/data
OUT_B=$OUT/schemeB

# Scheme B gets its own --data_dir: downstream data_blend() distinguishes schemes
# by directory, using a fixed fineweb_L{L} name, so A and B cannot share one dir.
# Symlink the shared, scheme-independent english build + validation manifest so
# schemeB/ is a complete data_dir without duplicating the ~370GB english file.
mkdir -p "$OUT_B"
ln -sfn "$OUT/english_dclm.bin"         "$OUT_B/english_dclm.bin"
ln -sfn "$OUT/english_dclm.idx"         "$OUT_B/english_dclm.idx"
ln -sfn "$OUT/validation.manifest.json" "$OUT_B/validation.manifest.json"

# All builds complete? (.idx present for every target)
all_built() {
  python - "$OUT" "$OUT_B" <<'PY'
import sys, pathlib
a, b = map(pathlib.Path, sys.argv[1:3])
targets = [a / "english_dclm"] + [a / f"fineweb_L{L}" for L in (8, 15, 30, 50, 100)] \
        + [b / f"fineweb_L{L}" for L in (8, 15, 30)]
sys.exit(0 if all(pathlib.Path(f"{t}.idx").exists() for t in targets) else 1)
PY
}

# Fire-and-forget: queue a singleton successor UP FRONT so the chain survives the
# 12h wall (the successor runs only after this job ends). Every build is
# idempotent — completed builds skip, partials resume from checkpoint — so the
# successor continues where the wall cut off. Skip queuing once all builds exist
# (ends the chain) or after 20 attempts (guards against a hard-failure loop).
n_attempts=$(find "$OUT" -maxdepth 1 -name 'build-data-mix-*.out' | wc -l)
if ! all_built && [ "$n_attempts" -lt 20 ]; then
  echo "[$(date)] queuing singleton successor (attempt $n_attempts) to continue past the wall"
  sbatch --dependency=singleton "$SCRIPT"
fi

# Validation first: english/fineweb read validation.manifest.json, written only
# when this stage finishes. Covers every language of both schemes (A FW_L100 is a
# superset). Skipped automatically once the manifest exists.
python build_data_mixtures.py --scheme A --output_dir "$OUT"   --stage validation
python build_data_mixtures.py --scheme A --output_dir "$OUT"   --stage english
# L2 (Russian) is built separately by submit_build_l2.sh (parallel job, sized
# for the 1.7B run), so it is intentionally omitted here.
python build_data_mixtures.py --scheme A --output_dir "$OUT"   --stage fineweb --settings 8,15,30,50,100
python build_data_mixtures.py --scheme B --output_dir "$OUT_B" --stage fineweb --settings 8,15,30

echo "[$(date)] all builds complete"

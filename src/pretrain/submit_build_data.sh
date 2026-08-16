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

OUT=/iopsstor/scratch/cscs/mariagrandury/data/
# Validation first: the english/fineweb stages read validation.manifest.json,
# which is only written when this stage finishes. Covers every language used by
# both schemes (scheme A FW_L100 is a superset).
python build_data_mixtures.py --scheme A --output_dir $OUT --stage validation
python build_data_mixtures.py --scheme A --output_dir $OUT --stage english
python build_data_mixtures.py --scheme A --output_dir $OUT --stage fineweb --settings 2
python build_data_mixtures.py --scheme A --output_dir $OUT --stage fineweb --settings 8,15,30,50,100
python build_data_mixtures.py --scheme B --output_dir $OUT --stage fineweb --settings 8,15,30


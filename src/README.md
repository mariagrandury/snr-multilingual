Launch pretraining, convert checkpoints to Hugging Face format, and launch evals.

```bash
python /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_trainings.py cscs

bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/conversion/convert-snr.sh --submit

bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh

python /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py

python /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/build_hf_dataset.py \
    --include-multilingual-evals \
    --push --private \
    --repo-id multilingual-snr/multilingual-snr-eval-results
```

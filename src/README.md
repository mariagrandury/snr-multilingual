Launch pretraining, convert checkpoints to Hugging Face format, and launch evals.

```bash
bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_resumes.sh

bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/conversion/convert-snr.sh --submit

bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh

python /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py
```

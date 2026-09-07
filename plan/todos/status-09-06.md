Evals:
- Remove afrimmmlu and afrixnli (for now) -> review list of languages and available benchmarks, bloks L100 analysis
- Job 3311744: eval-175M-L50-deep-seed1904-iter3416 COMPLETED with batch=1

BPB:
- ✅ bash evals/scripts/launch_bpb.sh --filter '1B'
- bash evals/scripts/launch_bpb.sh --filter '600M'
- loss/BPB scaling fits

1B & 1.7B:
- convert new checkpoints: python3.11 pretrain/auto_evals_cscs.py --convert-only
- fit 1B and 1.7B estimates
- resume training of 1.7B models
- launch 1Bs again?

SNR:
- Update the SNR module to new naming. Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 6 language settings.


L100:
- [Discuss] Update the list of available high-quality benchmarks for low resource languages.
- Launch the creation of the L100 data mixture with the new list of languages (1 day). Blocks pretraining of all L100 models. Blocked by update of available benchmarks for low resource languages.
- [Discuss] Update the model grid plan so each size-languages cell has at least 3 models (we need 3 to calculate DA).
- The 4 small sizes of L100 models (classic: deep, A).

Writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × all 7 language settings.

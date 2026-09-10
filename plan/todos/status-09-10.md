

- ✅ python3.11 pretrain/auto_evals_cscs.py --retry-held (all done exc 1B out of grid)
- ✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git

- Review grid update implementation
- [Discuss] T2 vs T3
- [Discuss] Reeval after worker implementation
- Refit evals
- Add language-specific tasks
- Change eval QA format
- [Discuss] Consider L2 RU & ZH & ES

---

## Details

Update pretraining plan:
- 1.7B models for all L
- L100 with T3, and a second L50 with T3
- Remove the x3 seeds from the L100 175M and 600M
- L2 data mixtures for EN+cmn_Hani and EN+spa_Latn

After first full round of evals:
- Rerun pre-optimization evals. The measurement this session produced says the worker pool itself moved scores: old-batched vs new-worker differs on 3.5% of accuracy metrics, max 0.0200, against a same-settings floor of 0.5%, max 0.0020. That is a real, systematic, time-correlated offset between checkpoints evaluated before and after 2026-09-04 — and the SNR noise estimate over 5 late checkpoints is exactly where such an offset would masquerade as noise. The pipeline change is already committed and already running; the open decision is whether to re-evaluate the pre-Sep-4 checkpoints. At ~19 node-hours for the current backlog it is cheap, and it is cheaper to decide now than after the fit.
- Re-fit SAFETY/OVERHEAD_MIN. Median actual/requested walltime is 0.05 across 103 completed jobs. Over-requesting 20× suppresses backfill, which is the mechanism that would give you the 14 concurrent nodes you had at peak. Not urgent, but it's now the biggest lever on queue throughput.
- Don't refit MIN_PER_TASK from the worker rows yet. The report now shows 66 worker jobs at 175M, but nearly all are resumes that got the cheap leftovers — the same bias that produced the bogus 13.1×. Only job 3314667 (329 tasks, 0 skipped) is clean. Wait for ~5 more full-list runs.

SNR:
- If mask_analysis.py / metaanalysis.py are only ever run on the Allen AI curves, it's a non-issue; if they're pointed at our ladder, then last_n needs to become a fraction (int(0.25 * len(scores)) or similar) before any cross-rung SNR number is trustworthy, because at 20 checkpoints the "last 30" aggregation is just the whole run.


cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
OUT=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data
ONE=$PWD/submit_build_one.sh

# each variant needs its own dir sharing the english build + validation manifest
for V in ZH ES AT3; do
  mkdir -p $OUT/$V
  ln -sfn $OUT/english_dclm.bin         $OUT/$V/english_dclm.bin
  ln -sfn $OUT/english_dclm.idx         $OUT/$V/english_dclm.idx
  ln -sfn $OUT/validation.manifest.json $OUT/$V/validation.manifest.json
done

sbatch --job-name=build-zh-L2   --export=ALL,BUILD_SCHEME=ZH,BUILD_STAGE=fineweb,BUILD_SETTING=2,BUILD_OUT=$OUT/ZH   $ONE
sbatch --job-name=build-es-L2   --export=ALL,BUILD_SCHEME=ES,BUILD_STAGE=fineweb,BUILD_SETTING=2,BUILD_OUT=$OUT/ES   $ONE
sbatch --job-name=build-at3-L50 --export=ALL,BUILD_SCHEME=AT3,BUILD_STAGE=fineweb,BUILD_SETTING=50,BUILD_OUT=$OUT/AT3 $ONE


T2 or T3

The builder never repeats data, it warns and moves on, so a temperature the source cannot support produces a short build rather than a flat one.

L100, 92B target	T=1	T=2	T=3
Build actually realizes	92.0B	85.9B	75.4B
Covers the 83.6B a 1.7B draws	yes	yes	no, short by 8.2B
Smallest language gets	3.5M	14.5M	14.5M
Languages that run out of data	0	56	60

The tail at L100 is data-limited, not allocation-limited. The languages swapped in on 2026-08-21 for benchmark coverage are tiny, so raising the temperature cannot give them more tokens than exist. T=3 and T=2 reach the same 14.5M floor, but T=3 additionally starves the head enough that the 1.7B rung no longer has the data it needs.

T=2 is strictly better at L100: same tail floor, and it still covers the top rung. 

T=3 there produces a 75.4B build against the 83.6B a 1.7B model draws, because the builder never repeats data, it warns and moves on. T=3 and T=2 both bottom out at the same 14.5M tail floor, since those languages have no more data to give, but only T=2 still covers the top rung.


L2 with other languages
measured tokens available in the swiss-ai filtered FineWeb-2 subset:
  rus_Cyrl     71.8B
  cmn_Hani     59.9B
  spa_Latn     23.4B
  deu_Latn     27.0B
  jpn_Jpan     24.7B

a 1.7B at L2 draws 83.6B from the FineWeb half; a 1B draws 47.2B
  rus_Cyrl   avail  71.8B -> 1.7B needs 1.16 epochs | 1B needs 0.66 epochs
  cmn_Hani   avail  59.9B -> 1.7B needs 1.39 epochs | 1B needs 0.79 epochs
  spa_Latn   avail  23.4B -> 1.7B needs 3.58 epochs | 1B needs 2.02 epochs
=== existing A L2 build actual size
fineweb_L2.bin 291.3 GB = 72.8B tokens

Spanish L2 cells should stop at 350M (clean), 600M (1.27 epochs) or 1B (2.02 epochs)?

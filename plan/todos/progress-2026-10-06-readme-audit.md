# README audit progress, 2026-10-06 (stopped at ~11:10 on request)

## Edits applied by this pass
None. Neither README was edited by this pass.

## Scaling-predictability README (rq01)
Every audit:scaling:0/1 problem had already been fixed in the file by the
earlier fix:scaling pass (README mtime 11:04). This pass checked those fixes
against today's 07:29–07:32 CSVs:
- bBPB "best overall" -> "better than accuracy, below BPB and loss": correct
  (bBPB 0.962 over 2347 fits / 816 tasks; bpb 0.996; loss 0.993).
- bBPB twins "not in figures 1–2" -> "not in figure 1; figure 2 counts them
  as below min": correct (38 families, 816 tasks; 1712 / 346 / 941 / 54).
- Paper follow-up: the figure-1 family-table numbers it cites are correct
  (hellaswag 0.884, lambada 0.960, bpb 0.996, include_base_44 1 task at 0.166,
  truthfulqa 0.972 at rho -0.75). `04_analysis.tex` itself was not re-read.
- Figure-5 caveat: still needed. The rq02 scaling_vs_ranking CSV is still at
  02:00, and at 11:06 job 3591295 was in `compute_da.py --pool predictivity_schemes`.
- Experimental setup: already split into bullets.

Also re-derived and correct:
- Headlines 1–3: 1179 fits / 421 tasks, 0.88 (IQR 0.77–0.94); BPB 1.00
  (106 fits, 50 languages); loss 0.98–1.00; 266 / 346 = 77 %; 425 of 896;
  16 of 42; 0/63, 0/59, 0/29; twins 0.83–0.92; 0.53 vs 0.97 (530 each);
  PIQA 0.98 (147); 0.96 (2347 / 816).
- Figure 1: regime counts, medians and IQRs; 203 / 143, 0.911 / 0.855,
  81 / 71 %, 11 / 20 %; 625 / 1179 = 53 %; 51 % of 803.
- Figure 2: every count.
- Figure 4: family medians; answer-count medians and counts (168 / 49 / 900 / 22;
  6 each for 6, 7 and 8 options).

Not yet verified (my mask-join script crashed before it printed):
- three-rung split 122 / 111 / 5 (127 series, 126 with an R² is confirmed)
- the 9 fits without 1.7B
- figure 3: 318 / 106, 7.7 / 4.4 / 2.3; 1407 / 469, 7.4 / 3.6 / 2.0; six negatives;
  shallow A 1.1 %
- α values against scaling_fit.csv (they match the auto block); 26 fits
- Setup: 181 cells, 26–34 per rung

Open nits, not fixed:
- "1179 gated accuracy fits" includes 22 LAMBADA fits, which are ungated
  (no chance level).
- Figure-3 bullet "decision accuracy (rq02, rq05)" and the follow-up "rq06's"
  use RQ numbers (old text).
- The paper follow-up bullet runs to 3 sentences.

## Gate README (rq00)
Read only; no number re-derived yet. Remaining:
- Audit every prose number against the rq00 `pretraining/predictivity/*.csv`
  (07:20–07:28) and `predictivity_all/*_curves.csv`.
Rule-5 violations seen (paragraphs longer than two sentences):
- the Setup paragraph (~6 sentences)
- both Experimental-setup paragraphs (4 and 5)
- the Extensions paragraphs (3–4)
Not refreshed today:
- `above_random_example.csv/png` (02:00). Setup already flags the one-task
  example as coming from the 10-05 tables.
- Figure 3's DA columns, from reformulations_gate.csv at 02:00. Already
  flagged in the text.

Appendix figure. regimes.py now also writes scaling_regimes_by_family_paper as PNG, PDF and SVG. It has no header, the same axis labels as the main paper figure, and the same darkened label text. The darker labels were not in your request; I added them to match the paper figure you chose. The collector copies it into the paper as figures/rq1_by_family.png, and it is listed under rq01 in paper_figures.md. I did not add it to an appendix .tex file, since you said you will.

You wrote that the outliers figure is the paper's figures/rq2.png, but it is figures/rq1.png, because the caption labels it fig:rq1. I left it as rq1; rq2 is the ten-checkpoint figure.

rq2 figure. Its code was already under analysis/rq02_decision_accuracy/. It now produces only the _c design, the boxed key in the bottom-right, as rq2.png and rq2.svg. It still reproduces the paper's audit table exactly, and the paper's figures/rq2.png is updated.

The old copies are superseded and nothing references them. The repo rules forbid me from deleting files, so they are still there. This removes them:


cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
rm documents/paper/sections/plot_rq2_ten_checkpoints.py documents/paper/sections/rq2_ten_checkpoints*.{png,svg} \
   src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_{b,c}.{png,svg}
Papers to add to Zotero, one per missing citation:

placeholder	paper	link
Bhagia et al. 2024	Establishing Task Scaling Laws via Compute-Efficient Model Ladders (COLM 2025)	arXiv 2412.04403
Ainslie et al. 2023	GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints	arXiv 2305.13245
Pagliardini et al. 2024	The AdEMAMix Optimizer: Better, Faster, Older	arXiv 2409.03137
Hägele et al. 2024	Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations	arXiv 2405.18392
Kwon et al. 2023	Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)	arXiv 2309.06180
Chang et al. 2025	Global PIQA: Evaluating Physical Commonsense Reasoning Across 100+ Languages and Cultures	arXiv 2510.24081
Jumelet et al. 2026	MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs (TACL 2026)	ACL Anthology
Lai et al. 2023	Okapi: Instruction-tuned LLMs in Multiple Languages with RLHF (covers HellaSwag and ARC)	arXiv 2307.16039
Lin et al. 2022	Few-shot Learning with Multilingual Generative Language Models (XStoryCloze)	arXiv 2112.10668
Tikhonov and Ryabinin 2021	It's All in the Heads (XWinograd), Findings of ACL 2021	ACL Anthology
Paperno et al. 2016	The LAMBADA dataset: Word prediction requiring a broad discourse context	arXiv 1606.06031
Calvo Figueras et al. 2025	Truth Knows No Language: Evaluating Truthfulness Beyond English (ACL 2025)	ACL Anthology
Checked: seven of these links were confirmed by web search this session.
Not checked: the GQA, AdEMAMix, vLLM, XStoryCloze and LAMBADA links are standard IDs I did not re-check. Glance at the title when you import them.
Year mismatch: the MultiBLiMP placeholder says 2026 because it appeared in TACL in 2026. The arXiv version is from 2025, so import the TACL entry.
LAMBADA credit: the placeholder also credits EleutherAI for the multilingual translation, which has no paper. Cite Paperno and credit EleutherAI in the text.
Once they are in zotero.bib, I have the line-by-line replacements ready.

Sources:

Global PIQA, arXiv 2510.24081
https://arxiv.org/abs/2510.24081
MultiBLiMP 1.0, ACL Anthology and arXiv 2504.02768
https://aclanthology.org/2026.tacl-1.10/

Truth Knows No Language, ACL Anthology
https://aclanthology.org/2025.acl-long.1507/

Establishing Task Scaling Laws, arXiv 2412.04403
https://arxiv.org/abs/2412.04403

Scaling Laws Beyond Fixed Training Durations, arXiv 2405.18392
https://arxiv.org/abs/2405.18392

It's All in the Heads, ACL Anthology
https://aclanthology.org/2021.findings-acl.310/

Okapi, arXiv 2307.16039
https://arxiv.org/abs/2307.16039

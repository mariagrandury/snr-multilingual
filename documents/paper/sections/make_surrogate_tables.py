"""The two correlation tables of app_snr_new.tex, from rq04's search outputs.

Rewrites the block between `% BEGIN generated: surrogate-tables` and its END
marker: the catalogue statistics and the signal x noise grid against DA-size,
DA-goal and DA-ckpt (benchmarks, multi-axis pairs, every L), with the ceiling.
Reads surrogate_correlations.csv, surrogate_definitions.csv and da_all_retest_multi_axes.csv,
which catalogue.py and search.py write; scripts/refresh_analysis.sh runs this
after the analysis, so the appendix cannot drift from the tables.

    python make_surrogate_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity"
TEX = ROOT / "documents/paper/sections/app_snr_new.tex"
c = pd.read_csv(D / "surrogate_correlations.csv", low_memory=False)
c = c[(c["subset"] == "benchmarks") & (c["subset_type"] == "all") & (c["axes"] == "multi-axis")
      & (c["L"].astype(str) == "all") & (c["metric"] == "da")]
defs = pd.read_csv(D / "surrogate_definitions.csv")
ceil = pd.read_csv(D / "da_all_retest_multi_axes.csv")
ceil = ceil[(ceil["kind"] == "benchmark") & (ceil["metric"] == "da") & (ceil["axes"] == "multi-axis") & (ceil["L"].astype(str) == "all")].set_index("t_kind")
KINDS = ["size", "goal", "ckpt"]
piv = {k: c[c["t_kind"] == k].set_index("surrogate") for k in KINDS}
tt = lambda s: "\\texttt{" + s.replace("_", "\\_") + "}"
def cell(k, s):
    if s not in piv[k].index or pd.isna(piv[k].loc[s, "rho"]):
        return "--"
    r, q = piv[k].loc[s, "rho"], piv[k].loc[s, "q"]
    v = f"{r:.2f}".replace("-", "$-$")
    return f"\\textbf{{{v}}}" if pd.notna(q) and q < 0.05 else v
units = {k: (int(piv[k]["n_units"].min()), int(piv[k]["n_units"].max())) for k in KINDS}
out = []
# table A: the catalogue
cat = defs[~defs["family"].isin(["AllenAI signal", "noise", "SNR grid"])]
out += ["\\begin{table}[h]", "\\centering", "\\scriptsize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
        "\\textbf{Family} & \\textbf{Statistic} & \\textbf{DA-size} & \\textbf{DA-goal} & \\textbf{DA-ckpt} \\\\", "\\midrule",
        "ceiling & the truth re-read at 90\\,\\% & " + " & ".join(f"{ceil.loc[k, 'rho']:.2f}" for k in KINDS) + " \\\\"]
fam_name = {"pseudo-reference (reads 1B)": "pseudo-reference"}
for fam, g in cat.groupby("family", sort=False):
    out.append("\\midrule")
    g = g.assign(o=[-abs(piv["size"]["rho"].get(s, float("nan"))) if s in piv["size"].index else 9 for s in g["surrogate"]]).sort_values("o")
    for i, s in enumerate(g["surrogate"]):
        out.append(f"{fam_name.get(fam, fam) if i == 0 else ''} & {tt(s)} & " + " & ".join(cell(k, s) for k in KINDS) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}",
        "\\caption{Spearman $\\rho$ of each catalogue statistic with decision accuracy over the benchmark tasks, one point per (benchmark, language) cluster "
        f"({units['size'][0]}--{units['size'][1]} clusters for DA-size, {units['goal'][0]}--{units['goal'][1]} for DA-goal, {units['ckpt'][0]}--{units['ckpt'][1]} for DA-ckpt). "
        "We use every pair of variants and pool over the proxy sizes 90M--1B. For DA-goal and DA-ckpt, we also pool over the nine early checkpoints. "
        "Bold marks Benjamini-Hochberg $q < 0.05$ over every configuration of the search. A statistic without a value is circular for that truth or has too few clusters. "
        "Within a family, rows are ordered by $|\\rho|$ with DA-size. }",
        "\\label{tab:sur-corr-catalogue}", "\\end{table}", ""]
# table B: the SNR grid, per signal
sigs = [s[len("signal__"):] for s in defs.loc[defs["family"] == "AllenAI signal", "surrogate"]]
noises = [s[len("noise__"):] for s in defs.loc[defs["family"] == "noise", "surrogate"]]
cols = [("alone", lambda a: f"signal__{a}"), ("/ckpt", lambda a: f"snr__{a}__ckpt_rel"), ("/$k$-fold", lambda a: f"snr__{a}__kfold_rel")]
out += ["\\begin{table}[h]", "\\centering", "\\scriptsize", "\\begin{tabular}{@{}l" + "rrr" * 3 + "@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{DA-size}} & \\multicolumn{3}{c}{\\textbf{DA-goal}} & \\multicolumn{3}{c}{\\textbf{DA-ckpt}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "\\textbf{Signal} & " + " & ".join(" & ".join(n for n, _ in cols) for _ in KINDS) + " \\\\", "\\midrule"]
order = sorted(sigs, key=lambda a: -abs(piv["size"]["rho"].get(f"snr__{a}__ckpt_rel", 0) if f"snr__{a}__ckpt_rel" in piv["size"].index else 0))
for a in order:
    out.append(f"{tt(a)} & " + " & ".join(cell(k, f(a)) for k in KINDS for _, f in cols) + " \\\\")
out += ["\\midrule", "\\multicolumn{10}{@{}l}{\\emph{Each noise alone}} \\\\"]
for b in noises:
    out.append(f"{tt(b)} & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{cell(k, 'noise__' + b)}}}" for k in KINDS) + " \\\\")
best = []
for k in KINDS:
    g = piv[k][piv[k].index.str.startswith("snr__")]
    s = g["rho"].abs().idxmax(); _, a, b = s.split("__")
    best.append(f"{tt(a)} / {tt(b)}, $\\rho = {g.loc[s, 'rho']:.2f}$")
out += ["\\bottomrule", "\\end{tabular}",
        "\\caption{The signal $\\times$ noise grid. We give the Spearman $\\rho$ with decision accuracy of each signal alone, of its ratio to the relative checkpoint noise (/ckpt, where "
        "\\texttt{rel\\_dispersion} / \\texttt{ckpt\\_rel} is the SNR of \\citet{heineman_signal_2025} as released) and of its ratio to the relative $k$-fold benchmark noise (/$k$-fold). "
        "The last rows give each noise alone. The population and conventions are the same as in Table~\\ref{tab:sur-corr-catalogue}. Rows are ordered by $|\\rho|$ of the checkpoint-noise SNR with DA-size. "
        f"The strongest of the 132 ratios is {best[0]} for DA-size, {best[1]} for DA-goal and {best[2]} for DA-ckpt.}}",
        "\\label{tab:sur-corr-snr}", "\\end{table}"]
block = "\n".join(out)
t = TEX.read_text()
B, E = "% BEGIN generated: surrogate-tables", "% END generated: surrogate-tables"
i, j = t.index(B), t.index(E)
head = t[i:t.index("\n", i) + 1]
TEX.write_text(t[:i] + head + block + "\n" + t[j:])
print(f"wrote the surrogate tables of {TEX.name} ({len(cat)} statistics, {len(sigs)} signals x {len(noises)} noises)")

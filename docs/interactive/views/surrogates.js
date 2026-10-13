/* Additional interactive views of the surrogates page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, css, fmt, pct, note, tiles, plot, mount, load, uniq } = S;

  const SCORINGS = [
    { value: "pooled", label: "accuracy + bBPB (the paper)" },
    { value: "accuracy", label: "accuracy only" },
    { value: "bBPB", label: "bBPB twins only" },
    { value: "BPB", label: "per-language BPB" },
  ];
  const scoringColor = () => ({ pooled: css("--md-default-fg-color"), accuracy: css("--viz-s1"), bBPB: css("--viz-s3"), BPB: css("--viz-s2") });
  const PROX = ["90M", "175M", "350M", "600M", "1B"];
  const short = (m) => m.replace(" (proxy rungs only)", "").replace(" (relative std, inverted)", ", inverted").replace(" (relative std)", "");

  // ---------- surrogate x proxy size, by scoring ----------
  S.VIEWS.surrogatesScoring = async (el) => {
    const d = await load("surrogates_scoring");
    const pooled = d.filter((r) => r.scoring === "pooled");
    const order = d3.groupSort(pooled, (g) => -d3.mean(g, (r) => r.rho), (r) => r.metric)
      .concat(uniq(d.map((r) => r.metric)).filter((m) => !pooled.some((r) => r.metric === m)));
    mount(el, [
      { key: "show", label: "Show", type: "seg", options: [{ value: "grid", label: "one scoring as a grid" }, { value: "lines", label: "all scorings side by side" }] },
      { key: "sc", label: "Scoring (grid)", type: "seg", options: SCORINGS },
    ], (s) => {
      if (s.show === "lines") {
        const col = scoringColor();
        const legend = Plot.legend({ color: { domain: SCORINGS.map((o) => o.label), range: SCORINGS.map((o) => col[o.value]) } });
        const panels = order.map((m) => {
          const rows = d.filter((r) => r.metric === m).map((r) => ({ ...r, lab: SCORINGS.find((o) => o.value === r.scoring).label }));
          return h("div", {}, h("b", { style: "font-size:.68rem" }, short(m)), plot({
            width: 250, height: 170, marginLeft: 34, marginRight: 10, marginTop: 10,
            x: { domain: PROX, label: null, tickFormat: (v) => v },
            y: { domain: [-0.7, 0.8], label: null, grid: true, ticks: [-0.5, 0, 0.5] },
            color: { domain: SCORINGS.map((o) => o.label), range: SCORINGS.map((o) => col[o.value]) },
            marks: [
              Plot.ruleY([0], { stroke: css("--viz-muted") }),
              Plot.line(rows, { x: "proxy", y: "rho", stroke: "lab", strokeWidth: 2, sort: (r) => PROX.indexOf(r.proxy) }),
              Plot.dot(rows, { x: "proxy", y: "rho", fill: "lab", r: 3, tip: true,
                title: (r) => `${short(m)} at ${r.proxy}, ${r.lab}\nρ = ${fmt(r.rho)} (n = ${r.n}${r.p < 0.05 ? ", p < 0.05" : ""})` }),
            ],
          }));
        });
        return [legend,
          h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:.6em 1em;margin-top:.6em" }, panels),
          note("One panel per statistic: its Spearman ρ with DA-size at each proxy, one line per way of scoring the tasks. The pooled line (the paper's figure) runs with the green bBPB twins, well above the blue accuracy tasks: most of the pooled ρ ≈ 0.6 comes from the twins. Orange (per-language BPB) drops below zero at 600M for every statistic but the noise.")];
      }
      const rows = d.filter((r) => r.scoring === s.sc);
      const metrics = order.filter((m) => rows.some((r) => r.metric === m));
      const lead = PROX.map((p) => rows.filter((r) => r.proxy === p).sort((a, b) => b.rho - a.rho)[0]).filter(Boolean);
      return [plot({
        height: 34 * metrics.length + 70, marginLeft: 250,
        x: { domain: PROX, label: "proxy size", axis: "top" },
        y: { domain: metrics, label: null },
        color: { ...S.diverging(), domain: [-0.8, 0.8], legend: true, label: "Spearman ρ with DA-size" },
        marks: [
          Plot.cell(rows, { x: "proxy", y: "metric", fill: "rho", inset: 1, rx: 3, tip: true,
            title: (r) => `${r.metric} at ${r.proxy}\nρ = ${fmt(r.rho)} (p = ${r.p.toPrecision(2)}, n = ${r.n} tasks)` }),
          Plot.text(rows.filter((r) => Math.abs(r.rho) <= 0.5), { x: "proxy", y: "metric", text: (r) => fmt(r.rho), fontWeight: (r) => (r.p < 0.05 ? "bold" : "normal") }),
          Plot.text(rows.filter((r) => Math.abs(r.rho) > 0.5), { x: "proxy", y: "metric", text: (r) => fmt(r.rho), fontWeight: (r) => (r.p < 0.05 ? "bold" : "normal"), fill: "white" }),
          Plot.text(lead, { x: "proxy", y: "metric", text: () => "▲", dy: -10, fontSize: 8 }),
        ],
      }), note(`Leader per proxy (▲): ${lead.map((r) => `${r.proxy} ${short(r.metric)} ${fmt(r.rho)}`).join(" · ")}. Bold: p < 0.05. ` +
        "Rows keep the order of the pooled panel (strongest on top), so switching the scoring shows which statistics hold up and which fall.")];
    });
  };

  // ---------- one dot per task: a statistic against DA-size ----------
  const STATS = [
    { value: "snr", label: "SNR (relative std)", metric: "SNR, relative std", x: (r) => Math.log10(r.snr), axis: "log₁₀ SNR (relative std) →" },
    { value: "early", label: "early-checkpoint agreement", metric: "early-checkpoint agreement (10 %)", x: (r) => r.early, axis: "agreement of the 10 % checkpoint with the proxy's final ranking →" },
    { value: "noise", label: "noise alone", metric: "noise alone (relative std, inverted)", x: (r) => Math.log10(r.noise), axis: "← quieter · log₁₀ checkpoint noise (relative), reversed", reverse: true },
    { value: "signal", label: "signal alone", metric: "signal alone (relative std)", x: (r) => Math.log10(r.signal), axis: "log₁₀ signal (relative sd across design variants) →" },
  ];
  const KIND_POP = { pooled: ["accuracy", "bBPB"], accuracy: ["accuracy"], bBPB: ["bBPB"], BPB: ["BPB"] };

  S.VIEWS.surrogatesScatter = async (el) => {
    const [d, grid] = await S.languages().then(() => Promise.all([load("surrogates_points"), load("surrogates_scoring")]));
    const all = S.expand(d).map((r) => ({ ...r, task: d.tasks[r.i].t, fam: d.tasks[r.i].f, lang: d.tasks[r.i].l, kind: d.tasks[r.i].k }));
    const name = (p) => `${d.names[p.fam] ?? p.fam} · ${S.langName(p.lang)}`;
    mount(el, [
      { key: "proxy", label: "Proxy size", type: "seg", options: PROX, value: "350M" },
      { key: "stat", label: "Statistic on the x axis", options: STATS.map(({ value, label }) => ({ value, label })), string: true },
      { key: "sc", label: "Tasks", type: "seg", options: SCORINGS },
      { key: "q", label: "Find a benchmark or language", type: "text" },
    ], (s) => {
      const st = STATS.find((o) => o.value === s.stat);
      const keep = KIND_POP[s.sc];
      const pts = all.filter((r) => r.proxy === s.proxy && keep.includes(r.kind))
        .map((r) => ({ ...r, x: st.x(r) })).filter((r) => r.x != null && Number.isFinite(r.x) && r.da != null);
      const t = (s.q ?? "").trim().toLowerCase();
      const hit = (p) => t && [p.task, p.fam, d.names[p.fam] ?? "", p.lang, S.langName(p.lang)].some((v) => String(v).toLowerCase().includes(t));
      const hits = pts.filter(hit);
      // the median DA-size of ten equal-count groups of tasks along x (no group of a handful of outliers)
      const sorted = [...pts].sort((p, q) => p.x - q.x);
      const trend = d3.range(10).map((g) => sorted.slice(Math.floor(g * sorted.length / 10), Math.floor((g + 1) * sorted.length / 10)))
        .filter((g) => g.length).map((g) => ({ x: d3.median(g, (p) => p.x), y: d3.median(g, (p) => p.da) }));
      const ref = grid.find((r) => r.scoring === s.sc && r.proxy === s.proxy && r.metric === st.metric);
      const col = scoringColor();
      const kinds = ["accuracy", "bBPB", "BPB"].filter((k) => keep.includes(k));
      if (!pts.length) return h("div", { class: "viz-empty" }, "This statistic has no value for these tasks at this proxy (the scaling-fit R² starts at 350M; the margin exists only for accuracy).");
      return [
        tiles([[pts.length, "tasks (above chance here and at 1.7B)"], [ref ? fmt(ref.rho) : "–", `Spearman ρ with DA-size${ref ? ` (n = ${ref.n})` : ""}`],
          [t ? hits.length : "–", "match the search"]]),
        plot({
          height: 400, marginBottom: 42,
          x: { label: st.axis, reverse: !!st.reverse },
          y: { label: "↑ DA-size: share of design pairs ranked like 1.7B", domain: [0, 1], grid: true, tickFormat: "%" },
          color: { domain: kinds, range: kinds.map((k) => col[k]), legend: kinds.length > 1 },
          marks: [
            Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4,3" }),
            Plot.dot(pts, { x: "x", y: "da", fill: "kind", r: 2.6, fillOpacity: t ? 0.15 : 0.45 }),
            Plot.line(trend, { x: "x", y: "y", stroke: css("--md-default-fg-color"), strokeWidth: 2.5 }),
            Plot.dot(trend, { x: "x", y: "y", fill: css("--md-default-fg-color"), r: 3.5 }),
            Plot.dot(hits, { x: "x", y: "da", r: 5, fill: "kind", stroke: css("--md-default-fg-color"), strokeWidth: 1.5 }),
            Plot.tip(pts, Plot.pointer({ x: "x", y: "da",
              title: (p) => `${name(p)} (${p.task})\n${st.label}: ${fmt(st.value === "early" ? p.x : Math.pow(10, p.x), st.value === "early" ? 2 : 4)}\nDA-size ${pct(p.da)} · SNR ${fmt(p.snr, 1)}` })),
          ],
        }),
        note("Each dot is one task (a benchmark in one language) at this proxy. The bold line is the median DA-size of ten equal-size groups of tasks along x: if it climbs from left to right, the statistic tells reliable tasks from unreliable ones. Try signal alone: the line stays flat. The ρ tile is the analysis' own number for this population."),
      ];
    });
  };

  // ---------- the 22 SNR definitions: two DA kinds side by side ----------
  const FAMILY = { rel_spread: "relative spread", dispersion: "dispersion", discrepancy: "discrepancy", robust: "robust", depth: "depth" };

  S.VIEWS.surrogatesVariants = async (el) => {
    const d = await load("surrogates_variants");
    const order = d.variants.map((v) => v.variant);
    const fam = Object.fromEntries(d.variants.map((v) => [v.variant, v.family]));
    const pts = d.variants.flatMap((v) => [
      { ...v, kind: "DA-size (proxy → 1.7B)", r: v.da_size, n: v.n_size },
      { ...v, kind: "DA-ckpt (early → final)", r: v.da_ckpt, n: v.n_ckpt },
    ]).filter((p) => p.r != null);
    const top = (k) => Object.entries(d.wins[k] ?? {}).sort((a, b) => b[1] - a[1])[0] ?? ["–", 0];
    mount(el, [], () => {
      const [fs, ns] = top("da_size"), [fc, nc] = top("da_ckpt");
      return [
        tiles([[`${ns} of ${d.n_languages}`, `languages whose best definition for DA-size is ${FAMILY[fs] ?? fs}`],
          [`${nc} of ${d.n_languages}`, `languages whose best definition for DA-ckpt is ${FAMILY[fc] ?? fc}`]]),
        plot({
          height: 22 * order.length + 70, marginLeft: 230,
          x: { label: "mean Pearson r of log₁₀ SNR with decision accuracy, over the languages", grid: true, domain: [-0.2, 0.7] },
          y: { domain: order, label: null, tickFormat: (v) => `${v} · ${FAMILY[fam[v]] ?? fam[v]}` },
          color: { domain: ["DA-size (proxy → 1.7B)", "DA-ckpt (early → final)"], range: [css("--viz-s1"), css("--viz-s2")], legend: true },
          marks: [
            Plot.ruleX([0], { stroke: css("--viz-muted") }),
            Plot.link(d.variants, { x1: "da_size", x2: "da_ckpt", y1: "variant", y2: "variant", stroke: css("--viz-grid"), strokeWidth: 4 }),
            Plot.dot(pts, { x: "r", y: "variant", fill: "kind", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (p) => `${p.variant} (${FAMILY[p.family] ?? p.family})\n${p.kind}: mean r ${fmt(p.r)} over ${p.n} languages` }),
          ],
        }),
        note("Each row is one SNR definition; the dots are how well it tracks the two kinds of decision accuracy, averaged over the languages. The relative-spread definitions sit on top as a block, a hair apart: recommend the family, not one formula. The depth definitions are no better than zero."),
      ];
    });
  };
})();

/* Additional interactive views of the surrogates page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, css, fmt, pct, note, tiles, plot, mount, load, uniq } = S;

  const SCORINGS = [
    { value: "pooled", label: "accuracy + bBPB (Figure 4)" },
    { value: "accuracy", label: "benchmark accuracy" },
    { value: "bBPB", label: "benchmark bBPB" },
    { value: "BPB", label: "per-language BPB" },
  ];
  const scoringColor = () => ({ pooled: css("--md-default-fg-color"), accuracy: css("--viz-s1"), bBPB: css("--viz-s3"), BPB: css("--viz-s2") });
  const PROX = ["90M", "175M", "350M", "600M", "1B"];
  const short = (m) => m.replace(" (proxy rungs only)", "").replace(" (relative std, inverted)", ", inverted").replace(" (relative std)", "");
  const row = (...nodes) => h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;justify-content:center" }, ...nodes);
  const proxyColors = () => d3.quantize(d3.interpolateRgb(S.sizeRamp()[0], S.sizeRamp().at(-1)), PROX.length);

  // ---------- surrogate x proxy size, by scoring (paper Figure 20; the pooled grid is Figure 4) ----------
  S.VIEWS.surrogatesScoring = async (el) => {
    const d = await load("surrogates_scoring");
    const pooled = d.filter((r) => r.scoring === "pooled");
    const order = d3.groupSort(pooled, (g) => -d3.mean(g, (r) => r.rho), (r) => r.metric)
      .concat(uniq(d.map((r) => r.metric)).filter((m) => !pooled.some((r) => r.metric === m)));
    const PANELS = SCORINGS.slice(1);
    mount(el, [
      { key: "show", label: "Show", type: "seg", options: [{ value: "panels", label: "three scorings side by side (paper)" },
        { value: "grid", label: "one scoring as a grid" }, { value: "lines", label: "one panel per statistic" }] },
      { key: "sc", label: "Scoring (grid)", type: "seg", options: SCORINGS },
    ], (s) => {
      if (s.show === "panels") {
        const rows = d.filter((r) => r.scoring !== "pooled").map((r) => ({ ...r, lab: SCORINGS.find((o) => o.value === r.scoring).label }));
        const metrics = order.filter((m) => rows.some((r) => r.metric === m));
        const col = proxyColors();
        const byProxy = (i) => rows.filter((r) => r.proxy === PROX[i]).map((r) => ({ ...r, i }));
        return [row(plot({
          width: 690, height: 46 * metrics.length + 80, marginLeft: 196, marginRight: 24, marginBottom: 40,
          fx: { domain: PANELS.map((o) => o.label), label: null, axis: "top", padding: 0.06 },
          x: { domain: [-0.7, 0.8], label: "Spearman ρ with DA-size (proxy vs 1.7B) →", grid: true, ticks: [-0.5, 0, 0.5] },
          y: { domain: metrics, label: null },
          color: { domain: PROX.map((p) => `${p} proxy`), range: col, legend: true },
          marks: [
            Plot.ruleX([0], { stroke: css("--viz-muted") }),
            ...PROX.flatMap((p, i) => [
              Plot.dot(byProxy(i), { fx: "lab", x: "rho", y: "metric", dy: (i - 2) * 7, r: 3, fill: () => `${p} proxy`, tip: true,
                title: (r) => `${r.metric}, ${r.lab}, ${r.proxy} proxy\nρ = ${fmt(r.rho)} over n = ${r.n} tasks${r.p < 0.05 ? " (p < 0.05)" : ""}` }),
              Plot.text(byProxy(i), { fx: "lab", x: "rho", y: "metric", dy: (i - 2) * 7, dx: 5, text: (r) => `n=${r.n}`, fontSize: 7, textAnchor: "start", fill: css("--viz-muted") }),
            ]),
          ],
        })), note("Spearman ρ between each surrogate and DA-size against 1.7B at each proxy size, on the benchmark tasks above chance scored by accuracy (left), their bBPB versions (middle) and per-language BPB (right); n is the number of tasks. " +
          "bBPB raises the level of the correlations of Figure 4: early-checkpoint agreement reads " +
          `${rangeOf(d, "bBPB", "early-checkpoint agreement (10 %)")} on bBPB against ${rangeOf(d, "accuracy", "early-checkpoint agreement (10 %)")} on accuracy. ` +
          "The discrepancy and the margin above chance exist for accuracy only, the scaling-fit R² needs three sizes and starts at 350M.")];
      }
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
        return [row(legend),
          h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:.6em 1em;margin-top:.6em" }, panels),
          note("One panel per statistic: its Spearman ρ with DA-size at each proxy, one line per way of scoring the tasks. The pooled line (accuracy and bBPB, Figure 4) runs with the green bBPB line, above the blue accuracy line. On per-language BPB (orange), five of the six statistics turn negative at 600M.")];
      }
      const rows = d.filter((r) => r.scoring === s.sc);
      const metrics = order.filter((m) => rows.some((r) => r.metric === m));
      const lead = PROX.map((p) => rows.filter((r) => r.proxy === p).sort((a, b) => b.rho - a.rho)[0]).filter(Boolean);
      return [row(plot({
        width: 760, height: 40 * metrics.length + 70, marginLeft: 250,
        x: { domain: PROX, label: "Proxy model size", axis: "top" },
        y: { domain: metrics, label: null },
        color: { ...S.diverging(), domain: [-0.8, 0.8], legend: true, label: "Spearman ρ with DA-size" },
        marks: [
          Plot.cell(rows, { x: "proxy", y: "metric", fill: "rho", inset: 1, rx: 3, tip: true,
            title: (r) => `${r.metric} at ${r.proxy}\nρ = ${fmt(r.rho)} (p = ${r.p.toPrecision(2)}, n = ${r.n} tasks)` }),
          Plot.text(rows, { x: "proxy", y: "metric", text: (r) => `${fmt(r.rho)}\n(n=${r.n})`, fontSize: 10,
            fill: (r) => (Math.abs(r.rho) > 0.5 ? "white" : css("--md-default-fg-color")) }),
          Plot.text(lead, { x: "proxy", y: "metric", text: () => "▲", dx: -24, dy: -6, fontSize: 8 }),
        ],
      })), note(`Leader per proxy (▲): ${lead.map((r) => `${r.proxy} ${short(r.metric)} ${fmt(r.rho)}`).join(" · ")}. ` +
        "Rows keep the order of the pooled grid (Figure 4, strongest on top), so switching the scoring shows which statistics hold up and which fall.")];
    });
  };
  const rangeOf = (d, sc, m) => {
    const v = d.filter((r) => r.scoring === sc && r.metric === m).map((r) => r.rho);
    return `${fmt(d3.min(v))}–${fmt(d3.max(v))}`;
  };

  // ---------- one dot per task: a statistic against DA-size ----------
  const STATS = [
    { value: "snr", label: "SNR (relative std)", metric: "SNR, relative std", x: (r) => Math.log10(r.snr), axis: "log₁₀ SNR (relative std) →" },
    { value: "early", label: "early-checkpoint agreement", metric: "early-checkpoint agreement (10 %)", x: (r) => r.early, axis: "agreement of the 10 % checkpoint with the proxy's final ranking →" },
    { value: "noise", label: "noise alone", metric: "noise alone (relative std, inverted)", x: (r) => Math.log10(r.noise), axis: "← quieter · log₁₀ checkpoint noise (relative), reversed", reverse: true },
    { value: "signal", label: "signal alone", metric: "signal alone (relative std)", x: (r) => Math.log10(r.signal), axis: "log₁₀ signal (relative std across variants) →" },
  ];
  const KIND_POP = { pooled: ["accuracy", "bBPB"], accuracy: ["accuracy"], bBPB: ["bBPB"], BPB: ["BPB"] };

  S.VIEWS.surrogatesScatter = async (el) => {
    const [d, grid] = await S.languages().then(() => Promise.all([load("surrogates_points"), load("surrogates_scoring")]));
    const all = S.expand(d).map((r) => ({ ...r, task: d.tasks[r.i].t, fam: d.tasks[r.i].f, lang: d.tasks[r.i].l, kind: d.tasks[r.i].k }));
    const name = (p) => `${d.names[p.fam] ?? p.fam} · ${S.langName(p.lang)}`;
    mount(el, [
      { key: "proxy", label: "Proxy size", type: "seg", options: PROX, value: "350M" },
      { key: "stat", label: "Statistic on the x axis", options: STATS.map(({ value, label }) => ({ value, label })), string: true },
      { key: "sc", label: "Tasks", type: "seg", options: SCORINGS, value: "pooled" },
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
        row(plot({
          width: 760, height: 400, marginBottom: 42,
          x: { label: st.axis, reverse: !!st.reverse },
          y: { label: "↑ DA-size: share of pairs of variants ranked like 1.7B", domain: [0, 1], grid: true, tickFormat: "%" },
          color: { domain: kinds, range: kinds.map((k) => col[k]), legend: kinds.length > 1 },
          marks: [
            Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4,3" }),
            Plot.dot(pts, { x: "x", y: "da", fill: "kind", r: 2.6, fillOpacity: t ? 0.15 : 0.45 }),
            Plot.line(trend, { x: "x", y: "y", stroke: css("--md-default-fg-color"), strokeWidth: 2.5 }),
            Plot.dot(trend, { x: "x", y: "y", fill: css("--md-default-fg-color"), r: 3.5 }),
            Plot.dot(hits, { x: "x", y: "da", r: 5, fill: "kind", stroke: css("--md-default-fg-color"), strokeWidth: 1.5 }),
            Plot.tip(pts, Plot.pointer({ x: "x", y: "da",
              title: (p) => `${name(p)}\n${st.label}: ${fmt(st.value === "early" ? p.x : Math.pow(10, p.x), st.value === "early" ? 2 : 4)}\nDA-size ${pct(p.da)} · SNR ${fmt(p.snr, 1)}` })),
          ],
        })),
        note("Each dot is one task (a benchmark in one language) at this proxy. The bold line is the median DA-size of ten equal-size groups of tasks along x: if it climbs from left to right, the statistic tells reliable tasks from unreliable ones. Try signal alone: the line stays flat. The ρ tile is the analysis' own number for this population."),
      ];
    });
  };

  // ---------- the 22 SNR definitions: two DA kinds side by side (paper Figure 19) ----------
  S.VIEWS.surrogatesVariants = async (el) => {
    const d = await load("surrogates_variants");
    const order = d.variants.map((v) => v.variant);
    const KINDS = ["DA-size", "DA-ckpt"];
    const pts = d.variants.flatMap((v) => [
      { ...v, kind: KINDS[0], r: v.da_size, n: v.n_size },
      { ...v, kind: KINDS[1], r: v.da_ckpt, n: v.n_ckpt },
    ]).filter((p) => p.r != null);
    mount(el, [], () => [
      row(plot({
        width: 640, height: 22 * order.length + 80, marginLeft: 170,
        x: { label: "Mean Pearson r across languages (log₁₀ SNR ↔ DA) →", grid: true, domain: [-0.2, 0.6] },
        y: { domain: order, label: null },
        color: { domain: KINDS, range: [css("--viz-s1"), css("--viz-s2")], legend: true },
        marks: [
          Plot.ruleX([0], { stroke: css("--viz-muted") }),
          Plot.link(pts, { x1: 0, x2: "r", y1: "variant", y2: "variant", stroke: "kind", strokeOpacity: 0.5, strokeWidth: 1.5 }),
          Plot.dot(pts, { x: "r", y: "variant", fill: "kind", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (p) => `${p.variant}\n${p.kind}: mean r ${fmt(p.r)} over ${p.n} languages` }),
        ],
      })),
      note(`For each of the ${order.length} SNR definitions, the mean over languages of the Pearson r between log₁₀ SNR and decision accuracy across the tasks of one language, for DA-size (blue) and DA-ckpt (orange). ` +
        `The relative-spread definitions lead as a block (${order.slice(0, 3).map((v) => `${v} ${fmt(d.variants.find((x) => x.variant === v).da_size)}`).join(", ")} on DA-size); the depth-based ones (tukey, projection) fall below zero.`),
    ]);
  };

  // ---------- within one benchmark: SNR against DA-size (Section 6, "up to ρ = 0.93 for HellaSwag") ----------
  const NOISE = { ckpt_rel: "checkpoint noise, relative", ckpt_abs: "checkpoint noise", kfold_rel: "benchmark noise, relative",
    kfold_abs: "benchmark noise", tukey_depth: "Tukey depth", projection_depth: "projection depth" };
  const defName = ([sig, noise]) => `${sig} / ${NOISE[noise] ?? noise}`;

  S.VIEWS.surrogatesBench = async (el) => {
    const d = await load("surrogates_bench");
    const KINDS = ["SNR, relative std", "strongest SNR definition"];
    mount(el, [
      { key: "k", label: "Benchmarks", type: "seg", options: [{ value: "all", label: "both scorings (paper)" }, { value: "accuracy", label: "accuracy" }, { value: "bBPB", label: "bBPB" }] },
    ], (s) => {
      const rows = d.filter((r) => r.all || s.k === "all" || r.k === s.k);
      const order = rows.map((r) => r.b);
      const pts = rows.flatMap((r) => [{ ...r, kind: KINDS[0], rho: r.ref }, { ...r, kind: KINDS[1], rho: r.best }]).filter((p) => p.rho != null);
      const top = rows.find((r) => !r.all);
      return [
        row(plot({
          width: 760, height: 20 * order.length + 80, marginLeft: 210, marginRight: 60,
          x: { domain: [-0.2, 1], label: "Spearman ρ with DA-size within the benchmark →", grid: true },
          y: { domain: order, label: null, tickFormat: (b) => (b === "every benchmark" ? "every benchmark (pooled)" : b) },
          color: { domain: KINDS, range: [css("--viz-s1"), css("--viz-s2")], legend: true },
          marks: [
            Plot.ruleX([0], { stroke: css("--viz-muted") }),
            Plot.link(rows, { x1: "ref", x2: "best", y1: "b", y2: "b", stroke: css("--viz-grid"), strokeWidth: 3 }),
            Plot.dot(pts, { x: "rho", y: "b", fill: "kind", r: 4.5, stroke: css("--viz-surface"), strokeWidth: 1, tip: true,
              title: (p) => `${p.b}, ${p.n} languages\nSNR, relative std: ρ = ${fmt(p.ref)}\nstrongest definition (${defName(p.best_def)}): ρ = ${fmt(p.best)}` }),
            Plot.text(rows, { x: "best", y: "b", text: (r) => `n=${r.n}`, dx: 8, textAnchor: "start", fontSize: 8, fill: css("--viz-muted") }),
          ],
        })),
        note(`Spearman ρ between an SNR and DA-size against 1.7B within one benchmark: one point per language (a benchmark task and its RF versions), proxies 90M–1B pooled, every pair of variants; benchmarks with at least 10 languages, n next to each row. ` +
          `Blue is the relative-standard-deviation SNR (over the relative checkpoint noise), orange the strongest of the 22 × 6 signal × noise definitions, a best-of-many value that is optimistic on a small benchmark. ` +
          (top ? `The strongest is ${top.b} at ρ = ${fmt(top.best)} (${defName(top.best_def)}, ${top.n} languages; relative-std SNR ${fmt(top.ref)}).` : "")),
      ];
    });
  };
})();

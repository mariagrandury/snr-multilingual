/* The interactive versions of the paper's four main-text figures, on the Findings
 * pages (registered on SNRSITE.VIEWS; data: scripts/site_data/paper.py). */
(() => {
  const S = window.SNRSITE;
  const { h, css, fmt, pct, note, plot, mount, load, uniq, series } = S;
  const SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"];
  const sizeColor = () => {
    const r = S.sizeRamp();
    return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), SIZES.length);
  };
  const row = (...nodes) => h("div", { style: "display:flex;flex-wrap:wrap;gap:8px" }, ...nodes);

  // ---------- Figure 1: above chance against the training tokens of the language ----------
  S.VIEWS.paperRq0 = async (el) => {
    const d = await load("paper_rq0");
    const benches = uniq(d.rows.map((r) => r.b)).sort((a, b) => d.names[a].localeCompare(d.names[b]));
    mount(el, [{ key: "b", label: "Benchmark", string: true, value: "include_base_44",
      options: benches.map((b) => ({ value: b, label: d.names[b] })) },
    { key: "llm", label: "Formats", type: "seg", options: [{ value: false, label: "original and RF (paper)" }, { value: true, label: "also LLM-RF" }] }], (s) => {
      const rows = d.rows.filter((r) => r.b === s.b && r.tokens > 0);
      const formats = ["Original", "RF", ...(s.llm ? ["LLM-RF"] : [])].filter((f) => rows.some((r) => r.f === f));
      return [row(...formats.map((f) => plot({
        width: formats.length > 1 ? 360 : 560, height: 260, marginLeft: 45,
        title: f === "Original" ? "Original format" : f,
        x: { type: "log", label: "Training tokens of the language", grid: true },
        y: { domain: [0, 1], label: "Share above chance", grid: true, tickFormat: "%" },
        color: { domain: SIZES, range: sizeColor(), legend: f === formats[0], label: "Model size" },
        marks: [
          Plot.line(rows.filter((r) => r.f === f), { x: "tokens", y: "share", stroke: "size", strokeWidth: 1.8, sort: "tokens" }),
          Plot.dot(rows.filter((r) => r.f === f), { x: "tokens", y: "share", fill: "size", r: 2.5, tip: true,
            title: (r) => `${r.size}, ${fmt(r.tokens / 1e6, 1)}M tokens of the language\n${pct(r.share)} of ${r.n} evaluations above chance` }),
        ],
      }))), note("Each marker: the share of (language, K, checkpoint) evaluations of the benchmark above chance, binned by the tokens of the task's language seen by the checkpoint; one line per model size. RF scores the answer text instead of the letter, LLM-RF rewrites each item as a cloze statement.")];
    });
  };

  // ---------- Figure 2: benchmark scaling predictability ----------
  S.VIEWS.paperRq1 = async (el) => {
    const all = await load("paper_rq1");
    const regimes = uniq(all.map((r) => r.regime)).sort();
    // the legend lists the regimes on screen; a regime keeps its colour when the filter changes
    const colorOf = (d) => {
      const shown = regimes.filter((g) => d.some((r) => r.regime === g));
      return { domain: shown, range: shown.map((g) => series(regimes.length)[regimes.indexOf(g)]), legend: true, label: "Regime" };
    };
    const PANELS = ["bpb", "hellaswag", "xnli", "multiblimp", "xwinograd", "xcopa"];   // the paper's panel (b)
    const name = (f) => all.find((r) => r.family === f).name;
    const tip = (r) => `${r.name}, ${r.language}\nsize R² ${fmt(r.r2_size)}, ρ ${fmt(r.rho_size)}, training R² ${fmt(r.r2_train)}\n${r.regime}`;
    const axes = { x: { domain: [0, 1], label: "Median R² across model-size fits", grid: true } };
    const ref = [Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }), Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" })];
    const fams = uniq(all.map((r) => r.family)).sort((a, b) => name(a).localeCompare(name(b)));
    mount(el, [
      { key: "fam", label: "Highlight", string: true, options: [{ value: "all", label: "every benchmark" }, ...fams.map((f) => ({ value: f, label: name(f) }))] },
      { key: "scores", label: "Tasks", type: "seg", options: [{ value: "paper", label: "accuracy and BPB (paper)" }, { value: "all", label: "also bBPB" }] },
      { key: "a", label: "Panel (a)", type: "seg", options: [{ value: "bench", label: "one dot per benchmark (paper)" }, { value: "task", label: "one dot per task" }] },
      { key: "b", label: "Panel (b)", type: "seg", options: [{ value: "paper", label: "six benchmarks (paper)" }, { value: "all", label: "every benchmark" }] },
    ], (s) => {
      const d = s.scores === "paper" ? all.filter((r) => !r.family.startsWith("bbpb_")) : all;
      const color = colorOf(d);
      const on = (r) => s.fam === "all" || r.family === s.fam;
      // the highlighted benchmark in colour, the others in grey
      const dots = (data, y, opts) => [Plot.dot(data.filter((r) => !on(r)), { x: "r2_size", y, r: 2.5, ...opts, fill: css("--viz-grid") }),
        Plot.dot(data.filter(on), { x: "r2_size", y, r: 3, fill: "regime", fillOpacity: 0.75, tip: true, title: tip, ...opts })];
      const bench = d3.groups(d, (r) => r.family).map(([f, rs]) => ({
        family: f, name: name(f), n: rs.length, r2_size: d3.median(rs, (r) => r.r2_size), rho_size: d3.median(rs, (r) => r.rho_size),
        regime: d3.greatest(d3.rollups(rs, (v) => v.length, (r) => r.regime), (g) => g[1])[0] }));
      const a = s.a === "bench"
        ? dots(bench, "rho_size", { r: "n", title: (b) => `${b.name} (${b.n} task${b.n > 1 ? "s" : ""})\nmedian size R² ${fmt(b.r2_size)}, median ρ ${fmt(b.rho_size)}` })
        : dots(d, "rho_size");
      const trainY = { domain: [0, 1], label: "Median R² across training fits", grid: true };
      const b = s.b === "paper"
        ? plot({ width: 720, height: 420, color: { ...color, legend: false }, ...axes, y: trainY,
          fx: { axis: null }, fy: { axis: null },
          marks: [...ref, Plot.text(d.filter((r) => PANELS.includes(r.family)), { x: "r2_size", y: "r2_train", text: "language", fill: "regime",
            fx: (r) => PANELS.indexOf(r.family) % 3, fy: (r) => Math.floor(PANELS.indexOf(r.family) / 3), tip: true, title: tip }),
          Plot.text(PANELS, { fx: (f) => PANELS.indexOf(f) % 3, fy: (f) => Math.floor(PANELS.indexOf(f) / 3), frameAnchor: "top-left", dx: 4, dy: 4, fontWeight: "bold",
            text: (f) => `${name(f)} (${d.filter((r) => r.family === f).length} tasks)` })],
          title: "(b) Across size and along training" })
        : plot({ width: 360, height: 330, title: "(b) Across size and along training", color: { ...color, legend: false }, ...axes, y: trainY,
          marks: [...ref, ...dots(d, "r2_train")] });
      return [row(
        plot({ width: 360, height: 330, title: "(a) Fit quality and direction across size", color, ...axes,
          y: { domain: [-1, 1], label: "Median Spearman ρ with model size", grid: true }, r: { range: [3, 14] },
          marks: [Plot.ruleY([0], { stroke: css("--viz-muted") }), ...a] }), b),
      note(`${d.length} benchmark-language tasks above chance${s.scores === "paper" ? " (accuracy and BPB, as in the paper)" : " (accuracy, BPB and bBPB)"}. (a) ${s.a === "bench" ? "One dot per benchmark at the median over its tasks, sized by its number of tasks" : "One dot per task"}: the R² and Spearman ρ of the log-linear fits across model size. (b) ${s.b === "paper" ? "The paper's six benchmarks, one label per task (its language)" : "One dot per task"}: the same R² against the median R² of the fits along training. Hover a mark for its benchmark and language.`)];
    });
  };

  // ---------- Figure 3: decision reliability across scale and training ----------
  S.VIEWS.paperRq2 = async (el) => {
    const d = await load("paper_rq2");
    const groups = uniq(d.size.map((r) => r.group)).filter((g) => g !== "all pairs");
    const greens = d3.quantize(d3.interpolateRgb("#0b5d2b", "#74c476"), groups.length);
    const gColor = { domain: ["all pairs", ...groups], range: [css("--md-default-fg-color"), ...greens] };
    const label = (g) => (g === "all pairs" ? "all pairs" : g.replace(/ \(.*\)/, ""));
    mount(el, [], () => {
      const size = d.size.filter((r) => r.da != null);
      const real = size.filter((r) => r.size !== "1.7B");
      const ref = size.filter((r) => r.size === "1.7B");
      const link = groups.concat("all pairs").flatMap((g) => {
        const last = real.filter((r) => r.group === g).at(-1), end = ref.find((r) => r.group === g);
        return last && end ? [{ ...last, seg: g }, { ...end, seg: g }] : [];
      });
      const panel = (name, ylab) => {
        const rows = d.run.filter((r) => r.panel === name);
        return plot({ width: 260, height: 300, marginLeft: 40,
          x: { domain: [0.3, 5.2], ticks: [1, 2, 3, 4, 5], tickFormat: (v) => `${v}C`, label: "Proxy training tokens" },
          y: { domain: [0.3, 1.02], label: ylab, grid: true },
          color: { domain: SIZES, range: sizeColor(), legend: name === "DA-ckpt", label: "Proxy size" },
          marks: [Plot.line(rows, { x: "x", y: "da", stroke: "size", strokeWidth: 1.8, sort: "x" }),
            Plot.dot(rows, { x: "x", y: "da", fill: "size", r: 2.5, tip: true,
              title: (r) => `${r.size} at ${r.x}C\n${name} ${fmt(r.da)} over ${r.tasks} tasks` })] });
      };
      return [row(
        plot({ width: 300, height: 300, marginLeft: 40, color: { ...gColor, legend: true, tickFormat: label },
          x: { type: "point", domain: SIZES, label: "Model size" },
          y: { domain: [0.3, 1.02], label: "DA-size", grid: true },
          marks: [Plot.line(real, { x: "size", y: "da", stroke: "group", z: "group", strokeWidth: (r) => (r.group === "all pairs" ? 2.5 : 1.5) }),
            Plot.line(link, { x: "size", y: "da", z: "seg", stroke: "group", strokeDasharray: "2 3", strokeOpacity: 0.5 }),
            Plot.dot(real, { x: "size", y: "da", fill: "group", r: 2.8, tip: true,
              title: (r) => `${label(r.group)} at ${r.size}\nDA-size ${fmt(r.da)} over ${r.tasks} tasks${r.pairs ? `, ${r.pairs} pairs` : ""}` }),
            Plot.dot(ref.slice(0, 1), { x: "size", y: "da", r: 3.5, stroke: css("--md-default-fg-color"), fill: css("--md-default-bg-color") })] }),
        panel("DA-ckpt", "DA-ckpt"), panel("DA-target", "DA-target"),
      ), note("(a) Scale agreement: DA of the final checkpoint of each size against the final 1.7B ranking, one line per design axis and the black line over all pairs. (b) Within-model training agreement: DA of each checkpoint against the final checkpoint of the same size. (c) Target agreement: DA of each checkpoint against the final 1.7B checkpoint. The hollow marker is 1 by construction (a ranking compared with itself). Single-axis decisions, over the tasks whose median DA-size or DA-ckpt is at least 0.66.")];
    });
  };

  // ---------- Figure 4: surrogate predictivity across proxy sizes ----------
  S.VIEWS.paperRq3 = async (el) => {
    const d = await load("paper_rq3");
    const order = d3.groupSort(d, (g) => -d3.mean(g, (r) => r.rho), (r) => r.metric);
    const proxies = SIZES.slice(0, 5);
    mount(el, [], () => [plot({
      width: 640, height: 40 * order.length + 60, marginLeft: 260,
      x: { domain: proxies, label: "Proxy model size", axis: "top" },
      y: { domain: order, label: null },
      color: { type: "diverging", scheme: "RdBu", domain: [-0.65, 0.65], legend: true, label: "Spearman ρ with DA-size" },
      marks: [
        Plot.cell(d, { x: "proxy", y: "metric", fill: "rho", inset: 1.5, rx: 3, tip: true,
          title: (r) => `${r.metric} at ${r.proxy}\nρ = ${fmt(r.rho)} over ${r.n} tasks` }),
        Plot.text(d, { x: "proxy", y: "metric", text: (r) => `${fmt(r.rho)}\n(n=${r.n})`, fontSize: 10,
          fill: (r) => (Math.abs(r.rho) > 0.45 ? "white" : css("--md-default-fg-color")) }),
      ],
    }), note("Spearman ρ between each surrogate statistic, computed at the proxy alone, and the proxy's DA-size against the 1.7B reference, across benchmark-language tasks at each proxy size; n is the number of tasks. The scaling-fit R² needs three model sizes, so it starts at 350M.")]);
  };
})();

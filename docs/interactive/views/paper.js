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
      options: benches.map((b) => ({ value: b, label: d.names[b] })) }], (s) => {
      const rows = d.rows.filter((r) => r.b === s.b && r.tokens > 0);
      const formats = ["Original", "RF", "LLM-RF"].filter((f) => rows.some((r) => r.f === f));
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
    const d = await load("paper_rq1");
    const fams = uniq(d.map((r) => r.family)).sort((a, b) => d.find((r) => r.family === a).name.localeCompare(d.find((r) => r.family === b).name));
    const regimes = uniq(d.map((r) => r.regime)).sort();
    mount(el, [{ key: "fam", label: "Highlight", string: true,
      options: [{ value: "all", label: "every benchmark" }, ...fams.map((f) => ({ value: f, label: d.find((r) => r.family === f).name }))] }], (s) => {
      const on = (r) => s.fam === "all" || r.family === s.fam;
      const color = { domain: regimes, range: series(regimes.length), legend: true, label: "Regime" };
      const dots = (y) => [
        Plot.dot(d.filter((r) => !on(r)), { x: "r2_size", y, r: 2.5, fill: css("--viz-grid") }),
        Plot.dot(d.filter(on), { x: "r2_size", y, r: 3, fill: "regime", fillOpacity: 0.75, tip: true,
          title: (r) => `${r.name}, ${r.language}\nsize R² ${fmt(r.r2_size)}, ρ ${fmt(r.rho_size)}, training R² ${fmt(r.r2_train)}\n${r.regime}` }),
      ];
      return [row(
        plot({ width: 360, height: 330, title: "(a) Fit quality and direction across size", color,
          x: { domain: [0, 1], label: "Median R² across model-size fits", grid: true },
          y: { domain: [-1, 1], label: "Median Spearman ρ with model size", grid: true },
          marks: [Plot.ruleY([0], { stroke: css("--viz-muted") }), ...dots("rho_size")] }),
        plot({ width: 360, height: 330, title: "(b) Across size and along training", color: { ...color, legend: false },
          x: { domain: [0, 1], label: "Median R² across model-size fits", grid: true },
          y: { domain: [0, 1], label: "Median R² across training fits", grid: true },
          marks: [Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
            Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }), ...dots("r2_train")] }),
      ), note("One dot per benchmark-language task above chance: (a) the median R² and Spearman ρ of its log-linear fits across model size, (b) the same R² against the median R² of its fits along training. Pick a benchmark to highlight its tasks.")];
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
        panel("DA-ckpt", "DA-ckpt"), panel("DA-goal", "DA-goal"),
      ), note("(a) Scale agreement: DA of the final checkpoint of each size against the final 1.7B ranking, one line per design axis and the black line over all pairs. (b) Within-model training agreement: DA of each checkpoint against the final checkpoint of the same size. (c) Full-scale agreement: DA of each checkpoint against the final 1.7B checkpoint. The hollow marker is 1 by construction (a ranking compared with itself). Single-axis decisions, over the tasks whose median DA-size or DA-ckpt is at least 0.66.")];
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

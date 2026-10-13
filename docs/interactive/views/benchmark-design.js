/* Additional interactive views of the benchmark-design page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, css, fmt, mount, note, uniq } = S;

  const CHARS = [
    { value: "domain", label: "Domain" },
    { value: "format_type", label: "Task format (continuation or lettered)" },
    { value: "format", label: "Task format (detailed)" },
    { value: "curation", label: "Curation (translated or native)" },
    { value: "curation_category", label: "Curation method (detailed)" },
    { value: "source", label: "Source (translated from English or not)" },
    { value: "n_options", label: "Answer options" },
    { value: "passage", label: "Reading passage" },
  ];
  const METRICS = [
    { value: "da_all", label: "DA-size, every task above chance" },
    { value: "da_rel", label: "DA-size, reliable tasks" },
    { value: "snr", label: "SNR at 1.7B" },
  ];
  const pretty = (v) => String(v).replaceAll("_", " ").replace("english", "English").replace(/^mt /, "MT ");
  const quantile = (xs, q) => {
    const s = [...xs].sort((a, b) => a - b), i = (s.length - 1) * q, lo = Math.floor(i);
    return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo);
  };
  // a stable jitter per task, so a redraw does not shuffle the dots
  const jitter = (key) => {
    let x = 0;
    for (const ch of key) x = (x * 31 + ch.charCodeAt(0)) >>> 0;
    return ((x % 1000) / 1000 - 0.5) * 0.6;
  };

  /* One dot per (benchmark, language) task, one row per level of the chosen
     characteristic: do the levels separate the tasks? */
  S.VIEWS.designStrip = async (el) => {
    await S.languages();
    const d = await S.load("benchmark_design_tasks");
    const fam = d.families;
    const benches = Object.keys(fam).sort((a, b) => fam[a].benchmark.localeCompare(fam[b].benchmark));
    mount(el, [
      { key: "char", label: "Characteristic", options: CHARS, string: true },
      { key: "metric", label: "Measure", type: "seg", options: METRICS },
      { key: "hi", label: "Highlight a benchmark", options: [{ value: "", label: "none" }, ...benches.map((b) => ({ value: b, label: fam[b].benchmark }))], string: true },
    ], (s) => {
      const isDA = s.metric !== "snr";
      const rows = d.tasks
        .filter((t) => t[s.metric] != null && fam[t.family]?.[s.char] != null)
        .map((t) => ({ ...t, v: t[s.metric], level: fam[t.family][s.char], name: fam[t.family].benchmark }));
      if (!rows.length) return h("div", { class: "viz-empty" }, "No task has a value for this reading.");
      const levels = uniq(rows.map((r) => r.level));
      const stats = levels.map((l) => {
        const xs = rows.filter((r) => r.level === l).map((r) => r.v);
        return { level: l, n: xs.length, nb: uniq(rows.filter((r) => r.level === l).map((r) => r.family)).length,
          med: quantile(xs, 0.5), q1: quantile(xs, 0.25), q3: quantile(xs, 0.75) };
      }).sort((a, b) => b.med - a.med);
      const idx = Object.fromEntries(stats.map((st, i) => [st.level, i]));
      const pts = rows.map((r) => ({ ...r, y: idx[r.level] + jitter(r.task) }));
      const hi = pts.filter((r) => r.family === s.hi);
      const ranked = [...stats].sort((a, b) => b.med - a.med);
      const spread = ranked.length > 1 ? ranked[0].med - ranked[ranked.length - 1].med : 0;
      const label = METRICS.find((m) => m.value === s.metric).label;
      return [
        plot({
          height: 56 * stats.length + 70, marginLeft: 210, marginRight: 20,
          x: isDA ? { label: `${label} (0.5 = chance agreement)`, grid: true, domain: [0, 1] }
            : { label: "SNR at 1.7B (log scale)", grid: true, type: "log" },
          y: { domain: [stats.length - 0.5, -0.5], ticks: stats.map((_, i) => i), label: null, tickSize: 0,
            tickFormat: (i) => { const st = stats[i]; return st ? `${pretty(st.level)} · ${st.n} tasks, ${st.nb} bench.` : ""; } },
          marks: [
            isDA ? Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }) : Plot.ruleX([1], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
            Plot.dot(pts, { x: "v", y: "y", r: 3, fill: css("--viz-s1"), fillOpacity: s.hi ? 0.18 : 0.45,
              tip: true, title: (r) => `${r.name} · ${S.langName(r.language)} (${r.language})\n${label}: ${fmt(r.v)}` +
                (isDA ? `\nmean over ${s.metric === "da_all" ? r.n_all : r.n_rel} proxies` : "") }),
            Plot.dot(hi, { x: "v", y: "y", r: 5, fill: css("--viz-s2"), stroke: css("--viz-surface"), strokeWidth: 1.5,
              tip: true, title: (r) => `${r.name} · ${S.langName(r.language)} (${r.language}): ${fmt(r.v)}` }),
            Plot.link(stats, { x1: "q1", x2: "q3", y1: (st) => idx[st.level], y2: (st) => idx[st.level], stroke: css("--md-default-fg-color"), strokeWidth: 2 }),
            Plot.link(stats, { x1: "med", x2: "med", y1: (st) => idx[st.level] - 0.34, y2: (st) => idx[st.level] + 0.34, stroke: css("--md-default-fg-color"), strokeWidth: 3 }),
            Plot.text(stats, { x: "med", y: (st) => idx[st.level] - 0.34, dy: -6, text: (st) => fmt(st.med), fontSize: 10 }),
          ],
        }),
        note(`Levels ordered by their median (thick tick; the horizontal bar spans the middle half of the tasks). ` +
          `The medians differ by ${fmt(spread)} from the highest to the lowest level; compare that gap with the spread of the dots inside each level. ` +
          `A benchmark whose sources do not settle this characteristic is left out of it.`),
      ];
    });
  };
})();

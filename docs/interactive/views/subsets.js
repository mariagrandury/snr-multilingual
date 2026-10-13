/* Additional interactive views of the subsets page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, css, fmt, pct, tiles, note, uniq } = S;
  const ORDER = ["90M", "175M", "350M", "600M", "1B", "1.7B"];

  // A subtask's label (`es`, `eng_Latn`, `easy`) with the language's name when it has one.
  const subName = (s) => {
    const code = s.split("_")[0];
    const name = S.langName(code);
    return name && name !== code ? `${name} (${s})` : s;
  };

  /* Cost against reliability: how the SNR of a benchmark moves as its
     languages are added best-first, against a random order, the full set and
     the random-subset null; the slider picks how many languages to keep. */
  S.VIEWS.subsetCurve = async (el) => {
    await S.languages();
    const d = await S.load("subsets_curves");
    const name = (t) => d.names[t] ?? t;
    const tasks = uniq(d.cells.map((c) => c.task)).sort((a, b) => name(a).localeCompare(name(b)));
    const cellOf = (t, z) => d.cells.find((c) => c.task === t && c.size === z);
    const st = { task: tasks.includes("hellaswag") ? "hellaswag" : tasks[0], size: null, k: null };

    const sel = h("select", { onchange: (e) => { st.task = e.target.value; st.size = null; st.k = null; render(); } },
      tasks.map((t) => h("option", { value: t }, name(t))));
    sel.value = st.task;
    const sizeBox = h("span", { class: "viz-seg" });
    const kOut = h("output");
    const slider = h("input", { type: "range", min: 1, max: 2, step: 1, value: 1,
      oninput: (e) => { st.k = +e.target.value; draw(); } });
    const bar = h("div", { class: "viz-controls" },
      h("label", {}, "Benchmark", sel), h("label", {}, "Model size", sizeBox),
      h("label", {}, h("span", {}, "Languages kept ", kOut), slider));
    const out = h("div");
    el.replaceChildren(bar, out);

    function render() {
      const sizes = ORDER.filter((z) => cellOf(st.task, z));
      if (!sizes.includes(st.size)) st.size = sizes[sizes.length - 1];
      sizeBox.replaceChildren(...sizes.map((z) => h("button", {
        type: "button", "aria-pressed": String(z === st.size),
        onclick: () => { st.size = z; st.k = null; render(); },
      }, z)));
      const c = cellOf(st.task, st.size);
      slider.max = c.n;
      if (st.k == null || st.k > c.n) st.k = c.best_n;
      slider.value = st.k;
      draw();
    }

    function draw() {
      const c = cellOf(st.task, st.size);
      const k = st.k;
      kOut.textContent = `${k} of ${c.n}`;
      const pts = c.cum.map((y, i) => ({ k: i + 1, y, lang: c.subs[i], own: c.own[i], kind: "best-first" }));
      const rnd = c.rand.map((y, i) => ({ k: i + 1, y, kind: "random order" }));
      const cur = pts[k - 1];
      const atSize = d.cells.filter((x) => x.size === st.size);
      const beat = atSize.filter((x) => (x.gain_null ?? 0) > 0).length;
      const nullTxt = c.null_p95 == null
        ? "no null: the full set is already the best prefix"
        : c.gain_null > 0 ? `the best ${c.best_n} beat random subsets of that size by ${fmt(c.gain_null)} SNR`
          : `the best ${c.best_n} do not beat random subsets of that size (95th percentile ${fmt(c.null_p95)})`;
      const yMax = Math.max(...c.cum, ...c.rand, c.null_p95 ?? 0, c.full) * 1.08;
      out.replaceChildren(
        tiles([
          [fmt(cur.y), `SNR with the best ${k} of ${c.n} languages`],
          [fmt(c.full), "SNR with all of them"],
          [pct(k / c.n), "of the languages to evaluate"],
          [c.gain_null == null ? "–" : (c.gain_null > 0 ? "+" : "") + fmt(c.gain_null), "SNR gain over the random-subset null"],
        ]),
        plot({
          height: 300, marginTop: 20,
          x: { label: "languages kept (best first)", domain: [0.5, c.n + 0.5], ticks: Math.min(c.n, 12), tickFormat: "d" },
          y: { label: "SNR of the kept languages", grid: true, domain: [0, yMax] },
          marks: [
            Plot.ruleY([c.full], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
            Plot.text([c.full], { x: c.n + 0.5, y: (v) => v, dy: -6, textAnchor: "end", text: () => "all languages", fill: css("--viz-muted") }),
            Plot.ruleX([k], { stroke: css("--viz-grid"), strokeWidth: 8, strokeOpacity: 0.8 }),
            Plot.line(rnd, { x: "k", y: "y", stroke: css("--viz-muted"), strokeWidth: 1.5, strokeDasharray: "2 3" }),
            Plot.line(pts, { x: "k", y: "y", stroke: css("--viz-s1"), strokeWidth: 2 }),
            Plot.dot([cur], { x: "k", y: "y", r: 7, fill: css("--viz-s1"), stroke: css("--viz-surface"), strokeWidth: 2 }),
            Plot.dot(pts, { x: "k", y: "y", r: 3.5, fill: css("--viz-s1"), stroke: css("--viz-surface"), strokeWidth: 1.5,
              tip: true, title: (p) => `best ${p.k} of ${c.n}: SNR ${fmt(p.y)}\nadds ${subName(p.lang)} (its own SNR ${fmt(p.own)})` }),
            c.null_p95 == null ? null : Plot.dot([c], { x: "best_n", y: "null_p95", symbol: "diamond", r: 6, fill: css("--viz-s2"),
              tip: true, title: () => `95th percentile of 100 random subsets of ${c.best_n} languages: ${fmt(c.null_p95)}` }),
            c.null_p95 == null ? null : Plot.text([c], { x: "best_n", y: "null_p95", dx: 9, textAnchor: "start", text: () => "random subsets, 95th pct.", fill: css("--viz-muted") }),
          ],
        }),
        h("p", { class: "viz-note", style: "line-height:1.9" },
          "Kept: ", ...c.subs.map((s, i) => h("span", {
            title: `own SNR ${fmt(c.own[i])}`,
            style: `display:inline-block;margin:0 .25em .2em 0;padding:0 .45em;border-radius:3px;${i < k
              ? `background:${css("--viz-s1")};color:#fff` : `border:1px solid ${css("--viz-grid")}`}`,
          }, subName(s)))),
        note(`Blue: the SNR of the best-first prefix, the languages ranked by their own SNR; grey dashed: one random order; orange diamond: ${nullTxt}. ` +
          `At ${st.size}, the chosen subset beats the null in ${beat} of the ${atSize.length} benchmarks swept.`),
      );
    }
    S.REDRAW.push(draw);
    render();
  };
})();

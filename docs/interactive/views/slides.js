/* Alternative interactive views, drafted for the slides (documents/slides_paper.md)
 * before they go to the site (registered on SNRSITE.VIEWS; see app.js).
 *
 *   ladderAxes             the model ladder filtered by one row of buttons per design axis;
 *                          data-panel="sizes" puts K on x and two size groups on y, with
 *                          the configuration written next to each square
 *   benchmarksConsidered   every benchmark considered, analysed and screened by task
 *                          (benchmarks_considered.json, scripts/site_data/benchmarks_considered.py)
 *   evalFormats            the answer formats and scoring rules on one example item (no data)
 */
(() => {
  const S = window.SNRSITE;
  const { h, note, tiles, uniq, series, BUILDS, LADDERS, GRID_ROWS, gridRow, gridLabel } = S;

  // ---------- the model ladder, one row of buttons per design axis ----------
  // each axis lists its levels baseline first
  const AXES = [
    { key: "M", label: "Language selection", order: ["A", "B", "C"], of: (c) => BUILDS[c.scheme].m, fmt: (v) => ({ A: "A: by data", B: "B: by diversity", C: "C" }[v]) },
    { key: "T", label: "Temperature", order: [1, 3], of: (c) => BUILDS[c.scheme].t, fmt: (v) => `T = ${v}` },
    { key: "arch", label: "Depth", order: ["deep", "shallow"], of: (c) => c.arch, fmt: (v) => v },
    { key: "activation", label: "Activation", order: ["xielu", "swiglu"], of: (c) => c.activation, fmt: (v) => ({ xielu: "xIELU", swiglu: "SwiGLU" }[v]) },
    { key: "optimizer", label: "Optimizer", order: ["ademamix", "muon"], of: (c) => c.optimizer, fmt: (v) => ({ ademamix: "AdEMAMix", muon: "Muon" }[v]) },
    { key: "seed", label: "Seed", order: ["1904", "replicates"], of: (c) => (c.seed === 1904 ? "1904" : "replicates"), fmt: (v) => (v === "1904" ? "1904" : "replicate seeds") },
  ];
  const color = (row) => series(12)[GRID_ROWS.indexOf(row) % 12];
  const square = (row, title) => h("span", { class: "ladder-chip", style: `--chip:${color(row)};box-sizing:border-box;flex:none`, title });

  S.VIEWS.ladderAxes = async (el) => {
    const d = await S.load("ladder");
    const bySizes = el.dataset.panel === "sizes";
    const on = Object.fromEntries(AXES.map((a) => [a.key, new Set(uniq(d.cells.map(a.of)))]));
    const out = h("div");
    // every level of every axis on or off at once
    const setAll = (v) => {
      for (const a of AXES) on[a.key] = new Set(v ? a.order : []);
      bar.querySelectorAll(".viz-seg button[data-level]").forEach((b) => b.setAttribute("aria-pressed", String(v)));
      render();
    };
    const bar = h("div", { class: "viz-controls" }, AXES.map((a) => {
      const btns = a.order.filter((v) => on[a.key].has(v)).map((v) => h("button", {
        type: "button", "aria-pressed": "true", "data-level": String(v),
        onclick: (e) => { on[a.key].has(v) ? on[a.key].delete(v) : on[a.key].add(v); e.currentTarget.setAttribute("aria-pressed", String(on[a.key].has(v))); render(); },
      }, a.fmt(v)));
      return h("label", {}, a.label, h("span", { class: "viz-seg" }, btns));
    }), h("label", {}, "All", h("span", { class: "viz-seg" },
      h("button", { type: "button", onclick: () => setAll(true) }, "select all"),
      h("button", { type: "button", onclick: () => setAll(false) }, "clear all"))));
    const render = () => {
      const keep = d.cells.filter((c) => AXES.every((a) => on[a.key].has(a.of(c))));
      const Ks = uniq(d.cells.map((c) => c.L)).sort((a, b) => a - b);
      out.replaceChildren(
        tiles([[keep.length, "runs"], [uniq(keep.map(gridRow)).length, "configurations"], [uniq(keep.map((c) => c.size)).length, "sizes"]]),
        bySizes ? bySizeGrid(keep, Ks) : byKGrid(keep, Ks),
        ...bySizes ? [] : [note("Columns: model size. Rows: number of training languages K. Each square is one run, coloured by its configuration. Use the buttons to add or remove the levels of each design axis.")],
      );
    };
    const byKGrid = (keep, Ks) => {
      const sizes = S.SIZES;
      const grid = h("div", { class: "ladder-grid", style: `grid-template-columns: 4em repeat(${sizes.length}, 1fr)` });
      for (const K of [...Ks].reverse()) {
        grid.append(h("div", { class: "rowhd" }, `K = ${K}`));
        for (const size of sizes) grid.append(h("div", { class: "cell" },
          keep.filter((c) => c.size === size && c.L === K).map((c) => square(gridRow(c), `${c.name}\n${gridLabel(gridRow(c))}`))));
      }
      grid.append(h("div", {}), ...sizes.map((z) => h("div", { class: "hd" }, z)));
      const shown = GRID_ROWS.filter((r) => keep.some((c) => gridRow(c) === r));
      return h("div", {}, h("div", { style: "margin-top:1em" }, grid),
        h("div", { class: "ladder-legend" }, shown.map((r) => h("span", {}, h("i", { style: `background:${color(r)};border-color:${color(r)}` }), gridLabel(r)))));
    };
    const bySizeGrid = (keep, Ks) => {
      const groups = [[S.LADDER.join("\n"), (c) => c.size !== "3B"], ["3B", (c) => c.size === "3B"]];
      const grid = h("div", { style: `display:grid;grid-template-columns:3.6em repeat(${Ks.length}, minmax(0, 1fr));gap:3px;margin-top:1em;font-size:.7rem` });
      for (const [label, inGroup] of groups) {
        grid.append(h("div", { class: "rowhd", style: "align-self:start;padding-top:.4em;font-weight:600;white-space:pre-line;line-height:1.35" }, label));
        for (const K of Ks) {
          const cells = keep.filter((c) => inGroup(c) && c.L === K);
          const rows = GRID_ROWS.filter((r) => cells.some((c) => gridRow(c) === r));
          grid.append(h("div", { style: "background:var(--md-default-fg-color--lightest);border-radius:4px;padding:4px;min-height:2.2em" },
            rows.map((r) => {
              const n = cells.filter((c) => gridRow(c) === r).length;
              return h("div", { style: "display:flex;align-items:center;gap:5px;line-height:1.5;margin-bottom:3px" }, square(r, `${gridLabel(r)}: ${n} runs`),
                h("span", { style: "white-space:nowrap" }, gridLabel(r)), h("span", { style: "color:var(--md-default-fg-color--light);margin-left:auto" }, `${n}`));
            })));
        }
      }
      grid.append(h("div", {}), ...Ks.map((K) => h("div", { class: "hd", style: "text-align:center;font-weight:600" }, `K = ${K}`)));
      return grid;
    };
    el.replaceChildren(bar, out);
    render();
  };

  // ---------- every benchmark considered, analysed and screened, by task ----------
  /* data-panel="groups" (default): analysed and screened side by side, each a
     multi-column flow of task blocks. data-panel="flow": one flow of task blocks,
     the analysed benchmarks in dark and the screened ones in grey. */
  S.VIEWS.benchmarksConsidered = async (el) => {
    const d = await S.load("benchmarks_considered");
    S.hoverTips(el);
    const tasks = uniq(d.map((r) => r.task)).sort((a, b) => d.filter((r) => r.task === b).length - d.filter((r) => r.task === a).length);
    const line = (r) => h("div", { style: `display:flex;justify-content:space-between;gap:6px;line-height:1.35${r.group === "screened" && el.dataset.panel === "flow" ? ";color:var(--md-default-fg-color--light)" : ""}` },
      h("span", { class: "has-tip", "data-tip": `${r.group === "analysed" ? "Analysed" : "Screened, not analysed"}\nFormats: ${r.formats}\nLow-resource languages: ${r.low_resource ?? "none"}` }, r.name),
      h("span", { style: "color:var(--md-default-fg-color--light);white-space:nowrap" }, `${r.trained}/${r.langs}`));
    const block = (t, rows) => rows.length ? h("div", { style: "break-inside:avoid;margin-bottom:6px" },
      h("div", { style: "font-weight:600;color:var(--md-primary-fg-color);border-bottom:1px solid var(--md-default-fg-color--lightest);margin-bottom:1px" }, t),
      rows.sort((a, b) => (a.group > b.group) - (a.group < b.group) || b.trained - a.trained).map(line)) : null;
    const flow = (cols, rows) => h("div", { style: `column-count:${cols};column-gap:14px` }, tasks.map((t) => block(t, rows.filter((r) => r.task === t))));
    const box = (title, cols, rows) => h("div", { style: `flex:${cols};background:var(--md-default-fg-color--lightest);border-radius:6px;padding:6px 8px` },
      h("div", { style: "font-weight:700;margin-bottom:4px" }, `${title} (${rows.length})`), flow(cols, rows));
    const body = el.dataset.panel === "flow"
      ? h("div", { style: "font-size:.62rem" }, flow(5, d))
      : h("div", { style: "display:flex;gap:8px;font-size:.62rem;align-items:flex-start" },
        box("Analysed", 3, d.filter((r) => r.group === "analysed")), box("Screened, not analysed", 2, d.filter((r) => r.group === "screened")));
    el.replaceChildren(body, note(el.dataset.panel === "flow"
      ? "Every benchmark we considered, grouped by the task it targets: analysed in dark, screened but not analysed in grey. The numbers are the trained languages over the languages the harness offers. Hover a benchmark for its formats and its low-resource languages."
      : "Every benchmark we considered, grouped by the task it targets. The numbers are the trained languages over the languages the harness offers. Hover a benchmark for its formats and its low-resource languages."));
  };

  // ---------- answer formats and scoring rules on one item ----------
  /* Default: the letter and cloze formats and the three scorings, in words.
     data-panel="detailed": the letter, cloze and continuation formats with the
     scores a small public model gives each option (eval_formats_example.json,
     scripts/eval_formats_example.py), and how bBPB is computed. */
  const box = (title, ...kids) => h("div", { style: "background:var(--md-default-fg-color--lightest);border-radius:6px;padding:8px 10px" },
    h("div", { style: "font-weight:600;margin-bottom:4px" }, title), ...kids);
  const pre = (text) => h("pre", { class: "viz-prompt", style: "margin:0;white-space:pre-wrap" }, text);
  const cell = (text, span) => h("div", { style: `border:1px solid var(--md-default-fg-color--lightest);border-radius:6px;padding:8px 10px${span ? `;grid-column:${span}` : ""}` }, text);
  const head = (text) => h("div", { style: "font-weight:600;align-self:center" }, text);

  S.VIEWS.evalFormats = async (el) => {
    if (el.dataset.panel === "detailed") return evalFormatsDetailed(el);
    const opts = (labels, gold) => h("ol", { style: "margin:.3em 0 0 1.2em;padding:0" },
      labels.map((o, i) => h("li", { style: i === gold ? "font-weight:600;color:var(--viz-good)" : null }, o, i === gold ? "  (gold)" : "")));
    el.replaceChildren(h("div", { style: "display:grid;grid-template-columns:10em 1fr 1fr;gap:6px;font-size:.78rem" },
      h("div", {}),
      box("Letter format: expects \u201cC\u201d",
        pre("Which is the capital of Spain?\nA. Paris\nB. Zurich\nC. Madrid\nD. Buenos Aires\nAnswer:"), opts(["A", "B", "C", "D"], 2)),
      box("Cloze format: expects \u201cMadrid\u201d",
        pre("Question: Which is the capital of Spain?\nAnswer:"), opts(["Paris", "Zurich", "Madrid", "Buenos Aires"], 2)),
      head("Accuracy (length-normalised)"),
      cell("The model picks the letter with the highest likelihood. Every option has one character, so the normalisation changes nothing. Chance is 1/4."),
      cell("The model picks the answer text with the highest likelihood per byte, so longer options are not penalised. Chance is 1/4."),
      head("bBPB"),
      cell("Bits per byte of the gold continuation \u201cC\u201d. Lower is better, and there is no chance level."),
      cell("Bits per byte of the gold continuation \u201cMadrid\u201d. Lower is better, and there is no chance level."),
      head("BPB"),
      cell("Bits per byte of held-out text in the language, with no question at all: the language-modelling signal behind every benchmark.", "2 / 4"),
    ), note("One item in the two answer formats and the three scorings we use. Accuracy needs a chance level, so the above-chance threshold applies to it. bBPB and BPB have none."));
  };

  async function evalFormatsDetailed(el) {
    const d = await S.load("eval_formats_example");
    const TITLES = { letter: "Letter: expects \u201cC\u201d", cloze: "Cloze: expects \u201cMadrid\u201d", continuation: "Continuation: expects \u201cMadrid\u201d" };
    const f2 = (x) => S.fmt(x, 2);
    const scores = (f) => h("table", { class: "viz-table", style: "margin-top:4px;font-size:.66rem" },
      h("thead", {}, h("tr", {}, ["option", "log p", "per byte", "bits/byte"].map((t) => h("th", { style: "cursor:default" }, t)))),
      h("tbody", {}, f.options.map((o, i) => h("tr", { style: i === d.gold ? "font-weight:600;color:var(--viz-good)" : null },
        h("td", {}, o.option.trim(), i === f.pick ? " \u2190 pick" : ""), h("td", { class: "num" }, f2(o.loglik)),
        h("td", { class: "num" }, f2(o.per_byte)), h("td", { class: "num" }, f2(o.bpb))))));
    const F = d.formats, g = d.gold;
    const bits = (k) => `${f2(-F[k].options[g].loglik)} nats / (ln 2 \u00d7 ${new TextEncoder().encode(F[k].options[g].option).length} bytes) = ${f2(F[k].options[g].bpb)}`;
    el.replaceChildren(h("div", { style: "display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:.72rem" },
      ["letter", "cloze", "continuation"].map((k) => box(TITLES[k], pre(F[k].prompt), scores(F[k]))),
      cell(h("span", {}, h("b", {}, "Accuracy (length-normalised). "),
        "The model picks the option with the highest log-probability per byte, so long options are not penalised. A letter has one character, so in the letter format this is plain accuracy. ",
        `Here every format picks the gold option, but in the letter format the margin is small (C at ${f2(F.letter.options[g].loglik)} against A at ${f2(F.letter.options[0].loglik)}), while in the cloze and continuation formats \u201cMadrid\u201d is far ahead.`), "1 / 4"),
      cell(h("span", {}, h("b", {}, "bBPB. "),
        "The negative log-probability of the gold option, divided by ln 2 and by its length in UTF-8 bytes, so that options and languages of different lengths compare. Lower is better, and there is no chance level, so a model at chance in accuracy still has a bBPB that moves. ",
        `Letter: ${bits("letter")}. Cloze: ${bits("cloze")}. Continuation: ${bits("continuation")}.`), "1 / 4"),
      cell(h("span", {}, h("b", {}, "BPB. "), "The same bits per byte on held-out text in the language, with no question at all."), "1 / 4"),
    ), note(`Scores from ${d.model}, a small public model, as the harness computes them: log p is the summed log-probability of the option given the prompt, in nats. Gold in green.`));
  }
})();

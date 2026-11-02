/* Additional interactive views of the english-only page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, css, fmt, pct, uniq, note, tiles, LADDER, LS } = S;

  // Plot's tooltip box reads --plot-background (white by default): follow the theme's surface.
  const tipBg = (el) => el.style.setProperty("--plot-background", "var(--viz-surface)");
  const missing = (el) => el.replaceChildren(h("div", { class: "viz-empty" }, "The results of this analysis are not in this build yet; they appear here once it lands."));
  async function loadOr(el, name) {
    tipBg(el);
    try { return await S.load(name); } catch (e) { missing(el); return null; }
  }
  const lColor = (L) => S.series(6)[LS.indexOf(+L)];
  const swatch = (c, hollow) => h("i", { style: `display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.35em;vertical-align:-1px;${hollow ? `border:2px solid ${c}` : `background:${c}`}` });
  const legend = (items) => h("div", { class: "ladder-legend" }, items.map(([c, label, hollow]) => h("span", {}, swatch(c, hollow), label)));

  // ---------- 1. the verdict as a scorecard ----------
  const PLAIN = {
    "higher English accuracy (deep)": "Higher scores on English benchmarks (deep models)",
    "higher English accuracy (shallow)": "Higher scores on English benchmarks (shallow models)",
    "lower English BPB (deep)": "Lower bits per byte on English text (deep)",
    "lower English BPB (shallow)": "Lower bits per byte on English text (shallow)",
    "clears chance earlier (deep)": "English benchmarks leave chance at a smaller size",
    "English BPB at least as predictable in size": "The 1.7B English BPB is forecast from small models at least as well",
    "English accuracy at least as predictable in size": "English accuracy follows the size trend at least as cleanly",
    "higher decision accuracy (mirrored families)": "Small models rank the training variants more like 1.7B does",
    "higher SNR (mirrored families)": "Higher signal-to-noise across the training variants",
    "higher SNR (deep against shallow, the shared decision)": "Higher signal-to-noise on the decision every K shares (deep vs shallow)",
    "L1 DA-size in DataDecide's range": "Ranking agreement inside the range of DataDecide's English ladder",
    "L1 SNR next to DataDecide": "Signal-to-noise next to DataDecide's",
  };
  const STATUS = {
    yes: ["✓", "holds", "--viz-good"],
    partly: ["~", "partly", "--viz-warn"],
    "not like-for-like": ["≠", "not like-for-like: the decisions differ by L", "--viz-muted"],
    "not comparable": ["–", "not comparable: different definitions", "--viz-muted"],
    "not computed": ["–", "not computed", "--viz-muted"],
  };

  S.VIEWS.englishScorecard = async (el) => {
    const d = await loadOr(el, "english_only");
    if (!d) return;
    const draw = () => {
      const n = (v) => d.verdict.filter((r) => r.verdict === v).length;
      el.replaceChildren(
        tiles([[n("yes"), "expectations hold"], [n("partly"), "hold partly"], [d.verdict.length - n("yes") - n("partly"), "cannot be graded (different decisions or definitions)"]]),
        h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.6em;margin-top:.8em" },
          d.verdict.map((r) => {
            const [icon, word, col] = STATUS[r.verdict] ?? ["?", r.verdict, "--viz-muted"];
            return h("div", { class: "viz-card", style: `margin:0;border-left:4px solid ${css(col)}` },
              h("p", { style: `margin:0 0 .3em;font-weight:600;color:${css(col)}` }, `${icon} ${word}`),
              h("p", { style: "margin:0 0 .4em;font-weight:600" }, PLAIN[r.expectation] ?? r.expectation),
              h("p", { class: "viz-note", style: "margin:0" }, h("b", {}, "English only: "), r.l1),
              h("p", { class: "viz-note", style: "margin:0" }, h("b", {}, "against: "), r.comparator));
          })),
        note("One card per expectation; the table above has the exact measure behind each. Grey cards are reported without a grade: their decisions or definitions differ between the two sides."));
    };
    S.REDRAW.push(draw);
    draw();
  };

  // ---------- 2. pick an English benchmark: the English-only line against every multilingual one ----------
  S.VIEWS.englishBench = async (el) => {
    const d = await loadOr(el, "english_only_bench");
    if (!d) return;
    const label = (t) => (t.task === "bpb_dclm" ? "English text, bits per byte (bpb_dclm)" : `${t.popular ? "★ " : ""}${t.name ?? t.task}${t.name ? ` (${t.task})` : ""}`);
    const order = [...d.tasks].sort((a, b) => (b.task === "bpb_dclm") - (a.task === "bpb_dclm") || b.popular - a.popular || label(a).localeCompare(label(b)));
    const byTask = Object.fromEntries(d.tasks.map((t) => [t.task, t]));
    S.mount(el, [
      { key: "task", label: "English benchmark", options: order.map((t) => ({ value: t.task, label: label(t) })), value: "hellaswag", string: true },
      { key: "arch", label: "Depth", type: "seg", options: ["deep", "shallow"] },
    ], (s) => {
      const t = byTask[s.task];
      const bpb = t.scoring === "bpb";
      const per = d.scores[s.task]?.[s.arch] ?? {};
      const sizes = LADDER.filter((z) => per[z]);
      if (!sizes.length) return h("div", { class: "viz-empty" }, "No score of this task at this depth.");
      const l1 = sizes.map((size) => ({ size, L: 1, y: per[size].l1, sd: per[size].sd, gated: per[size].gated.every(Boolean) }));
      const other = sizes.flatMap((size) => d.others.map((L, i) => ({ size, L, y: per[size].other[i], gated: per[size].gated[i] })).filter((r) => r.y != null));
      const top = sizes[sizes.length - 1];
      const atTop = other.filter((r) => r.size === top && !r.gated);   // the comparisons the gate keeps
      const better = (a, b) => (bpb ? a < b : a > b);
      const ahead = atTop.filter((r) => better(per[top].l1, r.y)).length;
      const gap = per[top].l1 - d3.mean(atTop, (r) => r.y);
      const all = [...l1, ...other];
      const ys = all.map((r) => r.y).concat(l1.filter((r) => r.sd).flatMap((r) => [r.y - 2 * r.sd, r.y + 2 * r.sd])).concat(t.chance != null ? [t.chance] : []);
      return [
        atTop.length ? tiles([[`${ahead} of ${atTop.length}`, `multilingual models the English-only one beats at ${top}${bpb ? "" : ", above chance"}`],
          [bpb ? `${fmt(-gap, 3)} bits` : `${fmt(100 * gap, 1)} pts`, `English-only ${bpb ? "lower" : "higher"} than the multilingual mean at ${top}`]])
          : tiles([["–", `at chance at ${top}: no comparison counts there`]]),
        plot({
          height: 320, marginLeft: 55,
          x: { type: "point", domain: sizes, label: "model size →", padding: 0.3 },
          y: { label: bpb ? "↓ bits per byte on English text" : "↑ score", grid: true, domain: d3.extent(ys), nice: true },
          marks: [
            t.chance != null ? Plot.ruleY([t.chance], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }) : null,
            t.chance != null ? Plot.text([t.chance], { x: sizes[0], y: (v) => v, dy: -7, dx: -10, textAnchor: "start", text: () => "chance", fill: css("--viz-muted") }) : null,
            Plot.ruleX(l1.filter((r) => r.sd), { x: "size", y1: (r) => r.y - 2 * r.sd, y2: (r) => r.y + 2 * r.sd, stroke: lColor(1), strokeWidth: 8, strokeOpacity: 0.25 }),
            Plot.line(other, { x: "size", y: "y", z: "L", stroke: (r) => lColor(r.L), strokeWidth: 1.5 }),
            Plot.line(l1, { x: "size", y: "y", stroke: lColor(1), strokeWidth: 3.5 }),
            Plot.dot(all, { x: "size", y: "y", r: (r) => (r.L === 1 ? 5 : 3.5), stroke: (r) => lColor(r.L), strokeWidth: 1.5,
              fill: (r) => (r.gated ? css("--viz-surface") : lColor(r.L)) }),
            Plot.dot(all, { x: "size", y: "y", r: 8, fill: "transparent", tip: true,
              title: (r) => `${r.L === 1 ? "English only (K1)" : `multilingual K${r.L}`} · ${r.size}: ${fmt(r.y, 3)}${r.sd ? ` (seed sd ${fmt(r.sd, 4)})` : ""}${r.gated ? "\nat chance here: not counted in the gap" : ""}` }),
          ].filter(Boolean),
        }),
        legend([[lColor(1), "English only (K1), with ±2 seed sd where replicates exist"], ...d.others.map((L) => [lColor(L), `K${L}`])]),
        note(`Bold: the model trained on English only. Thin: the ${s.arch} baseline model of the same size at each other language setting, which gives English half of its tokens. Hollow dots: at chance at that size, so the comparison is left out of the averages above. ★ marks the widely used benchmarks.`),
      ];
    });
  };

  // ---------- 3. which English benchmarks scale, per English corpus (the paper's Figure 11) ----------
  // The regimes in the order and colours of the scaling-predictability figures (paper.js, paperRq1).
  const REG_ORDER = ["declines with size", "predictable across both", "predictable across size only", "predictable during training only", "weak / unpredictable in both"];
  const regColor = (r) => (r === "filtered" ? css("--viz-grid") : S.series(REG_ORDER.length)[REG_ORDER.indexOf(r)] ?? css("--viz-neutral"));
  const REGIMES = {
    "predictable across both": ["scales", "scales with size and along training"],
    "predictable across size only": ["size only", "scales with size, not along training"],
    "predictable during training only": ["training only", "scales along training, not with size"],
    "weak / unpredictable in both": ["no trend", "no clear trend either way"],
    "declines with size": ["worse", "gets worse with size"],
    filtered: ["at chance", "does not pass the above-chance filter at enough sizes to fit"],
  };
  const CORPORA = ["DCLM-Edu", "DCLM", "FineWeb"];
  const CORPUS_NOTE = { "DCLM-Edu": "DCLM-Edu, the default English data", DCLM: "DCLM without the educational filter", FineWeb: "FineWeb" };
  const BOTH = "predictable across both";
  const row = (...nodes) => h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;justify-content:center" }, ...nodes);

  // A seg control choosing between modes, each drawn the first time it is shown.
  const modes = (el, label, options) => {
    const boxes = options.map(() => h("div"));
    const started = new Set();
    const seg = S.control({ type: "seg", label, options: options.map((o, i) => ({ value: i, label: o.label })) }, (i) => show(i));
    const show = (i) => {
      boxes.forEach((b, j) => { b.hidden = i !== j; });
      if (!started.has(i)) { started.add(i); options[i].draw(boxes[i]); }
    };
    el.replaceChildren(h("div", { class: "viz-controls" }, seg), ...boxes);
    show(0);
  };

  S.VIEWS.englishRegimes = async (el) => {
    const d = await loadOr(el, "english_only_regimes");
    if (!d) return;
    const tasks = uniq(d.rows.map((r) => r.task));
    const cell = Object.fromEntries(d.rows.map((r) => [`${r.task}|${r.corpus}`, r]));
    const meta = Object.fromEntries(d.rows.map((r) => [r.task, r]));
    const nBoth = (t) => CORPORA.filter((c) => cell[`${t}|${c}`]?.regime === BOTH).length;
    const nFilt = (t) => CORPORA.filter((c) => cell[`${t}|${c}`]?.regime === "filtered").length;
    const fitted = (c) => d.rows.filter((r) => r.corpus === c && r.regime !== "filtered");
    const pop = tasks.filter((t) => meta[t].popular);
    const popAbove = pop.filter((t) => nFilt(t) < 3);
    const popAll = pop.filter((t) => nBoth(t) === 3);
    const popNot = popAbove.filter((t) => nBoth(t) < 3);
    const allThree = tasks.filter((t) => nBoth(t) === 3);
    const nFit = uniq(d.rows.filter((r) => r.regime !== "filtered").map((r) => r.task)).length;
    const summary = () => tiles([
      [`${allThree.length} of ${nFit}`, "English accuracy tasks predictable across both size and training in all three corpora"],
      [CORPORA.map((c) => fitted(c).filter((r) => r.regime === BOTH).length).join(" / "), `in each corpus (${CORPORA.join(" / ")})`],
      [`${popAll.length} of ${popAbove.length}`, `commonly used benchmarks above chance that do so in all three${popNot.length ? ` (all but ${popNot.map((t) => meta[t].name).join(", ")})` : ""}`],
    ]);

    // the paper's view: R² of the model-size fit against the median R² of the training fits, one panel per corpus
    const scatter = (box) => {
      tipBg(box);
      S.mount(box, [{ key: "q", label: "Find a benchmark", type: "text" }], (s) => {
        const q = (s.q ?? "").trim().toLowerCase();
        const pts = d.rows.filter((r) => r.regime !== "filtered");
        const hit = (r) => q && `${r.name} ${r.task}`.toLowerCase().includes(q);
        const shown = REG_ORDER.filter((g) => pts.some((r) => r.regime === g));
        const lo = Math.min(0, d3.min(pts, (r) => r.r2_size), d3.min(pts, (r) => r.r2_trajectory));
        const tip = (r) => `${r.name}${r.popular ? " (commonly used)" : ""}\n${r.corpus}: ${r.regime}\nmodel-size R² ${fmt(r.r2_size)}, ρ ${fmt(r.rho_size)} · training R² ${fmt(r.r2_trajectory)}`;
        return [summary(),
          row(plot({
            width: 720, height: 300, marginLeft: 45, marginTop: 30,
            fx: { domain: CORPORA, label: null, tickFormat: (c) => `${c} (${fitted(c).length} tasks)`, padding: 0.12 },
            x: { domain: [lo, 1], label: "R² of the model-size scaling fit", grid: true, ticks: [0, 0.5, 1] },
            y: { domain: [lo, 1], label: "Median R² across training-trajectory fits", grid: true },
            color: { domain: shown, range: shown.map((g) => regColor(g)), legend: true, label: "Regime" },
            marks: [
              Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
              Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
              Plot.dot(pts, { fx: "corpus", x: "r2_size", y: "r2_trajectory", fill: "regime", r: (r) => (hit(r) ? 5.5 : 3.2), fillOpacity: q ? 0.35 : 0.85,
                stroke: (r) => (r.regime === "declines with size" ? css("--md-default-fg-color") : "none"), strokeWidth: 0.8 }),
              Plot.dot(pts.filter(hit), { fx: "corpus", x: "r2_size", y: "r2_trajectory", fill: "regime", r: 5.5, stroke: css("--md-default-fg-color"), strokeWidth: 1.2 }),
              Plot.text(pts.filter(hit), { fx: "corpus", x: "r2_size", y: "r2_trajectory", text: "name", dx: -9, textAnchor: "end", fontSize: 10, fill: css("--md-default-fg-color"), stroke: css("--viz-surface"), strokeWidth: 3, paintOrder: "stroke" }),
              Plot.dot(pts, { fx: "corpus", x: "r2_size", y: "r2_trajectory", r: 7, fill: "transparent", tip: true, title: tip }),
            ],
          })),
          note(`One dot per English accuracy task with a fit, for the monolingual English models (K = 1) trained on each corpus: the R² of the fit across model size against the median R² of the fits along training, coloured by regime (the ringed dots get worse with size). ${q ? "Hover" : "Type a name above to label its dots; hover"} a dot for its benchmark. These single-cell fits are not comparable in level with the medians over K of the multilingual figure.`)];
      });
      const input = box.querySelector("input[type=text]");
      if (input) input.placeholder = "hellaswag, mmlu, piqa…";
    };

    // the regime of every English task in each corpus, with its final accuracy per size
    const tableView = (box) => {
      tipBg(box);
      const sorted = [...tasks].sort((a, b) => meta[b].popular - meta[a].popular || (nFilt(a) === 3) - (nFilt(b) === 3)
        || nBoth(b) - nBoth(a) || meta[a].name.localeCompare(meta[b].name));
      let selected = "hellaswag";
      const { render } = S.mount(box, [
        { key: "show", label: "Rows", type: "seg", options: [{ value: "popular", label: "commonly used benchmarks" }, { value: "fitted", label: "every task with a fit" }, { value: "all", label: "every English task" }] },
        { key: "q", label: "Find a benchmark", type: "text" },
      ], (s) => {
        const q = (s.q ?? "").trim().toLowerCase();
        const keep = sorted.filter((t) => (s.show === "all" || (s.show === "popular" ? meta[t].popular : nFilt(t) < 3))
          && (!q || `${meta[t].name} ${t}`.toLowerCase().includes(q)));
        const chip = (r) => {
          const [word, long] = REGIMES[r?.regime] ?? ["–", "no value"];
          const tip = r && r.regime !== "filtered" ? `${long}\nmodel-size R² ${fmt(r.r2_size)} · training R² ${fmt(r.r2_trajectory)} · ρ with size ${fmt(r.rho_size)}` : long;
          return h("td", { title: tip, style: "white-space:nowrap" },
            h("i", { style: `display:inline-block;width:.8em;height:.8em;border-radius:3px;margin-right:.4em;vertical-align:-1px;background:${r ? regColor(r.regime) : css("--viz-neutral")}` }), word);
        };
        const rows = keep.map((t) => h("tr", {
          style: `cursor:pointer;${t === selected ? "outline:2px solid var(--viz-s1);outline-offset:-2px" : ""}`,
          onclick: () => { selected = t; render(); },
        }, h("td", {}, meta[t].popular ? h("b", {}, `★ ${meta[t].name}`) : meta[t].name),
        ...CORPORA.map((c) => chip(cell[`${t}|${c}`]))));
        const table = h("div", { class: "viz-scroll", style: s.show === "popular" ? "max-height:none" : null },
          h("table", { class: "viz-table" },
            h("thead", {}, h("tr", {}, h("th", {}, "benchmark (click a row for its curves)"), ...CORPORA.map((c) => h("th", { title: CORPUS_NOTE[c] }, c)))),
            h("tbody", {}, rows)));
        return [
          summary(),
          h("div", { class: "ladder-legend" }, Object.entries(REGIMES).map(([k, [word, long]]) => h("span", { title: long },
            h("i", { style: `display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:.35em;vertical-align:-2px;background:${regColor(k)}` }), `${word}: ${long}`))),
          keep.length ? table : h("div", { class: "viz-empty" }, "No English benchmark matches."),
          curves(selected),
          note("★ marks the benchmarks commonly used in model development. A task is fitted when it passes the above-chance filter at enough model sizes."),
        ];
      });
      const input = box.querySelector("input[type=text]");
      if (input) input.placeholder = "mmlu, piqa, arc…";
    };

    function curves(t) {
      const c = d.curves[t];
      if (!c) return h("p", { class: "viz-note" }, `No per-size scores for ${meta[t]?.name ?? t}.`);
      const pts = CORPORA.flatMap((corpus, i) => LADDER.filter((z) => c[corpus]?.[z]).map((size) => ({ corpus, i, size, y: c[corpus][size][0], above: c[corpus][size][1] === 1 })));
      const col = (i) => S.series(3)[i];
      const chance = d.chance[t];
      return h("div", { style: "margin-top:1em" },
        h("p", { class: "viz-note" }, h("b", {}, meta[t].name), ": the final accuracy of the English-only model (K = 1) of each corpus at each size. Hollow: at chance at that size (the fit leaves it out)."),
        row(plot({
          height: 260, marginLeft: 55,
          x: { type: "point", domain: LADDER, label: "model size →", padding: 0.3 },
          y: { label: "↑ accuracy", grid: true, domain: d3.extent(pts.map((p) => p.y).concat(chance != null ? [chance] : [])), nice: true },
          marks: [
            chance != null ? Plot.ruleY([chance], { stroke: css("--viz-bad"), strokeDasharray: "2 3" }) : null,
            Plot.line(pts, { x: "size", y: "y", z: "corpus", stroke: (p) => col(p.i), strokeWidth: 2 }),
            Plot.dot(pts, { x: "size", y: "y", r: 4.5, stroke: (p) => col(p.i), strokeWidth: 1.5, fill: (p) => (p.above ? col(p.i) : css("--viz-surface")), tip: true,
              title: (p) => `${p.corpus} · ${p.size}: ${fmt(p.y, 3)}${p.above ? "" : " (at chance)"}` }),
          ].filter(Boolean),
        })),
        legend(CORPORA.map((corpus, i) => [col(i), CORPUS_NOTE[corpus]])));
    }

    modes(el, "Show", [
      { label: "R² of the fits (paper)", draw: scatter },
      { label: "regime per corpus, with the scores", draw: tableView },
    ]);
  };
})();

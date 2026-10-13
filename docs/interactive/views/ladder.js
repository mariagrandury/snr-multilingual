/* Additional interactive views of the ladder page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, load, mount, note, tiles, table, series, css, langName, SIZES, uniq } = S;

  // ---------- ladderPairs: the grid read one design decision at a time ----------
  // Each decision is one axis of the registry's cells (ladder.json). A row
  // holds the runs that share everything but that axis, a column is a model
  // size, and each run is one dot: its colour is the level of the decision
  // (the baseline level always first, always blue), its fill how far it got
  // (ladder_status.json, from the ladder report).
  const ENGLISH = ["DCLM (edu-filtered)", "DCLM (no edu filter)", "FineWeb"];
  const DECISIONS = [
    { key: "arch", label: "Depth", levels: ["deep", "shallow"],
      name: { deep: "deep (baseline)", shallow: "shallow" },
      what: "Deep (baseline) or shallow model with the same number of non-embedding parameters. The 600M shallow model is not exactly size-matched (3.7 % more parameters, about 15 % more compute)." },
    { key: "activation", label: "Activation", levels: ["xielu", "swiglu"],
      name: { xielu: "xIELU (baseline)", swiglu: "SwiGLU" },
      what: "xIELU (baseline) or SwiGLU activation, on the deep model." },
    { key: "optimizer", label: "Optimizer", levels: ["ademamix", "muon"],
      name: { ademamix: "AdEMAMix (baseline)", muon: "Muon" },
      what: "AdEMAMix (baseline) or Muon optimizer, on the deep model." },
    { key: "list", label: "Language list", levels: ["A", "B"],
      name: { A: "list A · ranked by data (baseline)", B: "list B · chosen for diversity" },
      what: "Which languages enter the mixture: list A adds them in order of available data (baseline), list B picks them for diversity of family and script." },
    { key: "T", label: "Temperature", levels: [1, 3],
      name: { 1: "T = 1 · shares ∝ data (baseline)", 3: "T = 3 · flatter split" },
      what: "How the non-English half is split: T = 1 in proportion to the available data (baseline), T = 3 flatter, so smaller languages get more tokens." },
    { key: "lang2", label: "Second language", levels: ["ru", "zh", "es"],
      name: { ru: "Russian (baseline)", zh: "Chinese", es: "Spanish" },
      what: "At L = 2, the one language next to English: Russian (baseline), Chinese or Spanish." },
    { key: "english", label: "English data", levels: ENGLISH,
      name: { [ENGLISH[0]]: "DCLM, edu-filtered (baseline)", [ENGLISH[1]]: "DCLM, no edu filter", [ENGLISH[2]]: "FineWeb" },
      what: "At L = 1 (English only), where the English text comes from: DCLM with its educational-quality filter (baseline), DCLM without it, or FineWeb." },
    { key: "seed", label: "Seeds", levels: ["1904", "replicate"], of: (c) => (c.seed === 1904 ? "1904" : "replicate"),
      name: { 1904: "seed 1904 (every run)", replicate: "replicate seeds" },
      what: "The same configuration trained again from another seed: the yardstick of noise every design decision is compared with." },
  ];
  const CONTEXT = ["L", "arch", "activation", "optimizer", "list", "T", "lang2", "english", "seed"];
  const BASE = { arch: "deep", activation: "xielu", optimizer: "ademamix", list: "A", T: 1, lang2: "ru", english: ENGLISH[0], seed: 1904 };
  const WORD = {
    arch: (v) => v, activation: (v) => ({ xielu: "xIELU", swiglu: "SwiGLU" }[v] ?? v),
    optimizer: (v) => ({ ademamix: "AdEMAMix", muon: "Muon" }[v] ?? v), list: (v) => `list ${v}`,
    T: (v) => `T = ${v}`, lang2: (v) => langName(v), english: (v) => v, seed: (v) => `seed ${v}`,
  };
  const STATES = {
    evaluated: "trained and evaluated",
    evaluating: "trained, evaluation running",
    training: "still training (wedge = share of its tokens seen)",
    excluded: "excluded (diverged or off trend)",
    queued: "not started",
  };

  const STYLE = `
.lp-axis { margin: .2em 0 .8em; }
.lp-grid { display: grid; gap: 3px; min-width: 34em; align-items: stretch; }
.lp-grid .hd { font-weight: 600; color: var(--viz-muted); padding: .25em; text-align: center; font-size: .68rem; }
.lp-grid .hd.ref { border-bottom: 2px solid var(--viz-muted); }
.lp-grid .rowhd { padding: .3em .5em .3em 0; text-align: right; font-size: .68rem; line-height: 1.25; color: var(--md-default-fg-color); }
.lp-grid .rowhd small { display: block; color: var(--viz-muted); }
.lp-grid .cell { display: flex; flex-wrap: wrap; gap: 2px; justify-content: center; align-content: center; padding: 3px; min-height: 2em; border-radius: 4px; background: var(--md-default-fg-color--lightest); }
.lp-grid .cell.above { background: transparent; border: 1px dashed var(--md-default-fg-color--lighter); }
.lp-grid .ready { text-align: center; font-size: .68rem; color: var(--viz-muted); padding: .3em 0; border-top: 1px solid var(--md-default-fg-color--lighter); }
.lp-grid .ready b { color: var(--md-default-fg-color); }
.lp-dot { width: 22px; height: 22px; padding: 2px; border: 0; background: none; cursor: pointer; border-radius: 50%; }
.lp-dot:focus-visible, .lp-dot:hover { outline: 2px solid var(--md-default-fg-color); outline-offset: 0; }
.lp-dot svg { display: block; }
.lp-legend { display: flex; flex-wrap: wrap; gap: .3em 1.2em; margin: .5em 0; font-size: .68rem; color: var(--viz-muted); align-items: center; }
.lp-legend b { color: var(--md-default-fg-color); font-weight: 600; }
.lp-legend span { display: inline-flex; align-items: center; gap: .35em; }
.lp-tip { max-width: 260px; }
.lp-tip b { font-size: .72rem; }
.lp-tip .st { color: var(--viz-muted); }
`;

  // One run as a dot: colour = level, fill = how far it got.
  function dot(color, st, progress) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", 18); svg.setAttribute("height", 18); svg.setAttribute("viewBox", "0 0 18 18");
    svg.setAttribute("aria-hidden", "true");
    const add = (tag, attrs) => { const e = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v); svg.append(e); };
    if (st === "evaluated") add("circle", { cx: 9, cy: 9, r: 7.5, fill: color });
    else if (st === "evaluating") add("circle", { cx: 9, cy: 9, r: 6.5, fill: color, "fill-opacity": 0.35, stroke: color, "stroke-width": 2 });
    else if (st === "excluded") {
      add("circle", { cx: 9, cy: 9, r: 6.5, fill: "none", stroke: css("--viz-muted"), "stroke-width": 1.5 });
      add("path", { d: "M5.5 5.5 L12.5 12.5 M12.5 5.5 L5.5 12.5", stroke: css("--viz-muted"), "stroke-width": 1.8, "stroke-linecap": "round" });
    } else {
      add("circle", { cx: 9, cy: 9, r: 6.5, fill: "none", stroke: color, "stroke-width": 1.5, "stroke-dasharray": st === "queued" ? "2 2" : null });
      const p = Math.max(0, Math.min(1, progress ?? 0));
      if (p > 0.01) {
        const a = 2 * Math.PI * p, x = 9 + 6.5 * Math.sin(a), y = 9 - 6.5 * Math.cos(a);
        add("path", { d: p > 0.99 ? "M9 2.5 A6.5 6.5 0 1 1 8.99 2.5 Z" : `M9 9 L9 2.5 A6.5 6.5 0 ${p > 0.5 ? 1 : 0} 1 ${x.toFixed(2)} ${y.toFixed(2)} Z`, fill: color });
      }
    }
    return svg;
  }

  S.VIEWS.ladderPairs = async (el) => {
    if (!document.getElementById("lp-style")) document.head.append(h("style", { id: "lp-style" }, STYLE));
    await S.languages();
    const [d, st] = await Promise.all([load("ladder"), load("ladder_status").catch(() => ({ runs: {} }))]);
    const status = (c) => st.runs?.[c.name] ?? { state: "evaluated", analysed: null };
    el.style.position = "relative";
    const tip = h("div", { class: "viz-tip lp-tip", hidden: "" });

    const all = d.cells;
    const count = (s) => all.filter((c) => status(c).state === s).length;
    const head = tiles([
      [all.length, "runs in the grid"],
      [count("evaluated"), STATES.evaluated],
      [count("training") + count("evaluating"), "still training or being evaluated"],
      [all.filter((c) => status(c).analysed).length, "read by the current analysis tables"],
    ]);

    const levelOf = (dec, c) => (dec.of ? dec.of(c) : c[dec.key]);
    const describe = (c) => {
      const s = status(c);
      const model = `${c.arch} · ${WORD.activation(c.activation)} · ${WORD.optimizer(c.optimizer)}`;
      const data = c.L === 1 ? `100 % English (${c.english})`
        : `50 % English + ${c.L - 1} more language${c.L > 2 ? "s" : ""}${c.list ? `, list ${c.list}` : ""}, T = ${c.T}${c.lang2 ? `, second language ${langName(c.lang2)}` : ""}`;
      const state = {
        evaluated: `Trained; ${s.bench} checkpoints scored on the benchmarks, ${s.bpb} on bits per byte.`,
        evaluating: `Trained; ${s.bench} of ${st.tenths ?? 10} checkpoints scored so far.`,
        training: `Training: ${Math.round(100 * (s.progress ?? 0))} % of its tokens seen${s.bench ? `, ${s.bench} checkpoint${s.bench > 1 ? "s" : ""} scored` : ""}.`,
        excluded: "Diverged or off its scaling trend: left out of the analysis.",
        queued: "Planned, not started.",
      }[s.state];
      const analysed = s.analysed == null ? (c.size === "3B" ? "Above the 1.7B reference: read only by the 3B check (RQ10)." : "")
        : s.analysed ? "In the current analysis tables." : "Not yet in the analysis tables.";
      return [
        h("b", {}, c.name),
        h("span", {}, `${c.size} · L = ${c.L} · seed ${c.seed}`),
        h("span", {}, model),
        h("span", {}, data),
        c.tokens ? h("span", {}, `${(c.tokens / 1e9).toFixed(1)} B training tokens (100 per parameter)`) : null,
        h("span", { class: "st" }, state),
        analysed ? h("span", { class: "st" }, analysed) : null,
      ];
    };
    const show = (btn, c) => {
      tip.replaceChildren(...describe(c).filter(Boolean));
      tip.hidden = false;
      const a = btn.getBoundingClientRect(), b = el.getBoundingClientRect();
      const left = Math.min(a.left - b.left + 26, b.width - 270);
      tip.style.left = `${Math.max(0, left)}px`;
      tip.style.top = `${a.top - b.top + 24}px`;
    };
    const hide = () => { tip.hidden = true; };

    mount(el, [
      { key: "dec", label: "Design decision", type: "seg", options: DECISIONS.map((x, i) => ({ value: i, label: x.label })) },
    ], (s) => {
      const dec = DECISIONS[s.dec ?? 0];
      const colors = series(dec.levels.length);
      const ctxKeys = CONTEXT.filter((k) => k !== dec.key);
      const ctxOf = (c) => JSON.stringify(ctxKeys.map((k) => c[k]));
      const runs = all.filter((c) => levelOf(dec, c) != null && dec.levels.map(String).includes(String(levelOf(dec, c))));
      const groups = {};
      for (const c of runs) (groups[ctxOf(c)] ??= []).push(c);
      // a comparison needs at least two levels of the decision somewhere in its row
      const rows = Object.values(groups).filter((g) => uniq(g.map((c) => String(levelOf(dec, c)))).length > 1)
        .sort((a, b) => a[0].L - b[0].L || ctxOf(a[0]).localeCompare(ctxOf(b[0])));
      const shown = rows.flat();
      const sizes = SIZES.filter((z) => shown.some((c) => c.size === z));
      const lvl = (c) => dec.levels.map(String).indexOf(String(levelOf(dec, c)));

      const grid = h("div", { class: "lp-grid", style: `grid-template-columns: minmax(6.5em, max-content) repeat(${sizes.length}, minmax(3.2em, 1fr))` });
      grid.append(h("div", {}), ...sizes.map((z) => h("div", { class: `hd ${z === "1.7B" ? "ref" : ""}`, title: z === "1.7B" ? "the reference size" : null }, z === "1.7B" ? "1.7B ref." : z)));
      for (const g of rows) {
        const c0 = g[0];
        const extra = ctxKeys.filter((k) => k !== "L" && c0[k] != null && c0[k] !== BASE[k] && !(k === "T" && c0.L === 1)).map((k) => WORD[k](c0[k]));
        grid.append(h("div", { class: "rowhd" }, `L = ${c0.L}`, extra.length ? h("small", {}, extra.join(" · ")) : null));
        for (const z of sizes) {
          const here = g.filter((c) => c.size === z).sort((a, b) => lvl(a) - lvl(b) || a.seed - b.seed);
          grid.append(h("div", { class: `cell ${z === "3B" ? "above" : ""}` }, here.map((c) => {
            const stt = status(c);
            const b = h("button", {
              type: "button", class: "lp-dot",
              "aria-label": `${c.name}: ${dec.name[String(levelOf(dec, c))]}, ${STATES[stt.state]}`,
              onmouseenter: () => show(b, c), onfocus: () => show(b, c), onmouseleave: hide, onblur: hide,
            }, dot(colors[lvl(c)], stt.state, stt.progress));
            return b;
          })));
        }
      }
      // per size: rows whose runs cover at least two levels, all evaluated
      grid.append(h("div", { class: "rowhd ready" }, "comparisons ready"), ...sizes.map((z) => {
        const n = rows.filter((g) => uniq(g.filter((c) => c.size === z && status(c).state === "evaluated").map(lvl)).length > 1).length;
        const of = rows.filter((g) => uniq(g.filter((c) => c.size === z).map(lvl)).length > 1).length;
        return h("div", { class: "ready", title: `${n} of the ${of} comparisons at ${z} have every level trained and evaluated` }, of ? h("span", {}, h("b", {}, n), ` / ${of}`) : "–");
      }));

      const training = shown.filter((c) => ["training", "evaluating", "queued"].includes(status(c).state)).length;
      const Ls = uniq(shown.map((c) => c.L)).sort((a, b) => a - b);
      const summary = `${shown.length} runs in ${rows.length} comparison${rows.length > 1 ? "s" : ""} · L = ${Ls.join(", ")} · ${sizes[0]}–${sizes[sizes.length - 1]}`
        + (training ? ` · ${training} still training or being evaluated` : " · all trained and evaluated");
      const states = Object.keys(STATES).filter((k) => shown.some((c) => status(c).state === k));
      const legend = h("div", { class: "lp-legend" },
        h("b", {}, "Colour:"), dec.levels.map((v, i) => h("span", {}, dot(colors[i], "evaluated"), dec.name[String(v)])),
        h("b", {}, "Fill:"), states.map((k) => h("span", {}, dot(css("--viz-muted"), k, 0.6), STATES[k])));
      const rowsTable = shown.map((c) => {
        const t = status(c);
        return { name: c.name, size: c.size, L: c.L, level: dec.name[String(levelOf(dec, c))], state: STATES[t.state].split(" (")[0],
          bench: t.bench ?? null, progress: t.progress != null ? Math.round(100 * t.progress) : null };
      });
      return [
        h("p", { class: "lp-axis" }, h("b", {}, `${dec.label}. `), dec.what, h("br"), h("span", { class: "viz-note" }, summary)),
        legend,
        h("div", { class: "viz-scroll" }, grid),
        note("Each row is one comparison: runs identical in everything but this decision (the row label names what differs from the baseline besides L). Columns are model sizes, 1.7B is the reference and 3B (dashed) sits above it. Hover or focus a dot for the run. The bottom row counts, per size, the comparisons whose runs are all trained and evaluated."),
        h("details", {}, h("summary", {}, `The ${shown.length} runs as a table`),
          table(rowsTable, [
            { key: "name", label: "run" }, { key: "size", label: "size" }, { key: "L", label: "L" },
            { key: "level", label: dec.label.toLowerCase() }, { key: "state", label: "status" },
            { key: "progress", label: "% trained", num: true, digits: 0 }, { key: "bench", label: "checkpoints scored", num: true, digits: 0 },
          ], { sort: "L" })),
      ];
    });
    el.prepend(head);
    el.append(tip);
  };
})();

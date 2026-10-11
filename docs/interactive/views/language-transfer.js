/* Additional interactive views of the language-transfer page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, load, mount, note, fmt, pct, css, plot, uniq, LADDER } = S;
  const PROX = ["90M", "175M", "350M", "600M", "1B"];
  const chin = (f) => `${+(f * 5).toFixed(1)}C`;

  const style = () => document.getElementById("lt-style") || document.head.append(h("style", { id: "lt-style" }, `
.lt-tree { display: grid; grid-template-columns: repeat(auto-fill, minmax(15em, 1fr)); gap: .6em; font-size: .68rem; }
.lt-fam { border: 1px solid var(--viz-grid); border-radius: 6px; padding: .4em .5em; }
.lt-fam.wide { grid-column: 1 / -1; }
.lt-fam h5 { margin: 0 0 .3em; font-size: .7rem; color: var(--viz-muted); text-transform: uppercase; letter-spacing: .04em; }
.lt-sub { display: flex; gap: .4em; align-items: baseline; margin: .15em 0; }
.lt-sub > span { color: var(--viz-muted); min-width: 6.5em; flex: none; }
.lt-chips { display: flex; flex-wrap: wrap; gap: .25em; }
.lt-chip { display: inline-flex; align-items: center; gap: .3em; padding: .1em .45em .1em .3em; border-radius: 4px; border: 1px solid transparent; white-space: nowrap; cursor: default; }
.lt-chip i { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.lt-chip.none { color: var(--viz-muted); opacity: .6; }
.lt-chip.none i { border: 1px dashed var(--viz-muted); }
.lt-chip.tr { border-color: var(--viz-muted); font-weight: 600; }
.lt-chip.src { border: 2px solid var(--viz-s7); font-weight: 700; }
button.lt-chip { font: inherit; color: inherit; background: none; cursor: pointer; }
button.lt-chip:hover { background: var(--viz-neutral); }
.lt-legend { display: flex; flex-wrap: wrap; align-items: center; gap: .4em 1.2em; color: var(--viz-muted); font-size: .66rem; margin: .5em 0; }
.lt-ramp { display: inline-block; width: 9em; height: 10px; border-radius: 3px; vertical-align: -1px; margin: 0 .4em; }
.lt-grid { display: grid; grid-template-columns: minmax(9em, 1.6fr) repeat(5, minmax(2.6em, 1fr)) minmax(8em, 1.6fr); align-items: center; gap: .15em .3em; font-size: .68rem; min-width: 32em; }
.lt-grid > div { border-top: 1px solid var(--viz-grid); padding: .3em 0; min-height: 2.4em; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: .1em; }
.lt-grid > div.l { align-items: flex-start; }
.lt-grid > div.hd { border-top: 0; color: var(--viz-muted); font-weight: 600; min-height: 0; }
.lt-dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--viz-s1); background: var(--viz-surface); box-sizing: border-box; display: inline-block; }
.lt-dot.on { background: var(--viz-s1); }
.lt-grid small { color: var(--viz-muted); font-variant-numeric: tabular-nums; }
.lt-tl { display: flex; gap: 2px; }
.lt-tl i { width: 6px; height: 12px; border-radius: 2px; border: 1px solid var(--viz-s7); box-sizing: border-box; }
.lt-tl i.on { background: var(--viz-s7); }
`));

  /* Languages laid out by family (top level, then subfamily), one chip each.
     cell(code) -> {color, tip, cls} or null (no value); onPick(code) makes chips buttons. */
  function familyTree(LANG, codes, cell, onPick) {
    const fams = {};
    for (const c of codes) {
      const fam = LANG.iso[c]?.family ?? "No family on record";
      const [top, ...rest] = fam.split(",").map((x) => x.trim());
      ((fams[top] ??= {})[rest.join(", ")] ??= []).push(c);
    }
    const size = (f) => Object.values(fams[f]).flat().length;
    return h("div", { class: "lt-tree" }, Object.keys(fams).sort((a, b) => size(b) - size(a) || a.localeCompare(b)).map((top) => {
      const subs = Object.keys(fams[top]).sort((a, b) => fams[top][b].length - fams[top][a].length || a.localeCompare(b));
      return h("div", { class: size(top) > 12 ? "lt-fam wide" : "lt-fam" }, h("h5", {}, top),
        subs.map((sub) => h("div", { class: "lt-sub" }, sub ? h("span", {}, sub) : null,
          h("div", { class: "lt-chips" }, fams[top][sub].sort((a, b) => S.langName(a).localeCompare(S.langName(b))).map((c) => {
            const v = cell(c);
            const attrs = { class: `lt-chip ${v ? v.cls ?? "" : "none"}`, title: v ? v.tip : `${S.langName(c)}: no value` };
            if (onPick) { attrs.type = "button"; attrs.onclick = () => onPick(c); }
            return h(onPick ? "button" : "span", attrs, h("i", { style: v ? `background:${v.color}` : null }), S.langName(c));
          })))));
    }));
  }
  const ramp = (f, n = 8) => `linear-gradient(90deg, ${Array.from({ length: n + 1 }, (_, i) => f(i / n)).join(", ")})`;

  // ---------- BPB forecast by language, laid out by family ----------
  S.VIEWS.ltTree = async (el) => {
    style();
    const [LANG, d] = await Promise.all([S.languages(), load("transfer")]);
    const code = (task) => { const fw = task.replace("bpb_", "").split("_")[0]; return fw === "dclm" ? "en" : LANG.fw[fw]; };
    const rows = S.expand(d.per_language).map((r) => ({ ...r, code: code(r.task) })).filter((r) => r.code);
    mount(el, [
      { key: "L", label: "Training languages", options: S.LS.map((L) => ({ value: L, label: `L = ${L}` })), value: 8 },
      { key: "k", label: "Proxy rungs measured", type: "seg", options: [1, 2, 3, 4] },
      { key: "m", label: "Forecast", type: "seg", options: [{ value: "transfer", label: "pooled exponent" }, { value: "last", label: "largest proxy as is" }] },
    ], (s) => {
      const sel = rows.filter((r) => r.L === s.L && r.k === s.k);
      const by = Object.fromEntries(sel.map((r) => [r.code, r]));
      const MAX = 0.15;
      const col = (t) => d3.interpolateRgb(css("--viz-neutral"), css("--viz-s1"))(t);   // t = 1: exact
      const key = s.m === "transfer" ? "err_transfer" : "err_last";
      const err = (r) => Math.abs(r[key]);
      const tr = sel.filter((r) => r[key] != null);
      const errs = tr.map(err);
      const tree = familyTree(LANG, uniq(sel.map((r) => r.code)), (c) => {
        const r = by[c];
        if (!r || r[key] == null) return null;
        const e = err(r);
        return { color: col(1 - Math.min(e, MAX) / MAX), cls: r.trained ? "tr" : null,
          tip: `${S.langName(c)} (${r.trained ? "trained" : "never trained"} at L = ${s.L})\nobserved BPB at 1.7B ${fmt(r.observed, 3)}, forecast ${fmt(s.m === "transfer" ? r.pred_transfer : r.pred_last, 3)}: ${pct(e)} off` };
      });
      return [
        h("div", { class: "lt-legend" },
          h("span", {}, `${pct(MAX)} off or more`, h("span", { class: "lt-ramp", style: `background:${ramp(col)}` }), "exact"),
          h("span", {}, h("span", { class: "lt-chip tr" }, "bold, outlined"), " = trained at this L"),
          h("span", {}, `median error ${pct(d3.median(errs))} over ${errs.length} languages`)),
        tree,
        note(`Darker = the 1.7B BPB is forecast more closely. Median error here: ${pct(d3.median(tr.filter((r) => r.trained), err))} on the trained languages, `
          + `${pct(d3.median(tr.filter((r) => !r.trained), err))} on the never-trained ones.`),
      ];
    });
  };

  // ---------- the list decision by language group ----------
  S.VIEWS.ltGroups = async (el) => {
    style();
    const d = await load("language_transfer_views");
    const ORDER = ["trained by both levels", "trained by one level", "script trained", "script not trained"];
    const LABEL = { "trained by both levels": "Trained by both lists", "trained by one level": "Trained by one list only",
      "script trained": "Untrained, script trained", "script not trained": "Untrained, script untrained" };
    mount(el, [{ key: "thr", label: "DA threshold", type: "range", min: 0.55, max: 0.95, step: 0.05, value: 0.75 }], (s) => {
      const grid = h("div", { class: "lt-grid" }, h("div", { class: "hd l" }, "Languages"),
        PROX.map((p) => h("div", { class: "hd" }, p)), h("div", { class: "hd" }, "1.7B own run, 0.5C → 4.5C"));
      for (const g of ORDER) {
        grid.append(h("div", { class: "l" }, LABEL[g]));
        for (const p of PROX) {
          const r = d.groups.find((x) => x.group === g && x.kind === "size" && x.x === p);
          grid.append(r?.da == null ? h("div", {}, h("small", {}, "–"))
            : h("div", { title: `${LABEL[g]}: the ${p} proxy's final ranking of A vs B matches the 1.7B final on ${pct(r.da)} of the languages (mean over L8, L15, L30)` },
              h("i", { class: r.da >= s.thr ? "lt-dot on" : "lt-dot" }), h("small", {}, fmt(r.da))));
        }
        const ck = d.groups.filter((x) => x.group === g && x.kind === "ckpt").sort((a, b) => +a.x - +b.x);
        grid.append(h("div", { title: `${LABEL[g]}, the 1.7B run's own checkpoints against its final: ` + ck.map((r) => `${chin(+r.x)} ${fmt(r.da)}`).join(" · ") },
          h("span", { class: "lt-tl" }, ck.map((r) => h("i", { class: r.da >= s.thr ? "on" : null }))),
          h("small", {}, `${ck.filter((r) => r.da >= s.thr).length} of ${ck.length} ≥ ${fmt(s.thr)}`)));
      }
      return [h("div", { class: "viz-scroll" }, grid),
        note("Filled = the proxy picks the same language list as the 1.7B reference on at least the threshold share of these languages' BPB. The trained rows are what a developer measures; the two untrained rows are the transfer test.")];
    });
  };

  // ---------- which language predicts which ----------
  S.VIEWS.ltCross = async (el) => {
    style();
    const [LANG, d] = await Promise.all([S.languages(), load("language_transfer_views")]);
    const SETS = [{ value: "hellaswag", label: "HellaSwag" }, { value: "above_66_either", label: "reliable tasks (rq2)" }, { value: "above_chance", label: "every task above chance" }];
    const all = uniq(Object.values(d.cross).flatMap((m) => Object.keys(m))).sort((a, b) => S.langName(a).localeCompare(S.langName(b)));
    let ui;
    ui = mount(el, [
      { key: "set", label: "Tasks", type: "seg", options: SETS },
      { key: "dir", label: "The picked language is", type: "seg", options: [{ value: "src", label: "the proxy" }, { value: "tgt", label: "the target" }] },
      { key: "lang", label: "Language", options: all.map((c) => ({ value: c, label: S.langName(c) })), value: "en", string: true },
    ], (s) => {
      const M = d.cross[s.set];
      const codes = uniq([...Object.keys(M), ...Object.values(M).flatMap((m) => Object.keys(m))]);
      const val = (c) => (s.dir === "src" ? M[s.lang]?.[c] : M[c]?.[s.lang]);
      const neg = css("--viz-s2"), mid = css("--viz-neutral"), pos = css("--viz-s1");
      const col = (v) => { const t = Math.max(-1, Math.min(1, (v - 0.5) / 0.3)); return t < 0 ? d3.interpolateRgb(mid, neg)(-t) : d3.interpolateRgb(mid, pos)(t); };
      if (!codes.includes(s.lang)) return note(`${S.langName(s.lang)} has no task in this set; pick another language or task set.`);
      const others = codes.filter((c) => c !== s.lang && val(c) != null).map((c) => [c, val(c)]).sort((a, b) => b[1] - a[1]);
      const tree = familyTree(LANG, codes, (c) => {
        const v = val(c);
        if (v == null) return c === s.lang ? { color: "transparent", cls: "src", tip: "the picked language" } : null;
        const [p, t] = s.dir === "src" ? [s.lang, c] : [c, s.lang];
        return { color: col(v), cls: c === s.lang ? "src" : null,
          tip: `${S.langName(p)} at the proxies → ${S.langName(t)} at 1.7B: DA-size ${fmt(v)}${p === t ? " (the language's own DA)" : ""}\nclick to pick ${S.langName(c)}` };
      }, (c) => { ui.state.lang = c; el.querySelector(".viz-controls select").value = c; ui.render(); });
      const mean = d3.mean(others, (x) => x[1]);
      const name = (x) => `${S.langName(x[0])} ${fmt(x[1])}`;
      return [
        h("div", { class: "lt-legend" },
          h("span", {}, "0.2", h("span", { class: "lt-ramp", style: `background:${ramp((t) => col(0.2 + 0.6 * t))}` }), "0.8"),
          h("span", {}, "grey = 0.5, chance"),
          h("span", {}, h("span", { class: "lt-chip src" }, S.langName(s.lang)), " = picked (click any chip to pick it)")),
        tree,
        note(others.length
          ? `${S.langName(s.lang)} ${s.dir === "src" ? "ranks the other languages' 1.7B ranking" : "is ranked by the other languages"} at ${fmt(mean)} on average over ${others.length}. `
            + `Best: ${others.slice(0, 3).map(name).join(", ")}. Worst: ${others.slice(-3).reverse().map(name).join(", ")}.`
          : "No other language shares pairs with this one in this task set."),
      ];
    });
  };

  // ---------- relatives of the trained languages ----------
  S.VIEWS.ltFamily = async (el) => {
    style();
    const d = await load("language_transfer_views");
    const CELLS = uniq(d.family.lift.map((r) => r.cell));
    const CNAME = { "L2 A": "L2 A (English + Russian)", "L8 A": "L8 A (adds zh, de, ja, es, fr, it)", "L8 B": "L8 B (diversity-first list)" };
    mount(el, [
      { key: "mode", label: "Show", type: "seg", options: [{ value: "lift", label: "lift per subfamily" }, { value: "share", label: "share above chance by size" }] },
      { key: "level", label: "Family level", type: "seg", options: [{ value: "sub", label: "subfamily" }, { value: "top", label: "top-level family" }] },
      { key: "cell", label: "Cell", options: CELLS.map((c) => ({ value: c, label: CNAME[c] ?? c })), value: "L8 A", string: true },
    ], (s) => {
      const cc = S.series(CELLS.length);
      const liftTip = (r) => `${r.sub} under ${CNAME[r.cell] ?? r.cell}\n${r.trained ? "the cell trains a language of this subfamily" : "untrained subfamily"} · languages: ${r.langs}\n`
        + `${pct(r.share)} above chance vs ${pct(r.base)} English only (${r.lift > 0 ? "+" : ""}${Math.round(r.lift * 100)} pt, ${r.n} verdicts, ${r.sizes})`;
      if (s.mode === "lift") {
        const subs = uniq(d.family.lift.map((r) => r.sub));
        const mean = Object.fromEntries(subs.map((x) => [x, d3.mean(d.family.lift.filter((r) => r.sub === x), (r) => r.lift)]));
        subs.sort((a, b) => mean[b] - mean[a]);
        return [plot({
          height: 16 * subs.length + 70, marginLeft: 190, width: 680,
          x: { label: "share above chance minus English only, same tasks", grid: true, tickFormat: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)} pt` },
          y: { domain: subs, label: null },
          color: { domain: CELLS, range: cc, legend: true },
          marks: [
            Plot.ruleX([0], { stroke: css("--viz-muted") }),
            Plot.dot(d.family.lift.filter((r) => !r.trained), { x: "lift", y: "sub", stroke: "cell", fill: css("--viz-surface"), r: 4.5, strokeWidth: 2, tip: true, title: liftTip }),
            Plot.dot(d.family.lift.filter((r) => r.trained), { x: "lift", y: "sub", fill: "cell", stroke: css("--viz-surface"), r: 6.5, strokeWidth: 2, tip: true, title: liftTip }),
          ],
        }), note("Filled, larger dots: the cell trains a non-English language of that subfamily. They mostly sit right of zero, and the hollow dots (subfamilies the cell does not train) scatter around it. Drawn where a lift rests on at least 20 verdicts.")];
      }
      const rows = d.family.share.filter((r) => r.cell === s.cell && r.level === s.level)
        .flatMap((r) => [{ ...r, who: `${r.group}, ${s.cell}`, v: r.share, line: "cell" }, { ...r, who: `${r.group}, English only`, v: r.base, line: "English only" }]);
      const groups = ["same family", "other family"];
      return [plot({
        height: 300, x: { domain: LADDER, label: "model size" },
        y: { label: "share of untrained-language tasks above chance", grid: true, tickFormat: "%" },
        color: { domain: groups, range: S.series(2), legend: true },
        marks: [
          Plot.line(rows.filter((r) => r.line === "cell"), { x: "size", y: "v", z: "who", stroke: "group", strokeWidth: 2, sort: (r) => LADDER.indexOf(r.size) }),
          Plot.line(rows.filter((r) => r.line !== "cell"), { x: "size", y: "v", z: "who", stroke: "group", strokeWidth: 1.5, strokeDasharray: "4 3", sort: (r) => LADDER.indexOf(r.size) }),
          Plot.dot(rows.filter((r) => r.line === "cell"), { x: "size", y: "v", fill: "group", r: 4, tip: true,
            title: (r) => `${r.group} (${s.level === "sub" ? "subfamily" : "top-level family"}), ${r.size}\n${CNAME[s.cell] ?? s.cell}: ${pct(r.share)} of ${r.n} tasks in ${r.langs} languages above chance\nEnglish only, same tasks: ${pct(r.base)}` }),
        ],
      }), note("Solid: the cell; dashed: the English-only L1 cell on exactly the same tasks. The gap between a solid line and its dashed twin is what training the cell's languages adds; same family = the untrained language shares the family of a trained non-English language.")];
    });
  };
})();

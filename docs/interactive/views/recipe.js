/* Additional interactive views of the recipe page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, css, fmt, pct, mount, note, tiles, uniq } = S;
  const FORMAT = { original: "as published", rf: "RF", rfgm: "LLM-RF" };
  const SCORING = { acc: "accuracy", bbpb: "bBPB" };
  const variantLabel = (r) => `${FORMAT[r.format] ?? r.format} · ${SCORING[r.scoring] ?? r.scoring}`;

  /* Choose a language and a proxy size: per benchmark, the version (format and
     scoring) whose DA-size against the 1.7B model is highest there, and
     whether it is reliable from that size on. */
  S.VIEWS.recipeCards = async (el) => {
    await S.languages();
    const [t, rec] = await Promise.all([S.load("recipe_tasks"), S.load("recipe")]);
    const rows = t.data.map((r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i]])));
    const sizes = t.sizes;
    const tau = rec.tau;
    const name = (b) => t.names[b] ?? rec.names?.[b] ?? b;
    const langs = uniq(rows.map((r) => r.language)).sort((a, b) => S.langName(a).localeCompare(S.langName(b)));
    mount(el, [
      { key: "lang", label: "Your language", options: langs.map((l) => ({ value: l, label: `${S.langName(l)} (${l})` })), value: langs.includes("de") ? "de" : langs[0], string: true },
      { key: "size", label: "Your proxy size", type: "seg", options: sizes, value: "350M" },
      { key: "target", label: "bBPB checked against the 1.7B", type: "seg", options: [{ value: "bbpb_acc", label: "accuracy (same truth)" }, { value: "bbpb_bbpb", label: "bBPB (easier)" }] },
    ], (s) => {
      const zi = sizes.indexOf(s.size);
      const mine = rows.filter((r) => r.language === s.lang && (r.reading === "acc_acc" || r.reading === s.target));
      const benches = uniq(mine.map((r) => r.benchmark));
      const cards = benches.map((b) => {
        // one candidate per version: the mean over the benchmark's tasks in this language (most have one)
        const cands = Object.values(mine.filter((r) => r.benchmark === b).reduce((m, r) => {
          const k = `${r.format}|${r.scoring}`;
          (m[k] ??= { format: r.format, scoring: r.scoring, das: [], ranks: [] });
          if (r[s.size] != null) m[k].das.push(r[s.size]);
          if (r.safe_rank != null) m[k].ranks.push(r.safe_rank);
          return m;
        }, {})).map((c) => ({
          ...c, da: c.das.length ? c.das.reduce((a, x) => a + x, 0) / c.das.length : null,
          safe: c.ranks.length ? Math.max(...c.ranks) : null, n: c.das.length,
        }));
        cands.forEach((c) => { c.reliable = c.safe != null && c.safe <= zi && c.da != null; });
        const scored = cands.filter((c) => c.da != null)
          .sort((x, y) => (y.reliable - x.reliable) || (y.da - x.da));
        const overall = rec.recommendation.find((r) => r.benchmark === b && r.bbpb_reading === s.target);
        return { b, best: scored[0] ?? null, cands: scored, nAll: cands.length, overall };
      }).sort((x, y) => ((y.best?.reliable ?? false) - (x.best?.reliable ?? false)) || ((y.best?.da ?? -1) - (x.best?.da ?? -1)));
      const withVal = cards.filter((c) => c.best);
      const reliable = withVal.filter((c) => c.best.reliable);
      const bpbPick = withVal.filter((c) => c.best.scoring === "bbpb");
      const bar = (c) => {
        const w = Math.max(0, Math.min(1, c.da));
        const color = c.reliable ? css("--viz-good") : c.da > 0.5 ? css("--viz-s1") : css("--viz-muted");
        return h("div", { style: "display:grid;grid-template-columns:8.5em 1fr 2.6em;gap:.4em;align-items:center;font-size:.64rem" },
          h("span", { title: c.n > 1 ? `mean over ${c.n} tasks` : null }, variantLabel(c)),
          h("span", { style: `position:relative;height:8px;border-radius:4px;background:${css("--viz-grid")}` },
            h("span", { style: `position:absolute;left:0;top:0;bottom:0;width:${100 * w}%;border-radius:4px;background:${color}` }),
            h("span", { style: `position:absolute;left:${100 * tau}%;top:-3px;bottom:-3px;width:2px;background:${css("--md-default-fg-color")}`, title: `reliable: DA-size ≥ ${tau}` }),
            h("span", { style: `position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:${css("--viz-muted")}`, title: "0.5: chance" })),
          h("span", { class: "num", style: "text-align:right;font-variant-numeric:tabular-nums" }, fmt(c.da)));
      };
      const verdict = (c) => {
        if (!c.best) return h("span", { class: "badge avoid" }, "at chance at this size: no version reads it");
        const from = c.best.safe != null && c.best.safe < 5 ? sizes[c.best.safe] : null;
        if (c.best.reliable) return h("span", { class: "badge use" }, `✓ reliable from ${from} on`);
        if (c.best.da > 0.5) return h("span", { class: "badge care" }, from ? `△ not yet: reliable from ${from}` : "△ above chance, never reliable");
        return h("span", { class: "badge avoid" }, "✗ at chance");
      };
      return [
        tiles([
          [withVal.length, `benchmarks with a reading in ${S.langName(s.lang)} at ${s.size}`],
          [reliable.length, `with a version reliable from ${s.size} on`],
          [bpbPick.length, "where a bBPB version reads best"],
          [benches.length - withVal.length, "at chance at this size in every version"],
        ]),
        h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(17em,1fr));gap:.8em;margin-top:.8em" },
          cards.map((c) => h("div", { class: "viz-card", style: "margin:0" },
            h("h4", {}, name(c.b)),
            c.best ? h("div", { style: "font-size:.8rem;margin-bottom:.3em" }, "Evaluate it ", h("b", {}, variantLabel(c.best))) : null,
            verdict(c),
            h("div", { style: "display:grid;gap:.25em;margin-top:.5em" }, c.cands.map(bar)),
            c.overall ? h("p", { class: "viz-note" }, `Over all languages: ${c.overall.variant.split(" → ")[0]} (mean DA-size ${fmt(c.overall.mean_da_size)}, ` +
              `${pct(c.overall.safe_by_1B)} of ${c.overall.languages_ranked} languages reliable by 1B).`) : null))),
        note(`Each card is one benchmark in ${S.langName(s.lang)}. A bar is the DA-size of one version at ${s.size}: how often that proxy orders a pair of training choices as the 1.7B model does ` +
          `(black tick: the reliability threshold ${tau}; grey tick: chance). Reliable means DA-size ≥ ${tau} at ${s.size} and at every larger proxy up to 1B. ` +
          `The pick is the reliable version with the highest DA-size, or the highest DA-size when none is; it is read on one language and one size, so it is noisier than the benchmark-level table above.`),
      ];
    });
  };
})();

/* Additional interactive views of the design-decisions page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, load, mount, note, fmt } = S;
  const PROX = ["90M", "175M", "350M", "600M", "1B"];
  const FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const chin = (f) => `${+(f * 5).toFixed(1)}C`;          // a tenth of the run → Chinchilla multiples (5C = the full run)
  const same = (a, b) => Math.abs(a - b) < 1e-6;
  const POPS = [{ value: "bpb_trained", label: "per-language BPB" }, { value: "benchmark", label: "benchmarks" }];
  const POP_LONG = { bpb_trained: "the BPB of the languages both levels train", benchmark: "the benchmark tasks above chance" };
  const UNMATCHED = "600M shallow is not size-matched to 600M deep (+3.7 % non-embedding parameters, width 2048 vs 1536, about +15 % compute); the pairs are kept and flagged";

  // Page-scoped styles: the grid of decisions × proxy sizes both views share (added on first use).
  const style = () => document.getElementById("dd-style") || document.head.append(h("style", { id: "dd-style" }, `
.dd-grid { display: grid; grid-template-columns: minmax(9.5em, 1.7fr) repeat(var(--dd-n), minmax(2.7em, 1fr)) minmax(7.5em, 1.5fr); align-items: center; gap: .15em .3em; font-size: .68rem; min-width: 34em; }
.dd-grid.clock { grid-template-columns: minmax(9.5em, 1.7fr) repeat(var(--dd-n), minmax(4.6em, 1fr)); }
.dd-hd { color: var(--viz-muted); font-weight: 600; text-align: center; padding-bottom: .2em; }
.dd-hd.left { text-align: left; }
.dd-lab { line-height: 1.25; padding: .25em 0; border-top: 1px solid var(--viz-grid); }
.dd-c { display: flex; flex-direction: column; align-items: center; gap: .1em; border-top: 1px solid var(--viz-grid); padding: .25em 0; min-height: 2.6em; justify-content: center; }
.dd-dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--viz-s1); background: var(--viz-surface); box-sizing: border-box; }
.dd-dot.on { background: var(--viz-s1); }
.dd-c small { color: var(--viz-muted); font-variant-numeric: tabular-nums; }
.dd-c sup { color: var(--viz-warn); font-weight: 700; }
.dd-span { grid-column: 2 / span var(--dd-n); color: var(--viz-muted); font-style: italic; border-top: 1px solid var(--viz-grid); padding: .45em 0; text-align: center; }
.dd-bar { position: relative; height: 12px; background: var(--viz-neutral); border-radius: 0 4px 4px 0; }
.dd-bar b { position: absolute; left: 0; top: 0; bottom: 0; background: var(--viz-s1); border-radius: 0 4px 4px 0; }
.dd-bar i { position: absolute; top: -3px; bottom: -3px; border-left: 1px dashed var(--viz-muted); }
.dd-eff { border-top: 1px solid var(--viz-grid); padding: .25em 0; display: flex; flex-direction: column; gap: .15em; min-height: 2.6em; justify-content: center; }
.dd-eff span { color: var(--viz-muted); font-variant-numeric: tabular-nums; }
.dd-axis { position: relative; height: 1.1em; color: var(--viz-muted); }
.dd-axis span { position: absolute; transform: translateX(-50%); font-size: .6rem; }
.dd-tl { display: flex; gap: 2px; }
.dd-tl i { width: 5px; height: 12px; border-radius: 2px; border: 1px solid var(--viz-s1); box-sizing: border-box; }
.dd-tl i.on { background: var(--viz-s1); }
.dd-c.ref .dd-tl i { border-color: var(--viz-s7); }
.dd-c.ref .dd-tl i.on { background: var(--viz-s7); }
.dd-key { display: flex; flex-wrap: wrap; gap: .3em 1.2em; color: var(--viz-muted); font-size: .66rem; margin: .5em 0 0; align-items: center; }
.dd-key .dd-dot, .dd-key .dd-tl { display: inline-flex; vertical-align: -3px; margin-right: .3em; }
`));

  // log scale of the effect bar: 0.5× … 8× of the seed sd
  const LX = (v) => Math.max(0, Math.min(1, Math.log(v / 0.5) / Math.log(16))) * 100;

  async function data() {
    style();
    const d = await load("design_decisions_ladder");
    const at = (key, pop, size, f) => d.rows.find((r) => r.key === key && r.pop === pop && r.size === size && same(r.frac, f));
    const eff = (key, pop) => d.effect.find((e) => e.key === key && e.pop === pop);
    // rows of a population, ordered by the effect at the reference (largest first; unmeasured last)
    const keys = (pop) => {
      // a data build read on benchmarks only (L1, L2) has no row on BPB; the activation and optimizer ladders stay
      // listed without a row (Muon has no 1.7B evaluation yet), with why
      const ks = Object.keys(d.names).filter((k) => k === "optimizer" || k === "activation" || d.rows.some((r) => r.key === k && r.pop === pop));
      return ks.sort((a, b) => (eff(b, pop)?.ratio ?? -1) - (eff(a, pop)?.ratio ?? -1)
        || (d.rows.some((r) => r.key === b && r.pop === pop) - d.rows.some((r) => r.key === a && r.pop === pop)));
    };
    const muon = (f) => d.ckpt.filter((r) => r.key === "optimizer" && same(r.frac, f));
    return { d, at, eff, keys, muon };
  }

  const tip = (d, r, pop) => `${d.names[r.key].short}, ${r.size} at ${chin(r.frac)}${same(r.frac, 1) ? " (final)" : ""}\n`
    + `DA ${fmt(r.da)} against the 1.7B final, on ${POP_LONG[pop]}`
    + (r.cells ? `, mean over ${r.cells} language setting${r.cells > 1 ? "s" : ""}` : "")
    + `\n${r.items} items${r.source === "rq02" ? " · seed 1904, pooled over tasks (decision-accuracy analysis, per-axis table)" : ""}`
    + (r.size_matched ? "" : `\n† ${UNMATCHED}`);

  function missing(key, pop, m) {
    if (key === "optimizer") {
      const v = m.muon(0.9).sort((x, y) => PROX.indexOf(x.size) - PROX.indexOf(y.size));
      const lo = Math.min(...v.map((r) => r.da)), hi = Math.max(...v.map((r) => r.da));
      return `Muon has no 1.7B evaluation in this snapshot (${v.map((r) => r.size).join(", ")} evaluated), so no proxy can be read against the reference. `
        + `Against their own final checkpoint the Muon proxies agree at ${fmt(lo)}–${fmt(hi)} by 4.5C.`;
    }
    if (key === "activation" && pop === "bpb_trained") return "No per-language BPB pair for SwiGLU at 1.7B yet; switch to benchmarks.";
    return "Not read on this measurement.";
  }

  S.VIEWS.ddLadder = async (el) => {
    const m = await data();
    mount(el, [
      { key: "pop", label: "Read the decision on", type: "seg", options: POPS },
      { key: "thr", label: "DA threshold", type: "range", min: 0.55, max: 0.95, step: 0.05, value: 0.75 },
      { key: "f", label: "Proxy checkpoint", options: FRACS.map((f) => ({ value: f, label: same(f, 1) ? "5C (final)" : chin(f) })), value: 1 },
    ], (s) => {
      const keys = m.keys(s.pop);
      const grid = h("div", { class: "dd-grid", style: `--dd-n:${PROX.length}` },
        h("div", { class: "dd-hd left" }, "Decision"),
        PROX.map((p) => h("div", { class: "dd-hd" }, p)),
        h("div", { class: "dd-hd left" }, "Effect at 1.7B ÷ seed sd",
          h("div", { class: "dd-axis" }, [0.5, 1, 2, 4, 8].map((v) => h("span", { style: `left:${LX(v)}%` }, `${v}×`)))));
      let cleared = 0, total = 0;
      for (const k of keys) {
        const nm = m.d.names[k];
        grid.append(h("div", { class: "dd-lab", title: nm.long }, nm.short));
        const cells = PROX.map((p) => m.at(k, s.pop, p, s.f));
        if (cells.every((c) => !c)) grid.append(h("div", { class: "dd-span" }, missing(k, s.pop, m)));
        else for (const c of cells) {
          if (!c) { grid.append(h("div", { class: "dd-c", title: "no value" }, h("small", {}, "–"))); continue; }
          total++; if (c.da >= s.thr) cleared++;
          grid.append(h("div", { class: "dd-c", title: tip(m.d, c, s.pop) },
            h("i", { class: c.da >= s.thr ? "dd-dot on" : "dd-dot" }),
            h("small", {}, fmt(c.da), c.size_matched ? null : h("sup", {}, "†"))));
        }
        const e = m.eff(k, s.pop);
        grid.append(e
          ? h("div", { class: "dd-eff", title: `Median |effect| at the 1.7B reference in seed standard deviations, per L: ${e.Ls.map((L, i) => `L${L} ${fmt(e.per_L[i])}× (${S.pct(e.share_above_2[i])} of items above 2×)`).join(", ")}` },
            h("div", { class: "dd-bar" }, h("b", { style: `width:${LX(e.ratio)}%` }), h("i", { style: `left:${LX(1)}%` }), h("i", { style: `left:${LX(2)}%` })),
            h("span", {}, `${fmt(e.ratio, 1)}×`))
          : h("div", { class: "dd-eff" }, h("span", {}, "not measured")));
      }
      return [h("div", { class: "viz-scroll" }, grid),
        h("p", { class: "dd-key" },
          h("span", {}, h("i", { class: "dd-dot on" }), `clears ${fmt(s.thr)}`),
          h("span", {}, h("i", { class: "dd-dot" }), "does not"),
          h("span", {}, "† 600M depth pair not size-matched"),
          h("span", {}, "dashed lines: 1× and 2× the seed sd")),
        note(`${cleared} of ${total} cells clear ${fmt(s.thr)} at ${same(s.f, 1) ? "the proxies' final checkpoint" : chin(s.f)}. `
          + "Rows run from the largest effect at the reference to the smallest; a decision whose bar ends near 1× is about as large as re-running the same recipe with another seed."),
      ];
    });
  };

  S.VIEWS.ddClock = async (el) => {
    const m = await data();
    const cols = [...PROX, "1.7B"];
    mount(el, [
      { key: "pop", label: "Read the decision on", type: "seg", options: POPS },
      { key: "thr", label: "DA threshold", type: "range", min: 0.55, max: 0.95, step: 0.05, value: 0.75 },
    ], (s) => {
      const grid = h("div", { class: "dd-grid clock", style: `--dd-n:${cols.length}` },
        h("div", { class: "dd-hd left" }, "Decision"),
        cols.map((p) => h("div", { class: "dd-hd" }, p === "1.7B" ? "1.7B own run" : p)));
      for (const k of m.keys(s.pop)) {
        const nm = m.d.names[k];
        grid.append(h("div", { class: "dd-lab", title: nm.long }, nm.short));
        const series = cols.map((p) => FRACS.map((f) => m.at(k, s.pop, p, f)).filter(Boolean));
        if (series.every((x) => !x.length)) { grid.append(h("div", { class: "dd-span" }, missing(k, s.pop, m))); continue; }
        series.forEach((pts, i) => {
          if (!pts.length) { grid.append(h("div", { class: "dd-c" }, h("small", {}, "–"))); return; }
          // the earliest checkpoint from which every later one clears the threshold
          let from = null;
          for (let j = pts.length - 1; j >= 0 && pts[j].da >= s.thr; j--) from = pts[j].frac;
          const ref = cols[i] === "1.7B";
          grid.append(h("div", {
            class: ref ? "dd-c ref" : "dd-c",
            title: `${nm.short}, ${cols[i]}${ref ? " (the reference's own checkpoints against its final)" : ""}\n`
              + pts.map((r) => `${chin(r.frac)} ${fmt(r.da)}`).join(" · ") + (pts.some((r) => !r.size_matched) ? `\n† ${UNMATCHED}` : ""),
          },
          h("span", { class: "dd-tl" }, pts.map((r) => h("i", { class: r.da >= s.thr ? "on" : null }))),
          h("small", {}, from == null ? "never" : `from ${chin(from)}`, pts.some((r) => !r.size_matched) ? h("sup", {}, "†") : null)));
        });
      }
      return [h("div", { class: "viz-scroll" }, grid),
        h("p", { class: "dd-key" },
          h("span", {}, h("span", { class: "dd-tl" }, h("i", { class: "on" }), h("i", {})), `one tick per checkpoint, 0.5C → 5C; filled = DA ≥ ${fmt(s.thr)}`),
          h("span", {}, "1.7B own run: its checkpoints 0.5C → 4.5C against its own final (DA-ckpt)")),
        note("\"from 2C\" means the proxy agrees with the reference from 40 % of its run to the end. A proxy column that says \"never\" while the 1.7B column fills up means the decision exists at the reference but no small model sees it."),
      ];
    });
  };
})();

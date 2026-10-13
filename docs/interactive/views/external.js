/* Additional interactive views of the external page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { load, mount, note, fmt, tiles, plot, css, uniq } = S;

  // Slopegraph: each shared English task's SNR rank on our ladder (left) and on DataDecide (right).
  S.VIEWS.extSlope = async (el) => {
    const d = await load("external_ranks");
    const pairs = d.pairs.map((p) => p.split("|"));
    const rAt = (v, a, b) => d.r.find((x) => x.variant === v && x.ours === a && x.theirs === b);
    const variants = uniq(d.snr.map((x) => x.variant))
      .sort((a, b) => (rAt(b, "1B", "1B")?.r ?? -2) - (rAt(a, "1B", "1B")?.r ?? -2));
    const tasks = Object.keys(d.names);
    mount(el, [
      { key: "v", label: "SNR definition (r at 1B ↔ 1B)", options: variants.map((v) => ({ value: v, label: `${d.titles[v] ?? v}${v === d.headline.variant ? " — headline" : ""} (r ${fmt(rAt(v, "1B", "1B")?.r)})` })), value: d.headline.variant, string: true },
      { key: "p", label: "Our size ↔ DataDecide size", type: "seg", options: pairs.map(([a, b]) => ({ value: `${a}|${b}`, label: `${a} ↔ ${b}` })), value: "1B|1B" },
      { key: "y", label: "Position by", type: "seg", options: [{ value: "rank", label: "rank" }, { value: "snr", label: "SNR (log)" }] },
    ], (s) => {
      const [a, b] = s.p.split("|");
      const rows = d.snr.filter((x) => x.variant === s.v && x.ours === a && x.theirs === b);
      const rank = (key) => Object.fromEntries([...rows].sort((x, y) => y[key] - x[key]).map((x, i) => [x.task, i + 1]));
      const ra = rank("a"), rb = rank("b");
      const n = rows.length;
      const pts = rows.flatMap((x) => [
        { task: x.task, side: "our ladder", rank: ra[x.task], snr: x.a, name: d.names[x.task] },
        { task: x.task, side: "DataDecide", rank: rb[x.task], snr: x.b, name: d.names[x.task] }]);
      const links = rows.map((x) => ({ task: x.task, name: d.names[x.task], r1: ra[x.task], r2: rb[x.task], s1: x.a, s2: x.b }));
      const rr = rAt(s.v, a, b);
      // Spearman ρ of the two orders (no ties among these SNRs): 1 − 6Σd² / (n(n² − 1))
      const rho = n >= 3 ? 1 - (6 * d3.sum(links, (l) => (l.r1 - l.r2) ** 2)) / (n * (n * n - 1)) : null;
      const byRank = s.y === "rank";
      const yk = byRank ? "rank" : "snr";
      const col = { domain: tasks, range: S.series(tasks.length) };
      const tip = (p) => `${p.name}: rank ${p.rank} of ${n} on ${p.side}, SNR ${fmt(p.snr)}`;
      const chart = plot({
        height: byRank ? 70 + 40 * n : 360, width: 560, marginLeft: byRank ? 150 : 60, marginRight: 150, marginTop: 30,
        x: { domain: ["our ladder", "DataDecide"], label: null, axis: "top", padding: 0.1 },
        y: byRank ? { type: "point", domain: d3.range(1, n + 1), label: null, axis: null }
          : { type: "log", label: "SNR", grid: true },
        color: { ...col, legend: !byRank },
        marks: [
          Plot.link(links, { x1: () => "our ladder", x2: () => "DataDecide", y1: byRank ? "r1" : "s1", y2: byRank ? "r2" : "s2", stroke: "task", strokeWidth: 2, strokeOpacity: 0.8 }),
          Plot.dot(pts, { x: "side", y: yk, fill: "task", stroke: css("--viz-surface"), strokeWidth: 2, r: 6, tip: true, title: tip }),
          ...(byRank ? [
            Plot.text(pts.filter((p) => p.side === "our ladder"), { x: "side", y: "rank", text: (p) => `${p.name}  ${fmt(p.snr, 1)}`, textAnchor: "end", dx: -10, fill: css("--md-default-fg-color") }),
            Plot.text(pts.filter((p) => p.side === "DataDecide"), { x: "side", y: "rank", text: (p) => `${fmt(p.snr, 1)}  ${p.name}`, textAnchor: "start", dx: 10, fill: css("--md-default-fg-color") }),
          ] : []),
        ],
      });
      const moved = links.filter((l) => l.r1 !== l.r2).length;
      return [
        tiles([[n, "shared tasks above chance"], [fmt(rr?.r), "Pearson r of log SNR"], [fmt(rho), "Spearman ρ of the order"], [fmt(d3.median(links, (l) => l.s2 / l.s1), 1) + "×", "DataDecide SNR ÷ ours (median)"]]),
        chart,
        note(`${n - moved} of ${n} tasks keep their rank. ${byRank ? "Top = the highest SNR on that corpus; the numbers next to the names are the SNR values." : (links.every((l) => l.s2 > l.s1) ? "On one log axis every line rises: DataDecide's SNR is larger for every task, " : "On one log axis the levels differ by corpus, ") + "so compare the orders, not the levels."}`
          + (n < tasks.length ? ` At this size ${tasks.filter((t) => !rows.some((x) => x.task === t)).map((t) => d.names[t]).join(" and ")} ${tasks.length - n > 1 ? "are" : "is"} at chance on our side and leave${tasks.length - n > 1 ? "" : "s"} the comparison.` : "")),
      ];
    });
  };
})();

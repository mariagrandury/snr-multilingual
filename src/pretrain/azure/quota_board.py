#!/usr/bin/env python3.11
"""
quota_board.py — one-page H100/H200 quota status: terminal report + shareable PNG.

Called by `quota_status.sh board`. Answers the three questions the raw CLI
cannot answer on its own:

  1. What has actually been GRANTED?      -> Azure ML per-family dedicated cores
  2. What did we ASK for, and is it live? -> support tickets (subType BatchAml)
  3. Where COULD we ask?                  -> SKU allow-list x Azure ML presence

(3) is the one that matters most: a region where the SKU is restricted
`NotAvailableForSubscription` will never approve no matter how often it is
filed, and a region without Azure ML cannot run these jobs even if it would.
Cross-referencing the three is what turns "12 tickets, all Open" into a list
of which tickets are futile and which regions are still worth filing in.

Read-only: every call is a GET / list. Nothing here creates or changes state.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUB = "ef1ff20e-1168-4846-a78e-47d102dd35f6"
ML_API = "2024-10-01"

# The three shapes we care about, each tying a VM SKU to the Azure ML quota
# family a ticket for it lands in. Names differ between the two systems.
GROUPS = {
    "ND-H100": dict(skus=["Standard_ND96isr_H100_v5"],
                    family="standardndv5h100family",
                    label="ND96isr_H100_v5\n8xH100 SXM + IB"),
    "ND-H200": dict(skus=["Standard_ND96isr_H200_v5"],
                    family="standardndisrh200v5family",
                    label="ND96isr_H200_v5\n8xH200 SXM + IB"),
    "NC-H100": dict(skus=["Standard_NC80adis_H100_v5", "Standard_NC40ads_H100_v5"],
                    family="standardncadsh100v5family",
                    label="NC40/80ads_H100_v5\nH100 PCIe, no IB"),
}
FAMILY_TO_GROUP = {g["family"]: k for k, g in GROUPS.items()}

OK, BLOCKED, ABSENT = "ok", "blocked", "absent"


def az(*args):
    """Run an az command, return parsed JSON or None if it failed/was empty."""
    p = subprocess.run(["az", *args], capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def fetch_skus():
    """{region: {sku_name: ok|blocked}} for every H100/H200 SKU."""
    out = {}
    for size in ("H100", "H200"):
        for r in az("vm", "list-skus", "--size", size, "--all", "-o", "json") or []:
            loc = (r.get("locations") or ["?"])[0].lower()
            blocked = any(x["reasonCode"] == "NotAvailableForSubscription"
                          for x in (r.get("restrictions") or []))
            out.setdefault(loc, {})[r["name"]] = BLOCKED if blocked else OK
    return out


def fetch_aml(region):
    """Per-family dedicated limits + regional envelope, or None if no AML here."""
    data = az("rest", "--method", "get", "--url",
              f"https://management.azure.com/subscriptions/{SUB}/providers/"
              f"Microsoft.MachineLearningServices/locations/{region}/usages"
              f"?api-version={ML_API}", "-o", "json")
    if not data:
        return None
    fams, envelope = {}, None
    for u in data.get("value", []):
        name = u["name"]["value"]
        if name == "TotalDedicatedCores":
            envelope = (u["currentValue"], u["limit"])
        elif "dedicatedCores" in u["type"]:
            fams[name.lower()] = (u["currentValue"], u["limit"])
    return dict(families=fams, envelope=envelope)


def fetch_tickets():
    """One row per quota CHANGE REQUEST, not per ticket.

    A single ticket can carry several quotaChangeRequests (the 2026-08-25 batch
    filed ND and NC in the same ticket), and the description names only one of
    them — reading the description alone silently under-reports what was filed
    and invents "not yet filed" gaps. quotaTicketDetails is the real payload,
    and it is also the only place the Dedicated/LowPriority type appears.
    """
    data = az("rest", "--method", "get", "--url",
              f"https://management.azure.com/subscriptions/{SUB}/providers/"
              "Microsoft.Support/supportTickets?api-version=2020-04-01", "-o", "json")
    rows = []
    for t in (data or {}).get("value", []):
        p = t["properties"]
        common = dict(created=p.get("createdDate", "")[:10],
                      updated=p.get("modifiedDate", "")[:10],
                      status=p.get("status", ""), id=t["name"])
        reqs = (p.get("quotaTicketDetails") or {}).get("quotaChangeRequests") or []
        for q in reqs:
            try:
                pl = json.loads(q.get("payload") or "{}")
            except json.JSONDecodeError:
                continue
            rows.append(dict(region=(q.get("region") or "").lower(),
                             family=pl.get("VMFamily", ""),
                             cores=int(pl.get("NewLimit") or 0),
                             type=pl.get("Type", "Dedicated"), **common))
        if not reqs:   # older tickets without the structured payload
            m = re.search(r"Requesting (\d+) cores for VM Family - (\S+) in (\S+) region",
                          p.get("description") or "")
            if m:
                rows.append(dict(region=m.group(3).lower(), family=m.group(2),
                                 cores=int(m.group(1)), type="Dedicated", **common))
    return sorted(rows, key=lambda r: (r["region"], r["created"], r["family"]))


def group_state(skus_in_region, group):
    """ok if any SKU in the group is deployable, blocked if all are, else absent."""
    states = [skus_in_region.get(s) for s in GROUPS[group]["skus"]]
    if OK in states:
        return OK
    return BLOCKED if BLOCKED in states else ABSENT


def collect(cache=None, offline=False):
    """Gather the three raw views, caching them so re-renders are instant.

    Each `az` invocation costs seconds and there are ~45 of them, so a full
    refresh is minutes. The cache exists to iterate on the report, not to
    report stale numbers — `board` refreshes by default.
    """
    if offline:
        if not (cache and os.path.exists(cache)):
            sys.exit(f"--offline needs a cache at {cache}; run once without it first")
        raw = json.load(open(cache))
        skus, tickets, aml = raw["skus"], raw["tickets"], raw["aml"]
        print(f"using cached fetch from {raw['stamp']}")
    else:
        t0 = time.time()
        skus = fetch_skus()
        print(f"  SKU catalog: {len(skus)} regions ({time.time() - t0:.0f}s)")
        t1 = time.time()
        tickets = fetch_tickets()
        print(f"  tickets: {len(tickets)} ({time.time() - t1:.0f}s)")
        t2 = time.time()
        regions = sorted(set(skus) | {t["region"] for t in tickets})
        with ThreadPoolExecutor(max_workers=12) as pool:
            aml = dict(zip(regions, pool.map(fetch_aml, regions)))
        print(f"  Azure ML usage: {len(regions)} regions ({time.time() - t2:.0f}s)")
        if cache:
            json.dump(dict(skus=skus, tickets=tickets, aml=aml,
                           stamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
                      open(cache, "w"))

    filed = {}
    for t in tickets:
        g = FAMILY_TO_GROUP.get(t["family"].lower())
        filed.setdefault(t["region"], {}).setdefault(g, []).append(t)

    rows = []
    for loc in sorted(set(skus) | {t["region"] for t in tickets}):
        states = {g: group_state(skus.get(loc, {}), g) for g in GROUPS}
        if all(s == ABSENT for s in states.values()):
            continue
        rows.append(dict(region=loc, states=states, aml=aml.get(loc),
                         filed=filed.get(loc, {})))
    return rows, tickets, aml


def granted(row, group):
    """Cores granted for this group in this region (None if AML absent)."""
    if not row["aml"]:
        return None
    return row["aml"]["families"].get(GROUPS[group]["family"], (0, 0))[1]


def verdict(row, group):
    """Why a ticket here can or cannot ever be approved."""
    if not row["aml"]:
        return "no Azure ML in region"
    state = row["states"][group]
    if state == BLOCKED:
        return "SKU blocked for subscription"
    if state == ABSENT:
        return "SKU not offered here"
    return "viable"


# ----------------------------------------------------------------- text report
def report(rows, tickets):
    by_region = {r["region"]: r for r in rows}
    print(f"\n=== Open quota requests ({len(tickets)}) ===")
    print(f"  {'ticket':<10}{'region':<19}{'group':<10}{'cores':>6}  {'type':<12}"
          f"{'created':<11}{'granted':>8}  verdict")
    futile = []
    for t in tickets:
        g = FAMILY_TO_GROUP.get(t["family"].lower(), "?")
        row = by_region.get(t["region"])
        v = verdict(row, g) if row and g in GROUPS else "unknown SKU group"
        got = granted(row, g) if row and g in GROUPS else None
        if v != "viable":
            futile.append((t, g, v))
        print(f"  {t['id'][:8]:<10}{t['region']:<19}{g:<10}{t['cores']:>6}  "
              f"{t['type']:<12}{t['created']:<11}{'-' if got is None else got:>8}  {v}")
    kinds = sorted({t["type"] for t in tickets})
    print(f"  request types filed: {', '.join(kinds)}"
          + ("   <- no Spot/low-priority request has ever been filed"
             if kinds == ["Dedicated"] else ""))

    print("\n=== Where these SKUs can be deployed (allow-list x Azure ML) ===")
    print(f"  {'region':<19}" + "".join(f"{g:<11}" for g in GROUPS)
          + f"{'envelope':>9}  filed")
    for r in sorted(rows, key=lambda r: (r["aml"] is None, r["region"])):
        cells = "".join(f"{r['states'][g]:<11}" for g in GROUPS)
        env = r["aml"]["envelope"][1] if r["aml"] and r["aml"]["envelope"] else "-"
        aml_note = "" if r["aml"] else "  (no Azure ML)"
        print(f"  {r['region']:<19}{cells}{env:>9}  "
              f"{','.join(sorted(r['filed'])) or '-'}{aml_note}")

    gaps = [(r["region"], g) for r in rows if r["aml"]
            for g in GROUPS if r["states"][g] == OK and g not in r["filed"]]
    print(f"\n=== Deployable but NOT yet filed ({len(gaps)}) ===")
    for loc, g in gaps:
        print(f"  {loc:<19}{g}")
    # NOT "cancel these": NotAvailableForSubscription is an allow-list state
    # support CAN lift, and the quota ticket is the channel for asking. They
    # just will not approve while framed as a capacity ask.
    print(f"\n=== Filed against a blocked SKU ({len(futile)}) — "
          f"needs SKU ENABLEMENT, not more cores ===")
    for t, g, v in futile:
        print(f"  {t['region']:<19}{g:<10}{t['cores']:>6} cores  {v}  [{t['id'][:8]}]")
    return gaps, futile


# ------------------------------------------------------------------------- png
CLR = {OK: ("#c8e6c9", "#1b5e20"), BLOCKED: ("#ffcdd2", "#b71c1c"),
       ABSENT: ("#eeeeee", "#9e9e9e")}
# Deployable SKU but no Azure ML in the region: quota there would be unusable,
# so it must not read as green/good.
NO_AML_CLR = ("#ffe0b2", "#e65100")
TEXT = {OK: "deployable", BLOCKED: "blocked", ABSENT: "not offered"}


ROW = 0.26          # inches per table row — the unit the whole page is laid out in
MARGIN = 0.45


class Page:
    """Single-axes page laid out with a top-down cursor measured in ROW units.

    gridspec fought us here: sub-axes stretch to their allotted box, so every
    panel got a different row pitch and large dead gaps. One axes plus an
    explicit cursor keeps the pitch identical everywhere.
    """

    def __init__(self, units, width=13.0):
        self.units, self.width = units, width
        self.fig = plt.figure(figsize=(width, units * ROW + 2 * MARGIN))
        self.ax = self.fig.add_axes([MARGIN / width, MARGIN / (units * ROW + 2 * MARGIN),
                                     1 - 2 * MARGIN / width,
                                     1 - 2 * MARGIN / (units * ROW + 2 * MARGIN)])
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 0.0

    def _at(self, extra=0.0):
        return 1.0 - (self.y + extra) / self.units

    def text(self, s, size=9, weight="normal", color="#222222", x=0.0, advance=1.0):
        self.ax.text(x, self._at(0.7), s, fontsize=size, fontweight=weight,
                     color=color, va="center")
        self.y += advance

    def table(self, headers, rows, widths):
        x0 = [sum(widths[:i]) for i in range(len(widths))]
        for i, h in enumerate(headers):
            self.ax.text(x0[i], self._at(0.65), h, fontsize=8.5,
                         fontweight="bold", color="#555555", va="center")
        self.y += 1.05
        for row in rows:
            yc = self._at(0.5)
            for i, cell in enumerate(row):
                text, face, fg = (cell if isinstance(cell, tuple)
                                  else (cell, None, "#222222"))
                if face:
                    self.ax.add_patch(plt.Rectangle(
                        (x0[i] - 0.004, yc - 0.34 / self.units),
                        widths[i] - 0.014, 0.72 / self.units,
                        facecolor=face, edgecolor="none",
                        transform=self.ax.transAxes, zorder=0))
                self.ax.text(x0[i], yc, text, fontsize=8.5, color=fg,
                             va="center", zorder=1)
            self.y += 1.0


def render(rows, tickets, gaps, futile, out):
    by_region = {r["region"]: r for r in rows}
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    trows = []
    for t in tickets:
        g = FAMILY_TO_GROUP.get(t["family"].lower(), "?")
        row, v = by_region.get(t["region"]), "unknown"
        if row and g in GROUPS:
            v = verdict(row, g)
        got = granted(row, g) if row and g in GROUPS else None
        state = OK if v == "viable" else BLOCKED
        # Short ticket id: enough to quote back to Azure support / match the
        # portal, without spending a third of the row on a full GUID.
        trows.append([t["id"][:8], t["region"], g, str(t["cores"]), t["type"], t["created"],
                      "-" if got is None else str(got),
                      (v, *CLR[state])])

    # Only regions that are actionable: something deployable, or a ticket on
    # them. The all-blocked majority is a footnote, not 25 rows of noise.
    shown = [r for r in rows if any(s == OK for s in r["states"].values()) or r["filed"]]
    hidden = len(rows) - len(shown)
    mrows = []
    for r in sorted(shown, key=lambda r: (r["aml"] is None, r["region"])):
        cells = [r["region"]]
        for g in GROUPS:
            s = r["states"][g]
            mark, clr = TEXT[s], CLR[s]
            if s == OK and not r["aml"]:
                mark, clr = "unusable - no Azure ML", NO_AML_CLR
            elif s == OK and g in r["filed"]:
                mark += "  (filed)"
            elif s == OK:
                mark += "  <- FILE"
            cells.append((mark, *clr))
        env = r["aml"]["envelope"][1] if r["aml"] and r["aml"]["envelope"] else None
        cells.append(str(env) if env is not None else "no Azure ML")
        mrows.append(cells)

    arows = [[(f"FILE     {loc:<20}{g}", *CLR[OK])] for loc, g in gaps]
    arows += [[(f"ENABLE   {t['region']:<20}{g}   {t['cores']} cores - "
                f"ask to enable the SKU, not for capacity", *CLR[BLOCKED])]
              for t, g, v in futile]
    if not arows:
        arows = [["nothing to change"]]

    units = 5.2 + (len(trows) + 2.1) + (len(mrows) + 3.2) + (len(arows) + 2.1)
    p = Page(units)

    p.text("Azure H100 / H200 quota status", size=19, weight="bold", advance=1.5)
    total = sum(granted(r, g) or 0 for r in rows for g in GROUPS)
    p.text(f"subscription MSNR ({SUB})   |   {stamp}   |   "
           f"{len(tickets)} open tickets   |   {total} cores granted so far",
           size=9.5, color="#666666", advance=2.0)

    kinds = sorted({t["type"] for t in tickets})
    p.text(f"Open quota requests ({len(tickets)})", size=13, weight="bold", advance=1.3)
    p.table(["ticket", "region", "SKU group", "cores", "type", "filed",
             "granted", "verdict"],
            trows, [0.09, 0.14, 0.10, 0.06, 0.09, 0.09, 0.07, 0.26])
    if kinds == ["Dedicated"]:
        p.text("Every request filed is type Dedicated - no Spot / low-priority "
               "request has ever been submitted.", size=8.5, color="#e65100",
               advance=1.6)
    else:
        p.y += 1.8

    p.text("Where these SKUs can be deployed   (subscription allow-list  x  Azure ML presence)",
           size=13, weight="bold", advance=1.3)
    p.table(["region", *[GROUPS[g]["label"].split("\n")[0] for g in GROUPS],
             "regional envelope"],
            mrows, [0.16, 0.19, 0.19, 0.19, 0.20])
    p.text(f"{hidden} further region(s) omitted: blocked or not offered for all three SKUs, "
           f"and no ticket filed.", size=8, color="#888888", advance=1.6)

    p.text(f"Actions:  {len(gaps)} region(s) still worth filing,  "
           f"{len(futile)} request(s) blocked on SKU enablement",
           size=13, weight="bold", advance=1.3)
    p.table(["action"], arows, [0.9])

    p.fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="quota_status.png", help="PNG output path")
    ap.add_argument("--no-png", action="store_true", help="terminal report only")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__),
                                                    ".quota_cache.json"))
    ap.add_argument("--offline", action="store_true",
                    help="re-render from the cached fetch (no az calls)")
    a = ap.parse_args()

    rows, tickets, aml = collect(a.cache, a.offline)
    if not rows:
        sys.exit("no SKU data returned — check `az login` and the subscription")
    gaps, futile = report(rows, tickets)
    if not a.no_png:
        render(rows, tickets, gaps, futile, a.out)

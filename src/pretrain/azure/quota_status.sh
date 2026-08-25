#!/usr/bin/env bash
# =============================================================================
# quota_status.sh — inspect, track and file Azure GPU quota requests
# =============================================================================
#
# WHY THIS EXISTS
#   Quota lives in two different systems and neither is visible from one place:
#     1. Microsoft.Quota  — the "Quotas" blade / self-service requests. Auto-
#                           approves when capacity exists, otherwise records a
#                           Failed request WITH the reason.
#     2. Microsoft.Support — support tickets. Absent VM families and capacity
#                           escalations land here; the "unfortunately, due to
#                           high demand..." emails are ticket *communications*.
#   Both are queryable from the CLI, so you never have to open the portal or
#   dig through email to know where you stand.
#
# RUN ORDER (first time)
#   ./quota_status.sh setup                  # once: install the two extensions
#   ./quota_status.sh skus Standard_ND96isr  # CAN the sub deploy this at all?
#   ./quota_status.sh limits                 # what you already have (AML)
#   ./quota_status.sh tickets                # what you asked for + replies
#   ./quota_status.sh families italynorth    # exact family id for that region
#   ./quota_status.sh requests               # self-service requests + reasons
#
# RUN ORDER (daily, while waiting)
#   ./quota_status.sh board                  # one-page verdict + shareable PNG
#   ./quota_status.sh all                    # limits + tickets + board
#
# ON THIS SUBSCRIPTION, `limits` + `tickets` are the pair that answer the
# question ("what landed?" / "what did I ask for?"). `families`, `requests` and
# `request` go through Microsoft.Quota, which is NotRegistered here — they will
# say so rather than print a misleading blank. See the block below.
#
# TO FILE A REQUEST
#   ./quota_status.sh request italynorth standardNDSH100v5Family 768 dedicated
#
# REQUIREMENTS: az CLI, logged in (`az login`), correct subscription selected.
# =============================================================================

set -uo pipefail   # NOT -e: a failing region must not abort the whole sweep

SUB="${AZ_SUBSCRIPTION:-ef1ff20e-1168-4846-a78e-47d102dd35f6}"

# Every region where you have filed, or might file. Edit freely.
REGIONS=(italynorth norwayeast polandcentral uksouth canadacentral
         switzerlandnorth spaincentral japaneast australiaeast)

# Scope string used by every Microsoft.Quota call.
scope() { echo "/subscriptions/$SUB/providers/Microsoft.Compute/locations/$1"; }

hr() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# WHICH QUOTA SYSTEM IS AUTHORITATIVE HERE  (learned the hard way, 2026-08-25)
#
#   This subscription runs its GPUs through Azure ML. Every quota ticket on it
#   is filed with quotaChangeRequestSubType "BatchAml", so granted cores land in
#   the *Azure ML* counters (Microsoft.MachineLearningServices), never in
#   Compute's. On top of that, BOTH Microsoft.Compute and Microsoft.Quota are
#   NotRegistered here, which is why the obvious commands lie by omission:
#     az vm list-usage -l <loc>   -> []      (Compute unregistered)
#     az quota list --scope ...   -> error   (Quota RP unregistered, swallowed)
#   Neither prints an error you'd notice in a sweep; both just look like "no
#   quota anywhere", which is indistinguishable from "nothing was ever filed".
#
#   The AML usages endpoint below needs neither RP registered and IS the budget
#   a cluster allocates against. It is the number to trust.
#
#   To re-enable the self-service path (optional, changes the subscription):
#     az provider register -n Microsoft.Quota   --wait
#     az provider register -n Microsoft.Compute --wait
# -----------------------------------------------------------------------------
ML_API=2024-10-01

# Raw Azure ML usage counters for one region.
aml_usages() {
  az rest --method get --url \
    "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.MachineLearningServices/locations/$1/usages?api-version=$ML_API" \
    -o json 2>/dev/null
}

# Raw support tickets on the subscription (REST: the `az support` command group
# was renamed to `az support in-subscription tickets`, so the old path errors).
support_tickets() {
  az rest --method get --url \
    "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Support/supportTickets?api-version=2020-04-01" \
    "$@" 2>/dev/null
}

# Guard for the three commands that go through Microsoft.Quota. Without this
# they return empty and read as "no requests exist" — the opposite of the truth.
require_quota_rp() {
  local reg
  reg=$(az provider show -n Microsoft.Quota --query registrationState -o tsv 2>/dev/null)
  [ "$reg" = "Registered" ] && return 0
  hr "Microsoft.Quota is ${reg:-unavailable} on this subscription"
  echo "  Every 'az quota' call fails with MissingRegistrationForResourceProvider,"
  echo "  so families/requests/request can only return nothing here."
  echo "  Enable it:  az provider register -n Microsoft.Quota --wait"
  echo "  Meanwhile the AML tickets are the live record:  $0 tickets"
  return 1
}

# -----------------------------------------------------------------------------
# setup — install the two CLI extensions (idempotent, ~30 s)
# -----------------------------------------------------------------------------
cmd_setup() {
  az account set --subscription "$SUB"
  az extension add --name quota   --only-show-errors 2>/dev/null || \
    az extension update --name quota   --only-show-errors
  az extension add --name support --only-show-errors 2>/dev/null || \
    az extension update --name support --only-show-errors
  az account show -o table
  echo "OK: 'az quota' and 'az support' available."
}

# -----------------------------------------------------------------------------
# skus — CAN this subscription deploy a VM size, and where?
#
#   Run this BEFORE filing anything. A region listed as
#   'NotAvailableForSubscription' will never approve, no matter how you ask —
#   that is an allow-list decision, not a capacity shortage.
#
#   Read the output:
#     Restrictions empty / "None"      -> deployable, quota is the only blocker
#     NotAvailableForSubscription      -> don't bother filing here
#     'type: Location'                 -> whole region blocked for you
#     'Location+Zones'                 -> blocked only in some zones
# -----------------------------------------------------------------------------
cmd_skus() {
  local size="${1:-Standard_ND96isr}"
  hr "VM sizes matching '$size' across all regions"
  az vm list-skus --size "$size" --all \
     --query "[].{loc:locations[0], name:name, zones:join(',',locationInfo[0].zones||['-']),
                  restriction:join(';', restrictions[].reasonCode || ['None'])}" \
     -o table
}

# -----------------------------------------------------------------------------
# families — the EXACT family id to use when filing, per region
#
#   The portal shows different names per region for the same hardware
#   (NDSH100v5 in UK South vs NDv5H100 in Italy North). Filing with the wrong
#   string silently targets nothing. This prints what the API actually accepts.
#
#   Read the output: the 'name' column is the string to pass to `request`.
# -----------------------------------------------------------------------------
cmd_families() {
  require_quota_rp || return
  local locs=("${@:-${REGIONS[@]}}")
  for LOC in "${locs[@]}"; do
    hr "GPU quota families available in $LOC"
    az quota list --scope "$(scope "$LOC")" -o json 2>/dev/null \
      | grep -Ei '"(name|value)":.*(H100|H200|A100|NC|ND)' | sort -u \
      || echo "  (none returned — region may not expose GPU families to this sub)"
  done
}

# -----------------------------------------------------------------------------
# limits — granted Azure ML GPU quota per region  (THE number that matters)
#
#   Reads Microsoft.MachineLearningServices usages, not Compute's — see the
#   block at the top of this file for why Compute's counters are always empty.
#
#   Optional arg: a regex over family names (default H100|H200|A100).
#     ./quota_status.sh limits            # the GPU families you care about
#     ./quota_status.sh limits ND         # every ND family
#
#   Read the output:
#     dedicated 0     -> nothing granted; the ticket has NOT landed, whatever
#                        any email implied. This is the blocker.
#     dedicated >0    -> granted; 'used' vs that number is your headroom.
#     lowPriority     -> the spot budget, tracked separately. 'unset' (API -1)
#                        means no explicit cap, NOT that spot capacity exists.
# -----------------------------------------------------------------------------
cmd_limits() {
  local pat="${1:-H100|H200|A100}"
  for LOC in "${REGIONS[@]}"; do
    hr "AML GPU quota in $LOC (families matching '$pat')"
    aml_usages "$LOC" | python3 -c '
import json, re, sys
raw = sys.stdin.read().strip()
if not raw:
    print("  (no response - check az login, or the region name)"); sys.exit()
pat = re.compile(sys.argv[1], re.I)
fams, envelope = {}, None
for u in json.loads(raw).get("value", []):
    name = u["name"]["value"]
    # Regional envelope: raised from the 300 default before (and independently
    # of) any per-family grant, so it moves first when a ticket starts landing.
    if name == "TotalDedicatedCores":
        envelope = (u["currentValue"], u["limit"])
    if not pat.search(name):
        continue
    kind = "ded" if "dedicatedCores" in u["type"] else "low"
    fams.setdefault(name, {})[kind] = (u["currentValue"], u["limit"])
if not fams:
    print("  (no matching families in this region)"); sys.exit()
row = "  {:<40}{:>6}{:>11}{:>13}"
print(row.format("family", "used", "dedicated", "lowPriority"))
for name in sorted(fams):
    used, lim = fams[name].get("ded", (0, 0))
    low = fams[name].get("low", (0, -1))[1]
    print(row.format(name, used, lim, "unset" if low < 0 else low))
if envelope:
    print(row.format("-- regional dedicated envelope", envelope[0], envelope[1], ""))
' "$pat"
  done
}

# -----------------------------------------------------------------------------
# requests — self-service quota requests and WHY they failed
#
#   Read the output:
#     Succeeded  -> approved; the new limit is live, go create the cluster
#     InProgress /
#     Accepted   -> still queued (capacity-pending requests sit here for days)
#     Failed     -> rejected; the 'msg' column is the rejection reason, i.e.
#                   the same text the email sent you ("due to high demand...")
#   An empty result for a region means no self-service request exists there —
#   it was filed as a support ticket instead (see `tickets`).
# -----------------------------------------------------------------------------
cmd_requests() {
  require_quota_rp || return
  for LOC in "${REGIONS[@]}"; do
    hr "Quota requests in $LOC"
    out=$(az quota request status list --scope "$(scope "$LOC")" \
            --query "[].{id:name, state:provisioningState, msg:message}" \
            -o table 2>/dev/null)
    if [ -n "$out" ]; then
      echo "$out"
    else
      # Flattening differs across extension versions — fall back to raw REST,
      # which is stable and shows the per-item messages verbatim.
      az rest --method get --url \
        "https://management.azure.com$(scope "$LOC")/providers/Microsoft.Quota/quotaRequests?api-version=2023-02-01" \
        --query "value[].{id:name, state:properties.provisioningState,
                          msg:properties.message}" -o table 2>/dev/null \
        || echo "  (no requests found)"
    fi
  done
}

# -----------------------------------------------------------------------------
# tickets — support tickets and the reply emails, in the terminal
#
#   Read the output:
#     status Open        -> still with the capacity team
#     status Closed      -> decided; read the communications for the verdict
#   `communications` prints the actual email bodies, newest last. This is where
#   "your request will stay pending" and "no capacity in <region>" live.
# -----------------------------------------------------------------------------
cmd_tickets() {
  hr "Quota tickets on this subscription (oldest first)"
  support_tickets -o json | python3 -c '
import json, re, sys
raw = sys.stdin.read().strip()
if not raw:
    print("  (no tickets returned)"); sys.exit()
rows = []
for t in json.loads(raw).get("value", []):
    p = t["properties"]
    # The title is boilerplate ("Requesting quota increase in <region> for
    # <sub>"); the region/family/cores actually asked for live in description.
    m = re.search(r"Requesting (\d+) cores for VM Family - (\S+) in (\S+) region",
                  p.get("description") or "")
    rows.append((p.get("createdDate", "")[:10], p.get("modifiedDate", "")[:10],
                 p.get("status", ""), m.group(3) if m else "-",
                 m.group(2) if m else "-", m.group(1) if m else "-", t["name"]))
rows.sort()
row = "  {:<12}{:<12}{:<8}{:<18}{:<28}{:>6}  {}"
print(row.format("created", "updated", "status", "region", "family", "cores", "ticket"))
for r in rows:
    print(row.format(*r))
print("\n  {} ticket(s)".format(len(rows)))
'

  hr "Latest reply per ticket (HTML stripped)"
  for T in $(support_tickets --query "value[].name" -o tsv); do
    echo "--- $T"
    az rest --method get --url \
      "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Support/supportTickets/$T/communications?api-version=2020-04-01" \
      -o json 2>/dev/null | python3 -c '
import html, json, re, sys
raw = sys.stdin.read().strip()
msgs = json.loads(raw).get("value", []) if raw else []
if not msgs:
    print("    (no replies yet)"); sys.exit()
p = msgs[-1]["properties"]
body = html.unescape(re.sub(r"<[^>]+>", " ", p.get("body") or ""))
body = re.sub(r"\s+", " ", body).strip()
print("    {}  {}".format(p.get("createdDate", "")[:16], p.get("sender") or ""))
print("    " + body[:700] + ("..." if len(body) > 700 else ""))
'
  done
}

# -----------------------------------------------------------------------------
# request — file (or raise) a quota limit from the CLI
#
#   Usage: ./quota_status.sh request <region> <family> <cores> [dedicated|lowPriority]
#   e.g.   ./quota_status.sh request italynorth standardNDSH100v5Family 768 dedicated
#
#   <family> MUST come from `families <region>` — names differ per region.
#   <cores>  is vCPUs, not GPUs: ND96isr = 96 cores/node (8x H100),
#            NC80adis = 80 cores/node (2x H100). 768 = 8 ND nodes.
#
#   Tries `update` first (family already tracked) and falls back to `create`.
#   Outcome is immediate: Succeeded if capacity exists, Failed with a reason
#   if not — same answer as the email, minutes instead of days. Re-check later
#   with `requests`; a Failed capacity request does NOT auto-retry, so refile.
# -----------------------------------------------------------------------------
cmd_request() {
  require_quota_rp || return
  local loc="$1" family="$2" cores="$3" type="${4:-dedicated}"
  hr "Requesting $cores cores of $family in $loc ($type)"
  az quota update --resource-name "$family" --scope "$(scope "$loc")" \
     --limit-object value="$cores" --resource-type "$type" -o json \
  || az quota create --resource-name "$family" --scope "$(scope "$loc")" \
     --limit-object value="$cores" --resource-type "$type" -o json
  echo "Filed. Check with: $0 requests"
}

# -----------------------------------------------------------------------------
# aml — Azure ML's OWN quota counters (a separate budget from Compute's)
#
#   AML tracks dedicated and lowPriority cores per region independently of the
#   Compute family quota. A cluster can fail even with Compute quota granted if
#   this counter is 0 — worth a look when a compute create fails inexplicably.
# -----------------------------------------------------------------------------
cmd_aml() {
  for LOC in "${REGIONS[@]}"; do
    hr "Azure ML regional totals in $LOC"
    aml_usages "$LOC" | python3 -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("  (no response)"); sys.exit()
for u in json.loads(raw).get("value", []):
    if u["name"]["value"].startswith("Total"):
        print("  {:<44}{:>6} / {}".format(u["name"]["localizedValue"],
                                          u["currentValue"], u["limit"]))
'
  done
}

# -----------------------------------------------------------------------------
# spot — Spot placement score: WHERE capacity actually is, before you file
#
#   Predictive rather than retrospective: scores how likely N nodes of a SKU are
#   to be obtainable per region. If the api-version below is rejected, the error
#   lists the supported ones — copy the newest and retry.
#   Read the output: score High/Medium/Low per region; file where it's High.
# -----------------------------------------------------------------------------
cmd_spot() {
  local size="${1:-Standard_ND96isr_H100_v5}" count="${2:-8}"
  hr "Spot placement score for $count x $size"
  az rest --method post --url \
    "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Compute/locations/westeurope/placementScores/spot/generate?api-version=2024-03-01" \
    --body "{\"desiredSizes\":[{\"sku\":\"$size\"}],\"desiredCount\":$count,
             \"desiredLocations\":[\"italynorth\",\"norwayeast\",\"polandcentral\",
                                   \"uksouth\",\"swedencentral\",\"westeurope\"],
             \"availabilityZones\":false}" -o json \
    || echo "  (adjust api-version per the error above, then re-run)"
}

# -----------------------------------------------------------------------------
# board — the one-page answer: terminal report + a shareable PNG
#
#   Cross-references the three systems that each hold one third of the truth:
#     granted cores (Azure ML)  x  what we filed (tickets)  x  where we CAN
#     file (SKU allow-list, and whether Azure ML exists in that region).
#   That last join is the point: a region whose SKU is restricted
#   'NotAvailableForSubscription' can be filed forever and never approve, and
#   a region without Azure ML cannot run these jobs even if quota were granted.
#
#   Usage: ./quota_status.sh board [--out FILE.png] [--offline]
#     --out      where to write the PNG   (default ./quota_status.png)
#     --offline  re-render from the last fetch instead of re-querying Azure
#                (a full refresh is ~45 az calls, so it takes a couple of min)
# -----------------------------------------------------------------------------
cmd_board() { python3.11 "$(dirname "$0")/quota_board.py" "$@"; }

# -----------------------------------------------------------------------------
cmd_all() { cmd_limits; cmd_tickets; cmd_board; }

case "${1:-all}" in
  setup)    cmd_setup ;;
  skus)     shift; cmd_skus "$@" ;;
  families) shift; cmd_families "$@" ;;
  limits)   cmd_limits ;;
  requests) cmd_requests ;;
  tickets)  cmd_tickets ;;
  request)  shift; cmd_request "$@" ;;
  aml)      cmd_aml ;;
  board)    shift; cmd_board "$@" ;;
  spot)     shift; cmd_spot "$@" ;;
  all)      cmd_all ;;
  *) sed -n '1,40p' "$0"; exit 1 ;;
esac
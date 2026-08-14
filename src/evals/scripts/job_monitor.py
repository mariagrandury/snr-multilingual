#!/usr/bin/env python3.11
"""Event-driven Slurm job monitor + hang watchdog.

Exits the moment any job changes state (start/finish/fail/new/gone) OR a RUNNING
job is detected HUNG — the vLLM shm_broadcast deadlock: ≥5 repeated
'No available shared memory broadcast block found … processes are hanging'
warnings in its .out while writing zero samples. Prints the transitions + any
hung job IDs; the caller relays, (on confirm) scancels + re-launches the hung
ones, and re-arms. Snapshot persists in --snap so re-launches never miss a change.

  python3.11 scripts/job_monitor.py [--snap PATH] [--interval SEC]
"""
import subprocess, json, sys, time, os, glob
from collections import Counter

snap = "/tmp/jobmon_snap.json"
interval = 60
a = sys.argv[1:]
for i, x in enumerate(a):
    if x == "--snap" and i + 1 < len(a): snap = a[i + 1]
    if x == "--interval" and i + 1 < len(a): interval = int(a[i + 1])

HANG_MARK = "No available shared memory broadcast block found"
HANG_MIN = 5   # >=5 of the 60s warnings (~5 min wedged) => hung node

def poll():
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%T|%j|%M|%L|%P"],
                         capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split("|")
        if len(p) >= 3:
            d[p[0]] = {"state": p[1], "name": p[2], "elapsed": p[3] if len(p) > 3 else "",
                       "left": p[4] if len(p) > 4 else "", "part": p[5] if len(p) > 5 else ""}
    return d

def final_state(j):
    o = subprocess.run(["sacct", "-j", j, "-X", "-n", "-o", "State,Elapsed"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return o[0].strip() if o else "?"

def is_hung(jobid, name):
    fs = glob.glob(f"logs/{name}_{jobid}.out")
    if not fs:
        return False
    try:
        tail = subprocess.run(["tail", "-50", fs[0]], capture_output=True, text=True).stdout
    except OSError:
        return False
    return tail.count(HANG_MARK) >= HANG_MIN

st = json.load(open(snap)) if os.path.exists(snap) else {"jobs": {}, "hung": []}
if not st.get("jobs"):
    st = {"jobs": poll(), "hung": st.get("hung", [])}
    json.dump(st, open(snap, "w"))
prev, prev_hung = st["jobs"], set(st.get("hung", []))

while True:
    cur = poll()
    ch, new_hung = [], []
    for j in prev:
        if j not in cur:
            ch.append(f"  ■ {prev[j]['name']} ({j}) left queue → {final_state(j)}")
    for j in cur:
        if j not in prev:
            ch.append(f"  ＋ {cur[j]['name']} ({j}) NEW [{cur[j]['state']} {cur[j]['part']}]")
        elif cur[j]["state"] != prev[j]["state"]:
            ch.append(f"  → {cur[j]['name']} ({j}) {prev[j]['state']}→{cur[j]['state']}")
    for j, v in cur.items():
        if v["state"] == "RUNNING" and j not in prev_hung and is_hung(j, v["name"]):
            new_hung.append((j, v["name"]))
    if ch or new_hung:
        if ch:
            print(f"=== job status change @ {time.strftime('%H:%M:%S')} ===")
            print("\n".join(ch))
        if new_hung:
            print(f"=== ⚠ HUNG — vLLM shm deadlock, 0 progress @ {time.strftime('%H:%M:%S')} ===")
            for j, n in new_hung:
                print(f"  ✗ {n} ({j}) — needs scancel + re-launch (fresh node)")
        c = Counter(v["state"] for v in cur.values())
        print("  queue now: " + (", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "empty")
              + f"  (total {len(cur)})")
        json.dump({"jobs": cur, "hung": sorted(prev_hung | {j for j, _ in new_hung})}, open(snap, "w"))
        break
    time.sleep(interval)

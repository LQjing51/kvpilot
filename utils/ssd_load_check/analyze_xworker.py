"""Local vs. cross-worker SSD load, from a run_xworker.sh run.

    python3 analyze_xworker.py <log dir>     (reads prefill-*.log, trace.jsonl)

Per (prefix, suffix), median of reps (ms):
  compute    e2e of [1] on A (prefix in A's HBM)
  own ssd    e2e of [4] on A - compute: A reads its own bucket files
  peer ssd   e2e of [4] on B - compute: B reads the files A wrote
  peer dram  e2e of [2] on B - compute: B reads A's DRAM (shm)
  load       `KVPilot load ... load=` of the two SSD tests, larger rank
Each test is checked: [1] hit = prefix from HBM; [4] every value from SSD on
both ranks, from the worker's own pid<pid> directory on A and from another
worker's on B; [2] DRAM only. Mismatches are dropped and counted.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PID = re.compile(r"\((?:Worker_TP\d+_EP\d+|EngineCore) pid=(\d+)\)")
PREFILL = re.compile(r"KVPilot prefill req=(\d{8})-\S+ .*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
RETRIEVE = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)"
                      r".*?ssd_dirs=(\S+)")
LOAD = re.compile(r"KVPilot load req=(\d{8})-\S+ .*?load=([\d.]+)ms ssd=(\d+)")

d = Path(sys.argv[1])
rows = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines()]
prefill, retrieve, load = {}, defaultdict(list), defaultdict(list)
for log in sorted(d.glob("prefill-*.log")):
    for line in log.read_text(errors="replace").splitlines():
        if m := PREFILL.search(line):
            prefill[int(m[1])] = dict(hit=int(m[2]), dram=int(m[3]), e2e=float(m[4]))
        elif m := RETRIEVE.search(line):
            pid = PID.search(line)
            retrieve[int(m[1])].append(dict(chunks=int(m[2]), ssd=int(m[3]),
                                            dirs=m[4].split(","),
                                            own=f"pid{pid[1]}" if pid else None))
        elif m := LOAD.search(line):
            load[int(m[1])].append(float(m[2]))

groups = defaultdict(dict)
for i, r in enumerate(rows):
    groups[(r["prefix"], r["suffix"], r["rep"])][(r["tier"], r["worker"])] = i
cells = defaultdict(lambda: defaultdict(list))
bad = defaultdict(int)


def ok(i, tier, worker, p):
    pf, rt = prefill.get(i), retrieve.get(i, [])
    if pf is None:
        return "missing"
    if tier == 1:
        return "" if pf["hit"] >= p and pf["dram"] == 0 else "not from HBM"
    if pf["dram"] < p or len(rt) != 2:
        return "prefix not loaded"
    if tier == 2:
        return "" if all(x["ssd"] == 0 for x in rt) else "not from DRAM"
    if not all(x["ssd"] == 48 * x["chunks"] for x in rt):
        return "not all from SSD"
    own = [x["own"] in x["dirs"] for x in rt]
    if worker == "A" and not all(own and len(x["dirs"]) == 1 for x, own in zip(rt, own)):
        return "A did not read its own files"
    if worker == "B" and any(own):
        return "B read its own files"
    return ""


for (p, s, _), t in sorted(groups.items()):
    got = {}
    for key in ((1, "A"), (4, "A"), (4, "B"), (2, "B")):
        why = ok(t[key], *key, p)
        if why:
            bad[f"{key}: {why}"] += 1
        else:
            got[key] = prefill[t[key]]["e2e"]
    if (1, "A") not in got:
        continue
    c = got[(1, "A")]
    cells[(p, s)]["compute"].append(c)
    for key, name in (((4, "A"), "own"), ((4, "B"), "peer"), ((2, "B"), "dram")):
        if key in got:
            cells[(p, s)][name].append(got[key] - c)
            if key[0] == 4 and load.get(t[key]):
                cells[(p, s)][name + "_load"].append(max(load[t[key]]))


def med(v):
    return statistics.median(v) if v else float("nan")


print(f"{'prefix':>7} {'suffix':>6} {'compute':>8} {'own ssd':>8} {'%':>6} {'peer ssd':>9} "
      f"{'%':>6} {'peer dram':>9} {'own load':>9} {'peer load':>9} {'n':>2}")
for (p, s), c in sorted(cells.items()):
    comp = med(c["compute"])
    o, q, r = med(c["own"]), med(c["peer"]), med(c["dram"])
    print(f"{p:7d} {s:6d} {comp:8.1f} {o:8.1f} {100 * o / comp:5.1f}% {q:9.1f} "
          f"{100 * q / comp:5.1f}% {r:9.1f} {med(c['own_load']):9.1f} {med(c['peer_load']):9.1f} "
          f"{len(c['compute']):2d}")
print("dropped:", dict(bad) or "none")

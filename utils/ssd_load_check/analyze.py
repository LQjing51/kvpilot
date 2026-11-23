"""Compute vs. DRAM / SSD load delay from ssd_load_check runs.

    python3 analyze.py <log dir> <trace.jsonl> [<log dir> <trace.jsonl> ...]

Per trace line, from prefill-0.log:
  e2e        `KVPilot prefill ... e2e` (scheduled -> prefill done, no queueing)
  source     `KVPilot prefill` hit/dram, and `KVPilot prelookup retrieve`
             ssd_values (one line per TP rank)
  load       `KVPilot load ... load=` of the [2] / [4] test, larger rank: the
             loader span from load start (SSD: before the first read) to the
             last layer's scatter
  first      `first=`: layer 0 on the GPU, same clock as load
  start      `start=`: CPU time of the call that set the load up
  gpu_stall  `gpu_stall=`: compute-stream idle on the load, max over TP ranks
             per wait, then summed
compute = e2e of the [1] test; delay = e2e of the [2] / [4] test - compute.
gpu_stall is checked against delay request by request, each test paired with
the [1] of its own rep. Every test is checked to have loaded its prefix from the
tier it asked for; mismatches are dropped and counted.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PREFILL = re.compile(r"KVPilot prefill req=(\d{8})-\S+ line=\d+ prompt=(\d+) new=(\d+) "
                     r"hit=(\d+) dram=(\d+) e2e=([\d.]+)ms"
                     r"(?:.*? gpu_stall=([\d.]+)ms)?")
RETRIEVE = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")
LOAD = re.compile(r"KVPilot load req=(\d{8})-\S+ tokens=\d+ layers=(\d+) fetch=[\d.]+ms "
                  r"h2d=([\d.]+)ms load=([\d.]+)ms ssd=(\d+)"
                  r"(?: first=([\d.]+)ms start=([\d.]+)ms)?")


def med(v):
    return statistics.median(v) if v else float("nan")


def collect(log_dir, trace, cells, bad):
    log = Path(log_dir)
    log = log / "prefill-0.log" if log.is_dir() else log
    rows = [json.loads(x) for x in Path(trace).read_text().splitlines()]
    prefill, retrieve, load = {}, defaultdict(list), defaultdict(list)
    for line in log.read_text(errors="replace").splitlines():
        if m := PREFILL.search(line):
            prefill[int(m[1])] = dict(hit=int(m[4]), dram=int(m[5]), e2e=float(m[6]),
                                      gs=float(m[7]) if m[7] else None)
        elif m := RETRIEVE.search(line):
            retrieve[int(m[1])].append((int(m[2]), int(m[3])))
        elif m := LOAD.search(line):
            load[int(m[1])].append(dict(layers=int(m[2]), h2d=float(m[3]), load=float(m[4]),
                                        first=float(m[6]) if m[6] else None,
                                        start=float(m[7]) if m[7] else None))
    groups = defaultdict(dict)
    for i, r in enumerate(rows):
        groups[(r["prefix"], r["suffix"], r["rep"])][r["tier"]] = i
    for (p, s, _), lines in sorted(groups.items()):
        got = {}
        for tier in (1, 2, 4):
            i = lines.get(tier)
            if i is None or i not in prefill:
                bad[f"missing tier {tier}"] += 1
                continue
            pf, rt = prefill[i], retrieve.get(i, [])
            layers = load[i][0]["layers"] if load.get(i) else 48
            if tier == 1:
                ok = pf["hit"] >= p and pf["dram"] == 0
            elif tier == 2:
                ok = pf["dram"] >= p and rt and all(v == 0 for _, v in rt)
            else:
                ok = pf["dram"] >= p and rt and all(v == c * layers for c, v in rt)
            if not ok:
                bad[f"tier {tier} wrong source"] += 1
                continue
            got[tier] = pf["e2e"]
            if tier in (2, 4) and load.get(i):
                name = "dram" if tier == 2 else "ssd"
                ld = load[i]
                cells[(p, s)][f"{name}_load"].append(max(x["load"] for x in ld))
                if ld[0]["first"] is not None:
                    cells[(p, s)][f"{name}_first"].append(max(x["first"] for x in ld))
                    cells[(p, s)][f"{name}_start"].append(max(x["start"] for x in ld))
        if 1 in got:
            cells[(p, s)]["compute"].append(got[1])
            for tier, name in ((2, "dram"), (4, "ssd")):
                if tier in got:
                    d = got[tier] - got[1]
                    cells[(p, s)][f"{name}_delay"].append(d)
                    gs = prefill[lines[tier]]["gs"]
                    if gs is not None:
                        cells[(p, s)][f"{name}_gs"].append(gs)
                        cells[(p, s)][f"{name}_err"].append(gs - d)


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        args.append(str(Path(__file__).with_name("trace.jsonl")))
    cells = defaultdict(lambda: defaultdict(list))
    bad = defaultdict(int)
    for log_dir, trace in zip(args[::2], args[1::2]):
        collect(log_dir, trace, cells, bad)

    print("delay = e2e - compute (ms, median of reps); * marks ssd-safe (< 5%)")
    print(f"{'prefix':>7} {'suffix':>6} {'compute':>8} {'dram dly':>9} {'%':>6} "
          f"{'ssd dly':>9} {'%':>7} {'ssd load':>9} {'GiB/s':>6} {'n':>2}")
    for (p, s), c in sorted(cells.items()):
        comp, dd, sd, sl = (med(c[k]) for k in ("compute", "dram_delay", "ssd_delay", "ssd_load"))
        # Both ranks read their half of the KV (96 KiB/token in total) from the same disks.
        gib_s = p * 96 * 1024 / 2**30 / (sl / 1e3) if sl == sl else float("nan")
        mark = "*" if sd / comp < 0.05 else " "
        print(f"{p:7d} {s:6d} {comp:8.1f} {dd:9.1f} {100 * dd / comp:5.1f}% "
              f"{sd:9.1f} {100 * sd / comp:5.1f}%{mark} {sl:9.1f} {gib_s:6.2f} {len(c['compute']):2d}")
    if any(c["ssd_first"] for c in cells.values()):
        print("\nlayer 0: start = CPU time setting the load up; first = layer 0 on the GPU after that")
        print(f"{'prefix':>7} {'suffix':>6} {'dram start':>10} {'dram first':>10} "
              f"{'ssd start':>9} {'ssd first':>9} {'ssd dly':>8}")
        for (p, s), c in sorted(cells.items()):
            print(f"{p:7d} {s:6d} {med(c['dram_start']):10.2f} {med(c['dram_first']):10.2f} "
                  f"{med(c['ssd_start']):9.2f} {med(c['ssd_first']):9.2f} {med(c['ssd_delay']):8.1f}")
    if any(c["dram_gs"] or c["ssd_gs"] for c in cells.values()):
        print("\ngpu_stall vs delay (ms, median of reps)")
        print(f"{'prefix':>7} {'suffix':>6} | {'dram dly':>8} {'gpu_stall':>9} | "
              f"{'ssd dly':>8} {'gpu_stall':>9}")
        err, spread = defaultdict(list), defaultdict(list)
        for (p, s), c in sorted(cells.items()):
            band = "short" if s <= 1024 else "long"
            for name in ("dram", "ssd"):
                err[(name, band)] += c[f"{name}_err"]
                if c[f"{name}_delay"]:
                    spread[(name, band)].append(max(c[f"{name}_delay"]) - min(c[f"{name}_delay"]))
            print(f"{p:7d} {s:6d} | {med(c['dram_delay']):8.1f} {med(c['dram_gs']):9.1f} | "
                  f"{med(c['ssd_delay']):8.1f} {med(c['ssd_gs']):9.1f}")
        q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * (len(v) - 1) + 0.5))]
        print("\ngpu_stall - delay per request (ms); short = suffix <= 1024 (cudagraph); "
              "spread = delay range over the reps of a cell")
        for (name, band), v in sorted(err.items()):
            a = [abs(x) for x in v]
            print(f"  {name:4s} {band:5s} median {med(v):+6.1f}  |err| p50 {q(a, .5):5.1f} "
                  f"p90 {q(a, .9):5.1f} max {max(a):5.1f}  spread median "
                  f"{med(spread[(name, band)]):5.1f} max {max(spread[(name, band)]):5.1f}  "
                  f"n={len(v)}")
    print("dropped:", dict(bad) or "none")


main()

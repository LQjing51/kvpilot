"""Summarize run_wb_diag.sh: per mode, the write-back request and the probes
after it (e2e, gpu_stall, their difference), against the probes of rounds
without a write-back.

    python3 analyze_wb.py <dynamo-log dir> [off put skip]
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ (?:(?!KVPilot).)*?hit=(\d+) dram=(\d+) "
                 r"e2e=([\d.]+)ms(?:(?:(?!KVPilot).)*? gpu_stall=([\d.]+)ms)?")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")
WB = re.compile(r"KVPilot writeback req=(\d{8})-\S+ values=(\d+) alloc_ms=([\d.]+)")


def load(d: Path):
    rows = [json.loads(x) for x in open(d / "trace.jsonl")]
    pre, ssd, wb = {}, defaultdict(int), defaultdict(list)
    for line in open(d / "prefill-0.log", errors="replace"):
        for m in PRE.finditer(line):
            pre[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]), float(m[5] or 0))
        for m in RET.finditer(line):
            ssd[int(m[1])] += int(m[3])
        for m in WB.finditer(line):
            wb[int(m[1])].append((int(m[2]), float(m[3])))
    return rows, pre, ssd, wb


def show(label, xs):
    if not xs:
        return
    e = [x[2] for x in xs]
    g = [x[3] for x in xs]
    print(f"  {label:22s} n={len(xs):3d}  e2e med {st.median(e):6.1f}  gpu_stall med "
          f"{st.median(g):5.1f}  e2e-gpu_stall med {st.median(a - b for a, b in zip(e, g)):6.1f} ms")


def main():
    root = Path(sys.argv[1])
    for mode in sys.argv[2:] or ["off", "put", "skip"]:
        d = root / f"ssd-wbdiag-{mode}-logs"
        if not (d / "prefill-0.log").exists():
            continue
        rows, pre, ssd, wb = load(d)
        by = defaultdict(list)
        has_wb = {r["round"] for r in rows if r["kind"] == "wb"}
        for i, r in enumerate(rows):
            if i not in pre:
                continue
            if r["kind"] == "wb":
                by["wb"].append(pre[i])
            elif r["kind"] == "probe":
                tag = "after wb" if r["round"] in has_wb else "no wb"
                by[f"probe {r['k']} {tag}"].append(pre[i])
        wbs = [v for vs in wb.values() for v in vs]
        wb_ssd = [ssd.get(i, 0) for i, r in enumerate(rows) if r["kind"] == "wb"]
        print(f"== {mode}: wb requests read {st.median(wb_ssd) if wb_ssd else 0:.0f} SSD values "
              f"(both ranks, median); write-backs logged {len(wbs)}"
              + (f", values {st.median(v for v, _ in wbs):.0f}, alloc_ms med "
                 f"{st.median(p for _, p in wbs):.1f}" if wbs else ""))
        show("wb request", by["wb"])
        if mode.startswith("half"):
            for parity, tag in ((0, "written back"), (1, "not written back")):
                show(f"wb {tag}", [pre[i] for i, r in enumerate(rows)
                                   if r["kind"] == "wb" and r["round"] % 2 == parity and i in pre])
                show(f"probes after {tag}", [pre[i] for i, r in enumerate(rows)
                                             if r["kind"] == "probe" and r["round"] in has_wb
                                             and r["round"] % 2 == parity and i in pre])
        for k in range(1, 5):
            show(f"probe {k} after wb", by[f"probe {k} after wb"])
        show("probes, no-wb rounds", [x for k in range(1, 5) for x in by[f"probe {k} no wb"]])


if __name__ == "__main__":
    main()

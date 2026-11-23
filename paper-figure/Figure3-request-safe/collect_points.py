"""Per-request points for the time model: e2e of the [1] / [2] / [4] tests
and, for [2] / [4], the load and first-layer times the loaders logged.

    python3 collect_points.py swe swe_reqprio swe_hbm1to2 --out points.json

One point per trace src_line whose three tests passed their source checks
(tier_logs.check). load / first: `KVPilot load ... load= first=` (the slower
TP rank); first is 0 for the DRAM loader lines that have no first= field.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tier_logs import _prefill_logs, dataset, open_log, runs

HERE = Path(__file__).resolve().parent
LOAD = re.compile(r"KVPilot load req=(\d{8})-\S+ tokens=(\d+) .*?load=([\d.]+)ms"
                  r"(?: ssd=(\d+) first=([\d.]+)ms)?")


def loads(run):
    """req id -> (load ms, first ms), the slower rank, last episode wins."""
    per = defaultdict(list)
    for f in _prefill_logs(run):
        for line in open_log(f):
            for m in LOAD.finditer(line):
                per[int(m[1])].append((float(m[3]), float(m[5] or 0.0)))
    return {rid: (max(a for a, _ in v[-2:]), max(b for _, b in v[-2:])) for rid, v in per.items()}


def collect(datasets) -> list:
    """Points of raw/<dataset> runs, in order."""
    out = []
    for name in datasets:
        d = HERE / "raw" / name
        _, res, _ = dataset(d)
        n0 = len(out)
        for run in runs(d):
            ld = loads(run)
            rows = [json.loads(x) for x in open(run / "trace.jsonl")]
            meta = defaultdict(dict)
            for i, r in enumerate(rows):
                if r["kind"] in ("t1", "t2", "t4"):
                    meta[r["src_line"]][r["kind"]] = (r.get("line", i), r)
            for src, kinds in meta.items():
                if len(kinds) != 3:
                    continue
                e = {k: res.get((src, k), (None, "m"))[0] for k in kinds}
                if None in e.values():
                    continue
                r = kinds["t1"][1]
                p, s = r["prefix"], r["input_length"] - r["prefix"]
                if p == 0 or s <= 0:
                    continue
                l2, l4 = ld.get(kinds["t2"][0]), ld.get(kinds["t4"][0])
                if l2 is None or l4 is None:
                    continue
                out.append({"ds": name, "p": p, "s": s, "t1": e["t1"], "t2": e["t2"],
                            "t4": e["t4"], "l2": l2[0], "f2": l2[1], "l4": l4[0], "f4": l4[1]})
        print(f"{name}: {len(out) - n0} points")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="points.json")
    args = ap.parse_args()
    out = collect(args.datasets)
    (HERE / args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(out)}")


if __name__ == "__main__":
    main()

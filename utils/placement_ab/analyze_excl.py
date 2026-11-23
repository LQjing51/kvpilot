"""Compare A/B arms of one req-priority trace from their logs (stdlib only).

    python3 analyze_excl.py <signal_<trace>.json> LABEL=<log dir> [LABEL=<log dir> ...]

The first arm is the baseline. Each log dir holds benchmark.log and
prefill-0.log(.gz) (a run_ab.sh dynamo-log/<name>-logs directory or a copy).
Only trace lines completed in every arm count, matched by line. Same measures
as paper-figure/Eval-three-req-priority/plot.py: prefill e2e (scheduled to
done) split into exposed load (gpu_stall) and the rest, TTFT (prefill done -
sent), where prefix tokens came from, and how many came from a tier slower /
faster than the request's own measured one.
"""
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TIERS = ("HBM", "DRAM", "SSD")
NOT_NEXT = r"(?:(?!KVPilot).)*?"
PRE = re.compile(rf"KVPilot prefill req=(\S+) line=(\d+) {NOT_NEXT}hit=(\d+) dram=(\d+) "
                 rf"e2e=([\d.]+)ms{NOT_NEXT} gpu_stall=([\d.]+)ms done=([\d.]+)"
                 r"(?: blocked=([\d.]+)ms)?")
RET = re.compile(rf"KVPilot prelookup retrieve req=(\S+) {NOT_NEXT}ssd_tokens=(\d+)")
OK = re.compile(r"^\[ok \] idx=(\d+) .*? ttft=([\d.]+)ms.*? sent=([\d.]+)")


def open_log(p):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, errors="replace")


def load(d):
    sent = {}
    for line in open_log(d / "benchmark.log"):
        m = OK.match(line)
        if m:
            sent[int(m[1])] = float(m[3])
    plog = d / "prefill-0.log.gz"
    if not plog.exists():
        plog = d / "prefill-0.log"
    out, rid_line, ssd = {}, {}, defaultdict(int)
    for line in open_log(plog):
        for m in RET.finditer(line):
            ssd[m[1]] = max(ssd[m[1]], int(m[2]))
        for m in PRE.finditer(line):
            k = int(m[2])
            rid_line[m[1]] = k
            out[k] = dict(hit=int(m[3]), dram=int(m[4]), e2e=float(m[5]), stall=float(m[6]),
                          done=float(m[7]), blocked=float(m[8] or 0))
    for rid, k in rid_line.items():
        r = out[k]
        r["ssd"] = min(ssd.get(rid, 0), r["dram"])
        r["ttft"] = (r["done"] - sent[k]) * 1e3 if k in sent else None
    out.pop(-1, None)
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def main():
    table = json.loads(Path(sys.argv[1]).read_text())
    arms = [a.split("=", 1) for a in sys.argv[2:]]
    runs = {label: load(Path(d)) for label, d in arms}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(keys)} lines in every arm ("
          + ", ".join(f"{label} {len(r)}" for label, r in runs.items()) + ")")
    rank = {t: i for i, t in enumerate(TIERS)}
    base = arms[0][0]
    for label, _ in arms:
        run = runs[label]
        e2e = [run[k]["e2e"] for k in keys]
        stall = [run[k]["stall"] for k in keys]
        ttft = [run[k]["ttft"] for k in keys if run[k]["ttft"] is not None]
        tok = {"HBM": sum(run[k]["hit"] for k in keys), "SSD": sum(run[k]["ssd"] for k in keys)}
        tok["DRAM"] = sum(run[k]["dram"] for k in keys) - tok["SSD"]
        tot = sum(tok.values()) or 1
        slow = fast = n = 0
        for k in keys:
            want = (table.get(str(k)) or {}).get("own_tier")
            if want is None:
                continue
            r = run[k]
            for got, c in (("HBM", r["hit"]), ("DRAM", r["dram"] - r["ssd"]), ("SSD", r["ssd"])):
                n += c
                slow += c * (rank[got] > rank[want])
                fast += c * (rank[got] < rank[want])
        print(f"{label:10s} prefill e2e mean {statistics.mean(e2e):6.1f} p50 {pct(e2e, .5):6.1f} "
              f"p90 {pct(e2e, .9):6.1f} p99 {pct(e2e, .99):6.1f} ms | gpu_stall mean "
              f"{statistics.mean(stall):5.1f} ms sum {sum(stall) / 1e3:6.1f} s | rest "
              f"{statistics.mean(e2e) - statistics.mean(stall):6.1f} ms")
        print(f"{'':10s} TTFT mean {statistics.mean(ttft):6.1f} p50 {pct(ttft, .5):6.1f} p90 "
              f"{pct(ttft, .9):6.1f} p99 {pct(ttft, .99):6.1f} ms | prefix {tot / 1e6:.1f}M tokens "
              "from HBM/DRAM/SSD " + "/".join(f"{100 * tok[t] / tot:.1f}" for t in TIERS)
              + f"% | vs own tier slower {100 * slow / max(n, 1):.1f}% faster "
              f"{100 * fast / max(n, 1):.1f}%")
        blocked = [run[k]["blocked"] for k in keys if run[k]["blocked"] > 0]
        print(f"{'':10s} waited for HBM: {len(blocked)} requests, total "
              f"{sum(blocked) / 1e3:.1f} s"
              + (f", per waiting request mean {statistics.mean(blocked):.1f} p50 "
                 f"{pct(blocked, .5):.1f} p90 {pct(blocked, .9):.1f} max {max(blocked):.1f} ms"
                 if blocked else ""))
        if label == base:
            continue
        a, b = runs[base], run
        d = [b[k]["e2e"] - a[k]["e2e"] for k in keys]
        ds = [b[k]["stall"] - a[k]["stall"] for k in keys]
        dr = [(b[k]["e2e"] - b[k]["stall"]) - (a[k]["e2e"] - a[k]["stall"]) for k in keys]
        both = [k for k in keys if a[k]["ttft"] is not None and b[k]["ttft"] is not None]
        tg = statistics.mean(b[k]["ttft"] - a[k]["ttft"] for k in both)
        print(f"{'':10s} vs {base}: e2e {statistics.mean(d):+.1f} ms/req "
              f"({100 * statistics.mean(d) / statistics.mean(a[k]['e2e'] for k in keys):+.1f}%), "
              f"median {statistics.median(d):+.1f}, {sum(x < 0 for x in d)}/{len(d)} faster | "
              f"gpu_stall {statistics.mean(ds):+.1f} | rest {statistics.mean(dr):+.2f} ms | TTFT "
              f"{tg:+.1f} ms ({100 * tg / statistics.mean(a[k]['ttft'] for k in both):+.1f}%)")


if __name__ == "__main__":
    main()

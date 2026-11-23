"""Compare TTFT and KV sourcing across placement A/B runs.

    python3 analyze.py <log_dir> [<log_dir> ...] [--table signal_<trace>.json] [--skip N]

Each log dir is dynamo-log/pab-<ds>-<arm>-<tag>-logs. Reads benchmark.log
(client send time and first-token TTFT per request, keyed by trace line), the
trace named in run.out (prefix per line) and prefill-0.log (per request:
HBM-hit tokens, connector-loaded tokens, prefill e2e, gpu_stall, finish time,
SSD values loaded). Requests present in every run are also compared pairwise.

TTFT is taken at the end of prefill: the prefill worker's `done=` minus the
client's `sent=` (same host, same clock). The client's first-token TTFT adds
the decode side pulling the prompt's KV, which is the same request for request
in both arms and grows with prompt length; it is printed for reference only.

Exposed load is `gpu_stall` (compute-stream idle on load events), the only
measure of it.

--skip N leaves out the first N trace lines (warm-up: the caches fill and the
arms do not differ yet) from every comparison.

--table (signal table, make_req_table.py) scores where each request's prefix came from
against its own measured request-safe tier: tokens from HBM (`hit`), SSD
(`ssd_tokens` of the retrieve) and DRAM (the rest of `dram`). A token from a
slower tier than the request's is a misread that exposes load; from a faster
one it holds faster capacity it did not need.
"""

import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OK = re.compile(r"^\[ok \] idx=(\d+) .*? ttft=([\d.]+)ms(?:.*? sent=([\d.]+))?")
PRE = re.compile(
    r"KVPilot prefill req=(\d{8})-\S+ (?:(?!KVPilot).)*?hit=(\d+) dram=(\d+) "
    r"e2e=([\d.]+)ms(?:(?:(?!KVPilot).)*? gpu_stall=([\d.]+)ms)?"
    r"(?:(?:(?!KVPilot).)*? done=([\d.]+))?"
)
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)"
                 r"(?:.*? ssd_tokens=(\d+))?")
TIERS = ("HBM", "DRAM", "SSD")


def _open(p: Path):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, errors="replace")


def load(d: Path) -> dict:
    client, sent = {}, {}
    for line in _open(d / "benchmark.log"):
        m = OK.match(line)
        if m:
            client[int(m[1])] = float(m[2])
            if m[3]:
                sent[int(m[1])] = float(m[3])
    prefix = {}
    sched = d / "schedule.jsonl"
    run_out = d / "run.out"
    trace = None
    if run_out.exists():
        for line in open(run_out):
            if line.startswith("trace="):
                trace = Path(line.strip().removeprefix("trace="))
    if sched.exists():
        for i, line in enumerate(open(sched)):
            if i == 0:
                continue
            r = json.loads(line)
            prefix[r["line_no"]] = r["prefix"]
    if trace is not None and trace.exists() and not sched.exists():
        for i, line in enumerate(open(trace)):
            prefix[i] = json.loads(line)["prefix_length"]
    pre, done, gpu, ssd, ssd_tok = {}, {}, {}, defaultdict(int), defaultdict(int)
    plog = d / "prefill-0.log"
    if not plog.exists():
        plog = d / "prefill-0.log.gz"
    if plog.exists():
        for line in _open(plog):
            for m in PRE.finditer(line):
                k = int(m[1])
                pre[k] = (int(m[2]), int(m[3]), float(m[4]))
                if m[5]:
                    gpu[k] = float(m[5])
                if m[6]:
                    done[k] = float(m[6])
            for m in RET.finditer(line):
                ssd[int(m[1])] += int(m[3])
                if m[4]:
                    # One line per TP rank, each counting the same tokens.
                    ssd_tok[int(m[1])] = max(ssd_tok[int(m[1])], int(m[4]))
    # TTFT at the end of prefill when both timestamps exist, else the client's.
    ttft = {k: (done[k] - sent[k]) * 1e3 for k in client if k in done and k in sent}
    source = "prefill end (done - sent)"
    if not ttft:
        ttft, source = dict(client), "client first token"
    return {"ttft": ttft, "client": client, "source": source, "prefix": prefix,
            "pre": pre, "ssd": ssd, "ssd_tok": ssd_tok, "gpu": gpu}


def misreads(name: str, run: dict, table: dict, keys) -> None:
    """Prefix tokens by (the request's own tier, tier they came from)."""
    rank = {t: i for i, t in enumerate(TIERS)}
    mis = defaultdict(int)
    for k in keys:
        want = (table.get(str(k)) or {}).get("own_tier")
        if want is None or k not in run["pre"]:
            continue
        hit, loaded, _ = run["pre"][k]
        ssd = min(run["ssd_tok"].get(k, 0), loaded)
        for got, n in (("HBM", hit), ("DRAM", loaded - ssd), ("SSD", ssd)):
            mis[(want, got)] += n
    total = sum(mis.values()) or 1
    low = sum(n for (w, g), n in mis.items() if rank[g] > rank[w])
    high = sum(n for (w, g), n in mis.items() if rank[g] < rank[w])
    share = "/".join(f"{100 * sum(mis[(w, g)] for w in TIERS) / total:.1f}" for g in TIERS)
    print(f"{name:34s} misplaced {100 * (low + high) / total:5.1f}% of {total / 1e6:.2f}M prefix tok "
          f"(slower {100 * low / total:5.1f}%, faster {100 * high / total:5.1f}%) | "
          f"from HBM/DRAM/SSD {share}%")
    print(f"{'':34s} tokens own->from " + "  ".join(
        f"{w}->{g}:{mis[(w, g)] / 1e6:.2f}M" for w in TIERS for g in TIERS))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else float("nan")


def summarize(name: str, run: dict, keys=None) -> None:
    t = run["ttft"]
    keys = sorted(t) if keys is None else keys
    v = [t[k] for k in keys]
    warm = [t[k] for k in keys if run["prefix"].get(k, 0) > 0]
    pre = [run["pre"][k] for k in keys if k in run["pre"]]
    hit = sum(p[0] for p in pre)
    loaded = sum(p[1] for p in pre)
    e2e = [p[2] for p in pre]
    ssd_reqs = sum(1 for k in keys if run["ssd"].get(k, 0) > 0)
    c = [run["client"][k] for k in keys if k in run["client"]]
    gpu_stall = sum(run["gpu"].get(k, 0.0) for k in keys)
    print(
        f"{name:34s} n={len(v):5d}  TTFT[{run['source']}] mean={statistics.mean(v):7.1f} "
        f"p50={pct(v, .5):7.1f} p90={pct(v, .9):7.1f} p99={pct(v, .99):7.1f} ms | "
        f"warm-prefix mean={statistics.mean(warm) if warm else float('nan'):7.1f}\n"
        f"{'':34s} prefill e2e mean={statistics.mean(e2e) if e2e else float('nan'):6.1f} ms, "
        f"sum={sum(e2e) / 1e3:7.1f} s | HBM-hit {hit / 1e6:6.2f}M tok, "
        f"loaded {loaded / 1e6:6.2f}M tok, reqs w/ SSD {ssd_reqs}\n"
        f"{'':34s} exposed load: gpu_stall sum={gpu_stall / 1e3:6.1f} s\n"
        f"{'':34s} client first-token TTFT (incl. decode-side KV pull) mean="
        f"{statistics.mean(c) if c else float('nan'):6.1f} p90={pct(c, .9):6.1f} ms"
    )


def main() -> None:
    args = sys.argv[1:]
    table = None
    if "--table" in args:
        i = args.index("--table")
        table = json.load(open(args[i + 1]))
        del args[i : i + 2]
    skip = 0
    if "--skip" in args:
        i = args.index("--skip")
        skip = int(args[i + 1])
        del args[i : i + 2]
    dirs = [Path(a) for a in args]
    runs = {d.name.removesuffix("-logs"): load(d) for d in dirs}
    print(f"== all requests of each run{f' from line {skip}' if skip else ''} ==")
    for name, run in runs.items():
        summarize(name, run, sorted(k for k in run["ttft"] if k >= skip))
    common = set.intersection(*(set(r["ttft"]) for r in runs.values()))
    common = sorted(k for k in common if k >= skip)
    print(f"\n== {len(common)} requests completed in every run ==")
    for name, run in runs.items():
        summarize(name, run, common)
    if table is not None:
        print("\n== KV source vs each request's own request-safe tier (same requests) ==")
        for name, run in runs.items():
            misreads(name, run, table, common)
    names = list(runs)
    base = runs[names[0]]["ttft"]
    for other in names[1:]:
        b = runs[other]["ttft"]
        diffs = [b[k] - base[k] for k in common]
        print(
            f"paired {other} - {names[0]}: TTFT mean diff {statistics.mean(diffs):+.1f} ms "
            f"({100 * statistics.mean(diffs) / statistics.mean(base[k] for k in common):+.1f}%), "
            f"median diff {statistics.median(diffs):+.1f} ms, "
            f"{sum(d < 0 for d in diffs)}/{len(diffs)} requests faster"
        )
        pb, pa = runs[other]["pre"], runs[names[0]]["pre"]
        gb, ga = runs[other]["gpu"], runs[names[0]]["gpu"]
        both = [k for k in common if k in pb and k in pa]
        if both:
            d = [pb[k][2] - pa[k][2] for k in both]
            print(
                f"paired {other} - {names[0]}: prefill e2e mean diff "
                f"{statistics.mean(d):+.1f} ms, gpu_stall diff "
                f"{statistics.mean(gb.get(k, 0.0) - ga.get(k, 0.0) for k in both):+.1f} ms/req"
            )


if __name__ == "__main__":
    main()

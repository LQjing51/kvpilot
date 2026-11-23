"""Per-request log parsing and source checks for raw/<dataset> runs, shared by
retry.py, plot.py and fit_times.py.

A run is a trace replayed on one service: each request is named after its
row's "line" field, or its 0-based row number when there is none (replay -p 1,
so rows run in order). A dataset is one run (raw/<dataset>/) or, when
make_trace split it, one run per raw/<dataset>/part*/. For each row, from the
prefill logs:
  hit, dram, e2e   KVPilot prefill: HBM-hit tokens, connector-loaded tokens,
                   scheduled -> prefill done (ms, no queueing)
  ssd values       KVPilot prelookup retrieve, one line per TP rank
"""
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BLOCK, LAYERS = 512, 48
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ (?:(?!KVPilot).)*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")


def open_log(f):
    return gzip.open(f, "rt", errors="replace") if f.suffix == ".gz" else open(f, errors="replace")


def _prefill_logs(d):
    """One file per prefill-N: prefer uncompressed over .gz (avoid double-count)."""
    d = Path(d)
    by_stem = {}
    for f in d.glob("prefill-*.log*"):
        stem = f.name[: -len(".gz")] if f.suffix == ".gz" else f.name
        if stem not in by_stem or f.suffix != ".gz":
            by_stem[stem] = f
    return sorted(by_stem.values())


def parse(d):
    """Last episode wins per req id (RETs then PRE). Re-runs with the same
    id (e.g. a fresh-service retry concatenated onto main) replace, not append.
    """
    pre, ret = {}, {}
    pending = defaultdict(list)
    for f in _prefill_logs(d):
        for l in open_log(f):
            for m in RET.finditer(l):
                pending[int(m[1])].append((int(m[2]), int(m[3])))
            for m in PRE.finditer(l):
                rid = int(m[1])
                pre[rid] = (int(m[2]), int(m[3]), float(m[4]))
                ret[rid] = pending.pop(rid, [])
    for rid, v in pending.items():
        ret.setdefault(rid, v)
    return pre, ret


def check(r, got, ret):
    """None if the row's request took its prefix from its tier, else why not."""
    if got is None:
        return "missing log"
    hit, conn, _ = got
    p, t = r["prefix"], r["tier"]
    if t == 0:
        return None
    if t == 1:
        return None if hit >= p and conn == 0 else "[1] source"
    if hit != 0 or conn < p or len(ret) != 2:
        return f"[{t}] source"
    want = (lambda c: 0) if t == 2 else (lambda c: LAYERS * c)
    return None if all(c == p // BLOCK and v == want(c) for c, v in ret) else f"[{t}] source"


def results(d, trace, logs=None):
    """{(src_line, kind): (e2e or None, why not)} for every test row of a run.
    `logs` is parse(d) when the caller already has it."""
    rows = [json.loads(l) for l in open(trace)]
    pre, ret = logs or parse(d)
    out = {}
    for i, r in enumerate(rows):
        rid = r.get("line", i)
        if r["tier"]:
            why = check(r, pre.get(rid), ret.get(rid, []))
            out[(r["src_line"], r["kind"])] = (None if why else pre[rid][2], why)
    return out


def runs(d):
    """Run dirs of a dataset: its part*/ when the trace was split, else d."""
    d = Path(d)
    return sorted(p for p in d.glob("part*") if p.is_dir()) or [d]


def dataset(d):
    """(main-trace rows, results, Counter of kinds the retry fixed) over every
    run of a dataset. A retry result replaces the main one only where the main
    one failed its source check."""
    rows, res, fixed = [], {}, Counter()
    for run in runs(d):
        logs = parse(run)
        got = results(run, run / "trace.jsonl", logs)
        retry = run / "trace_retry.jsonl"
        if retry.exists():
            for k, v in results(run, retry, logs).items():
                if got.get(k, (None, "missing"))[1] and not v[1]:
                    got[k] = v
                    fixed[k[1]] += 1
        rows += [json.loads(l) for l in open(run / "trace.jsonl")]
        res.update(got)
    return rows, res, fixed

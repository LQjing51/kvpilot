"""Retry trace for rows whose [2] / [4] source check failed on the main run.

    python3 retry.py <log dir> <run dir>   # reads <run dir>/trace.jsonl,
                                           # writes <run dir>/trace_retry.jsonl

Per failed line: warm on a new prefix of the same length, then the failed
tier(s). The main run's prefix is not reused: by now it may be left only on
SSD, and warm loads an SSD hit without putting it back in DRAM. [4] waits
RETRY_DELAY_MS for the offload. Same service; request ids start at ID_BASE.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from make_trace import HEAD, Fresh, line_rows
from tier_logs import results

RETRY_DELAY_MS, ID_BASE = 4000, 10**6

logdir, run = Path(sys.argv[1]), Path(sys.argv[2])
trace = run / "trace.jsonl"
res = results(logdir, trace)
rows = [json.loads(l) for l in open(trace)]
src = {r["src_line"]: r for r in rows if r["tier"]}
bad = defaultdict(list)
for (l, k), (_, why) in sorted(res.items()):
    if why and k in ("t2", "t4"):
        bad[l].append(int(k[1]))
ids = [h for r in rows for h in r["hash_ids"]]
fresh = Fresh(max(h for h in ids if h < HEAD) + 1)
heads = Fresh(max(h for h in ids if h >= HEAD) + 1)
out = []
for l, tiers in sorted(bad.items()):
    r = src[l]
    out += line_rows(l, r["input_length"], r["prefix"], tuple(tiers), fresh, heads,
                     RETRY_DELAY_MS)
with open(run / "trace_retry.jsonl", "w") as f:
    for i, r in enumerate(out):
        f.write(json.dumps({**r, "line": ID_BASE + i}) + "\n")
why = Counter((k, w) for (_, k), (_, w) in res.items() if w)
retried = Counter(t for tiers in bad.values() for t in tiers)
print(f"failed checks {dict(why) or 'none'}; retrying [2]={retried[2]} [4]={retried[4]} "
      f"({len(bad)} lines, {len(out)} requests)")

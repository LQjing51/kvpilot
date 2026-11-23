"""Pick sessions so HBM / DRAM / SSD-safe requests come out near a mix.

    python3 select_safe_mix.py TRACE                       # expected mix of TRACE as is
    python3 select_safe_mix.py TRACE --sessions-out FILE   # pick sessions (H:D = 1:2), write ids
    python3 select_safe_mix.py TRACE --sessions-out FILE --mix 1:2:2   # H:D:S

No GPU involved: a request's class is a probability looked up by (prefix,
suffix) in the SWE request-safe measurement (paper-figure/Figure3-request-safe,
4,900 requests on 1P1D H20), binned by prefix and suffix. prefix / suffix mean
what they mean there: prefix = leading full 512-token blocks already in an
earlier line, capped so at least one block is computed; suffix = the rest;
lines with prefix 0 are not counted. A trace's expected H / D / S is the sum
of those probabilities.

Picking scores each session with its prefixes counted inside the session
alone, so a session's score does not depend on which others are kept, and
keeps sessions whole. With a two-term --mix (H:D), sessions sorted by
expected H / (H + D); the longest head whose total still has H >= D * ratio
is the largest set at that ratio. A three-term --mix (H:D:S) sorts by excess
over the target shares along a direction that is swept until the head hits
all three (pick_mix). When the set holds more unique KV than
--min-unique-kv-gib, sessions are taken from it at an even stride in that
order, which keeps its mix.
Rebuild the trace from the ids with prepare_swe_trace.py --sessions, then run
this script on the result to see the mix with cross-session sharing back in.
"""
import argparse
import json
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG3 = ROOT / "paper-figure/Figure3-request-safe"
sys.path.insert(0, str(FIG3))
from tier_logs import dataset  # noqa: E402

BLOCK, KV_BYTES, THRESH = 512, 98_304, 0.05
PREFIX_BINS = (0, 4096, 16384)
SUFFIX_EDGES = (0, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768,
                1024, 1536, 2048, 3072, 4096)
MIN_CELL = 20


def measured():
    """(prefix, suffix, class) of every measured SWE line; class is H / D / S."""
    rows, res, _ = dataset(FIG3 / "raw/swe")
    out = []
    for r in rows:
        if r["kind"] != "t1":
            continue
        t1, t2, t4 = (res.get((r["src_line"], k), (None,))[0] for k in ("t1", "t2", "t4"))
        if None in (t1, t2, t4):
            continue
        c = "S" if (t4 - t1) / t1 < THRESH else "D" if (t2 - t1) / t1 < THRESH else "H"
        out.append((r["prefix"], r["input_length"] - r["prefix"], c))
    return out


def build(points):
    """Per prefix bin: (suffix cell starts, [(P(H), P(D), P(S))]). Adjacent
    SUFFIX_EDGES bins merge until a cell holds MIN_CELL measured lines.
    Prefixes beyond the measured range (64K) use the last prefix bin."""
    table = []
    for i, lo in enumerate(PREFIX_BINS):
        hi = PREFIX_BINS[i + 1] if i + 1 < len(PREFIX_BINS) else float("inf")
        fine = [Counter() for _ in SUFFIX_EDGES]
        for p, s, c in points:
            if lo <= p < hi:
                fine[bisect_right(SUFFIX_EDGES, s) - 1][c] += 1
        starts, cells, acc, start = [], [], Counter(), 0
        for j, c in enumerate(fine):
            acc += c
            if sum(acc.values()) >= MIN_CELL:
                starts.append(SUFFIX_EDGES[start])
                cells.append(acc)
                acc, start = Counter(), j + 1
        if acc and cells:
            cells[-1] += acc
        elif acc:
            starts.append(0)
            cells.append(acc)
        table.append((starts, [tuple(c[k] / sum(c.values()) for k in "HDS") for c in cells]))
    return table


def prob(table, p, s):
    starts, cells = table[bisect_right(PREFIX_BINS, p) - 1]
    return cells[bisect_right(starts, s) - 1]


def sessions(trace, table, standalone):
    """{session: (expected H, D, S, lines, counted lines, full block ids)} in
    file order. prefix counts blocks of earlier lines in the same session when
    standalone, of every earlier line otherwise."""
    out, seen, cur = {}, set(), None
    for d in map(json.loads, open(trace)):
        s = d["session_id"]
        if standalone and s != cur:
            seen = set()
        cur = s
        n, ids = d["input_length"], d["hash_ids"]
        full = ids[: n // BLOCK]
        k = 0
        while k < len(full) and full[k] in seen:
            k += 1
        seen.update(full)
        p = min(k, (n - 1) // BLOCK) * BLOCK
        e = out.setdefault(s, [0.0, 0.0, 0.0, 0, 0, set()])
        e[3] += 1
        e[5].update(full)
        if p:
            for i, x in enumerate(prob(table, p, n - p)):
                e[i] += x
            e[4] += 1
    return out


def kv_gib(stats, chosen):
    return len({h for s in chosen for h in stats[s][5]}) * BLOCK * KV_BYTES / 2**30


def summary(stats, chosen):
    h, d, s, n, m = (sum(stats[x][i] for x in chosen) for i in range(5))
    return (f"{len(chosen):,} sessions, {n:,} lines ({m:,} with prefix>0), "
            f"unique full-block KV {kv_gib(stats, chosen):,.0f} GiB; expected "
            f"H {h / m:.1%}  D {d / m:.1%}  S {s / m:.1%}  "
            f"H:D:S = 1:{d / h:.2f}:{s / h:.2f}")


def shrink(stats, largest, kv_target):
    """`largest` itself, or the fewest sessions taken from it at an even
    stride (in its order) that still hold kv_target GiB."""
    head = len(largest)
    if not kv_target or kv_gib(stats, largest) <= kv_target:
        return largest

    def stride(c):
        return [largest[int((i + 0.5) * head / c)] for i in range(c)]

    lo, hi = 1, head
    while lo < hi:
        mid = (lo + hi) // 2
        if kv_gib(stats, stride(mid)) >= kv_target:
            hi = mid
        else:
            lo = mid + 1
    return stride(lo)


def pick(stats, ratio, kv_target):
    """H : D only: sessions by expected H / (H + D), longest head with H >= ratio * D."""
    cand = [s for s, e in stats.items() if e[0] + e[1] > 0]
    cand.sort(key=lambda s: -stats[s][0] / (stats[s][0] + stats[s][1]))
    h = d = 0.0
    head = 0
    for i, s in enumerate(cand):
        h, d = h + stats[s][0], d + stats[s][1]
        if h >= ratio * d:
            head = i + 1
    largest = cand[:head]
    return largest, shrink(stats, largest, kv_target)


def pick_mix(stats, mix, kv_target, tol=0.005, step_deg=0.25):
    """H : D : S. A session's excess over the target shares is a point
    (H - n t_H, S - n t_S) (D's follows). For a direction u, sort sessions by
    their per-line excess along u and take the longest head whose total excess
    along u is still >= 0; u sweeps the circle, and the largest head whose
    total excess is within tol of zero in both coordinates wins."""
    t = np.asarray(mix, dtype=float) / sum(mix)
    cand = [s for s, e in stats.items() if e[4] > 0]
    E = np.array([stats[s][:3] for s in cand])
    n = E.sum(1)
    X = (E - n[:, None] * t)[:, [0, 2]]
    best = []
    for deg in np.arange(0, 360, step_deg):
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        order = np.argsort(-(X @ u) / n, kind="stable")
        cum = np.cumsum(X[order], 0)
        ok = np.nonzero(cum @ u >= 0)[0]
        if not len(ok):
            continue
        k = ok[-1] + 1
        if np.abs(cum[k - 1]).max() <= tol * n[order[:k]].sum() and k > len(best):
            best = [cand[i] for i in order[:k]]
    return best, shrink(stats, best, kv_target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="Mooncake 格式 trace（如 swe_full.jsonl）")
    ap.add_argument("--sessions-out", help="写出选中的 session id，一行一个，按文件顺序")
    ap.add_argument("--mix", default="1:2",
                    help="目标比例：两项是 HBM:DRAM safe（SSD 不管），三项是 HBM:DRAM:SSD safe")
    ap.add_argument("--min-unique-kv-gib", type=float, default=400,
                    help="最大可行集合超过这么多不同整块 KV 时按等间隔抽到这个量；0 = 不抽")
    a = ap.parse_args()
    mix = [float(x) for x in a.mix.split(":")]
    if len(mix) not in (2, 3):
        ap.error("--mix 要两项或三项")

    table = build(measured())
    if not a.sessions_out:
        stats = sessions(a.trace, table, standalone=False)
        print(summary(stats, list(stats)))
        return
    stats = sessions(a.trace, table, standalone=True)
    print("all       :", summary(stats, list(stats)))
    if len(mix) == 2:
        largest, chosen = pick(stats, mix[0] / mix[1], a.min_unique_kv_gib)
    else:
        largest, chosen = pick_mix(stats, mix, a.min_unique_kv_gib)
    if not largest:
        sys.exit(f"没有能达到 {a.mix} 的 session 组合")
    print("largest   :", summary(stats, largest))
    print("chosen    :", summary(stats, chosen))
    keep = set(chosen)
    with open(a.sessions_out, "w") as f:
        for s in stats:
            if s in keep:
                f.write(s + "\n")
    print(f"-> {a.sessions_out}")


if __name__ == "__main__":
    main()

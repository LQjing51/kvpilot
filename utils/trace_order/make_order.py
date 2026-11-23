"""Reorder a Mooncake-format agent trace so that same-session reuse is far apart.

    python3 make_order.py <trace.jsonl> <out.jsonl> [--align end]
    python3 make_order.py <trace.jsonl> <out.jsonl> --align start --window W
    python3 make_order.py <trace.jsonl> <out.jsonl> --align budget --budget-gib G
    python3 make_order.py <trace.jsonl> <out.jsonl> --align poisson \
        [--session-rate 0.5] [--mean-gap 60] [--seed 0]

The first two are a round robin over sessions: each step sends the next round
of one session, and between two rounds of a session every other active session
sends one round. Rounds of a session keep their order, so the reuse distance of
a prefix is the combined prompt length of the other active sessions:

  < HBM pool         the prefix is still in HBM
  HBM .. DRAM        it was evicted from HBM but not from DRAM
  > DRAM             part of it was evicted from DRAM too. A chunk read from SSD
                     is not written back, so DRAM settles on the subset that
                     fits and the rest keeps coming from SSD.

  --align end    (default) all sessions take part, and a session with R rounds
                 joins in cycle R_max - R, so every session sends its last round
                 in the last cycle. Sessions are at their longest together and
                 the active working set peaks at the whole trace.
  --align start  W sessions are active at once and start together; a session
                 that has sent its last round is replaced by the next one in
                 file order.
  --align budget like start, but by KV instead of a count: each session
                 reserves its mean prompt size, sessions join in file order
                 while the reservations fit --budget-gib, and a finished
                 session's reservation goes to the next ones. Phases mix, so
                 the active sessions' KV settles near the budget.

These three give every line timestamp 0, so mooncake_trace_replay.py sends in
file order as fast as -p allows.

  --align poisson  real timestamps (ms): sessions start in file order as a
                 Poisson process of --session-rate per second, and the gap
                 between two rounds of a session is exponential with mean
                 --mean-gap seconds. Lines are sorted by timestamp. Sessions
                 start close together but end far apart, so the working set
                 stays well below the whole trace.

The replay makes a round also wait for the previous round of its session to
finish. `src_line` is the line in the input file; the replay's
request ids carry the output line.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, OrderedDict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_sim import (  # noqa: E402
    BLOCK, DRAM_GIB, HBM_BLOCKS, dram_blocks, fmt, simulate, summarize,
)

BLOCK_GIB = 48 / 1024  # one 512-token block of Qwen3-30B-A3B bf16, both TP ranks


def round_robin(sessions: list[list[dict]], window: int) -> list[dict]:
    pool = deque(deque(s) for s in sessions)
    active: deque = deque()
    out = []
    while pool or active:
        while len(active) < window and pool:
            active.append(pool.popleft())
        s = active.popleft()
        out.append(s.popleft())
        if s:
            active.append(s)
    return out


def full_blocks(r: dict) -> list:
    return r["hash_ids"][: r["input_length"] // BLOCK]


def budget_rr(sessions: list[list[dict]], budget_blocks: float) -> list[dict]:
    """Round robin over the active sessions. Each session reserves its mean
    prompt size (full blocks, over its rounds); sessions join in file order
    while the reservations fit the budget and free theirs after their last
    round, which is when the next ones join."""
    resv = [sum(len(full_blocks(r)) for r in s) / len(s) for s in sessions]
    pool = deque(range(len(sessions)))
    active: deque = deque()
    sent = [0] * len(sessions)
    used = 0.0
    out = []
    while pool or active:
        while pool and (not active or used + resv[pool[0]] <= budget_blocks):
            used += resv[pool[0]]
            active.append(pool.popleft())
        i = active.popleft()
        out.append(sessions[i][sent[i]])
        sent[i] += 1
        if sent[i] < len(sessions[i]):
            active.append(i)
        else:
            used -= resv[i]
    return out


def live_blocks(rows: list[dict]) -> list[int]:
    """After each line: distinct full blocks in the latest prompt of every
    session that still has rounds to send."""
    left = Counter(r["session_id"] for r in rows)
    cur: dict = {}
    ref: Counter = Counter()
    out = []
    for r in rows:
        s = r["session_id"]
        for b in cur.pop(s, ()):
            ref[b] -= 1
            if not ref[b]:
                del ref[b]
        left[s] -= 1
        if left[s]:
            cur[s] = full_blocks(r)
            ref.update(cur[s])
        out.append(len(ref))
    return out


def end_aligned(sessions: list[list[dict]]) -> list[dict]:
    cycles = max(len(s) for s in sessions)
    out = []
    for c in range(cycles):
        for s in sessions:
            k = c - (cycles - len(s))
            if k >= 0:
                out.append(s[k])
    return out


def poisson(sessions: list[list[dict]], rate: float, mean_gap: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    start = 0.0
    for s in sessions:
        start += rng.expovariate(rate)
        t = start
        for k, r in enumerate(s):
            if k:
                t += rng.expovariate(1 / mean_gap)
            r["timestamp"] = round(t * 1000)
            out.append(r)
    out.sort(key=lambda r: r["timestamp"])
    t0 = out[0]["timestamp"]
    for r in out:
        r["timestamp"] -= t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--align", choices=("end", "start", "budget", "poisson"), default="end")
    ap.add_argument("--window", type=int, default=0,
                    help="--align start: sessions active at once (0 = all)")
    ap.add_argument("--budget-gib", type=float, default=220,
                    help="--align budget: KV the active sessions reserve, GiB "
                         "(each its mean prompt size)")
    ap.add_argument("--session-rate", type=float, default=0.5,
                    help="--align poisson: session starts per second")
    ap.add_argument("--mean-gap", type=float, default=60.0,
                    help="--align poisson: mean seconds between rounds of a session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hbm", type=int, default=HBM_BLOCKS,
                    help="HBM blocks for the printed simulation")
    ap.add_argument("--dram-gib", type=int, default=DRAM_GIB,
                    help="Mooncake GiB per TP rank for the printed simulation")
    a = ap.parse_args()

    by_sess: OrderedDict = OrderedDict()
    for i, line in enumerate(open(a.trace)):
        r = json.loads(line)
        r["src_line"] = i
        by_sess.setdefault(r["session_id"], []).append(r)
    sessions = list(by_sess.values())
    for s in sessions:
        s.sort(key=lambda r: r["round_index"])

    if a.align == "end":
        rows = end_aligned(sessions)
    elif a.align == "start":
        rows = round_robin(sessions, a.window or len(sessions))
    elif a.align == "budget":
        rows = budget_rr(sessions, a.budget_gib / BLOCK_GIB)
    else:
        rows = poisson(sessions, a.session_rate, a.mean_gap, a.seed)
    last: dict = {}
    for r in rows:
        assert last.get(r["session_id"], -1) < r["round_index"], "session order broken"
        last[r["session_id"]] = r["round_index"]
        if a.align != "poisson":
            r["timestamp"] = 0
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    detail = {"end": "", "start": f" window {a.window}", "budget": f" budget {a.budget_gib:g} GiB",
              "poisson": f" session_rate {a.session_rate}/s mean_gap {a.mean_gap}s seed {a.seed}"}
    print(f"{len(rows)} requests, {len(sessions)} sessions, align {a.align}{detail[a.align]} -> {a.out}")
    live = sorted(live_blocks(rows))
    q = lambda p: live[int(p * (len(live) - 1))] * BLOCK_GIB  # noqa: E731
    print(f"KV of the active sessions' latest prompts: p10 {q(.1):.0f} / p50 {q(.5):.0f} / "
          f"p90 {q(.9):.0f} / max {live[-1] * BLOCK_GIB:.0f} GiB")
    if a.align == "poisson":
        ts = [r["timestamp"] / 1000 for r in rows]
        print(f"span {ts[-1] / 60:.1f} min; 50% / 90% of requests sent by "
              f"{ts[len(ts) // 2] / 60:.1f} / {ts[len(ts) * 9 // 10] / 60:.1f} min")
    print(f"simulated (hbm {a.hbm} blocks, dram {a.dram_gib} GiB/rank):",
          fmt(summarize(simulate(rows, a.hbm, dram_blocks(a.dram_gib)))))


if __name__ == "__main__":
    main()

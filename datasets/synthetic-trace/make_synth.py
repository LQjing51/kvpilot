"""A synthetic agent trace built to separate LRU < belady < KVPilot.

The three placements keep different things in the fast tiers:

  LRU     whatever was touched last, including sessions that have ended;
  belady  whatever is used again soonest, whether or not its load would be
          hidden by this turn's compute;
  KVPilot whatever would stall the GPU if it were not there.

Four kinds of session, mixed by `--cycle` (turns of each per cycle):

  warm   a short prefix reused every few requests: small enough that every
         placement holds it, so LRU keeps a real hit rate and the baseline is
         not "everything from SSD".
  keep   load-bound: a long prefix and a short suffix, so a prefix read from
         SSD is almost pure GPU idle. Reused late, so belady drops it first --
         the mistake the design targets.
  hide   compute-bound: a suffix whose compute covers even an SSD read of its
         prefix. Reused soon, so belady holds it in HBM although it would
         lose nothing on SSD.
  dead   ends after `--dead-turns` turns, then a fresh session takes its slot.
         Its last blocks are the most recently touched of all, so LRU holds
         them although nobody reads them again; belady and KVPilot drop them.
         Off by default: every dead session's first turn computes its whole
         prefix in every placement, which only dilutes the differences, and
         LRU loses without it (its write-through DRAM mostly copies HBM).

Sizing: `warm` + `keep` should fit in HBM + DRAM, `+ hide` should not, so
belady spills `keep` (used later) while KVPilot spills `hide`. A kind's reuse
gap is its session count times the cycle length over its turns per cycle; the
gaps decide what belady keeps, so keep `hide` and `warm` gaps below `keep`'s.

`warm` and `keep` are static by default (`--static`): each turn re-reads the
same context with a new suffix, so a `keep` turn computes just its suffix and
its prefix load is nearly all exposed. A growing session would also recompute
the partial block its previous turn ended in -- half a hybrid block (768
tokens on Qwen3.6) on average, six times the suffix. Prefixes are multiples of
3072 (every model's block) for that reason. `hide` grows: a hybrid model keeps
a state only where a turn ends, and its suffix is longer than a block.

Rows are Mooncake replay lines (vllm-dev/benchmarks/multi_turn): one request
per line, each session's turns in order, hash ids continuing the previous
turn's prompt (and output, when the session grows).

    python3 make_synth.py --out synth.jsonl
    python3 sim_synth.py synth.jsonl qwen3-30b,qwen3.6-fp8,qwen3.8 lru,belady,exclb
"""
import argparse
import json
import random
from pathlib import Path

# Tokens per trace hash id, as every KVPilot trace.
BLOCK = 512
KINDS = ("warm", "keep", "hide", "dead")


def build(a):
    rng = random.Random(a.seed)
    nid = iter(range(1, 1 << 30))
    spec = {k: (getattr(a, f"{k}_prefix"), getattr(a, f"{k}_suffix"), getattr(a, f"{k}_turns"))
            for k in KINDS}
    cycle = dict(x.split(":") for x in a.cycle.split(","))
    cycle = {k: int(v) for k, v in cycle.items()}
    static = set(filter(None, a.static.split(",")))
    for k in static:
        # A hybrid's state is where the previous turn ended: inside the shared
        # prefix only if the prefix ends on a block and the suffix is shorter
        # than one. Blocks: 512 (Qwen3-30B), 1536 (Qwen3.6), 1024 (Qwen3.8).
        assert spec[k][0] % 3072 == 0 and spec[k][1] < 1024, (k, spec[k])

    def start(kind, joined=0):
        prefix, suffix, turns = spec[kind]
        tokens = prefix if kind in static else prefix + joined * (suffix + a.output)
        return f"{kind}-{next(nid)}", {
            "tokens": tokens, "left": turns - joined,
            "hashes": [next(nid) for _ in range(tokens // BLOCK + 2)]}

    sess, queues = {}, {}
    for kind in KINDS:
        queues[kind] = []
        for _ in range(getattr(a, kind)):
            sid, s = start(kind, rng.randrange(spec[kind][2]))
            sess[sid] = s
            queues[kind].append(sid)
    plan = [k for k in KINDS for _ in range(cycle.get(k, 0))]
    rng.shuffle(plan)
    out, turn = [], {k: 0 for k in KINDS}
    while len(out) < a.n:
        for kind in plan:
            if len(out) >= a.n or not queues[kind]:
                continue
            q = queues[kind]
            i = turn[kind] % len(q)
            sid = q[i]
            turn[kind] += 1
            s = sess[sid]
            n_tok = s["tokens"] + spec[kind][1]
            ids = s["hashes"][: s["tokens"] // BLOCK]
            if kind in static:
                # This turn's own tokens are new; the context before them is not.
                ids = ids + [next(nid) for _ in range(n_tok // BLOCK - len(ids))]
            else:
                ids = s["hashes"][: n_tok // BLOCK]
                s["tokens"] = n_tok + a.output
            out.append({"timestamp": 0, "session_id": sid, "round_index": 0,
                        "input_length": n_tok, "prefix_length": 0, "output_length": a.output,
                        "hash_ids": ids})
            while len(s["hashes"]) < s["tokens"] // BLOCK + 2:
                s["hashes"].append(next(nid))
            s["left"] -= 1
            if s["left"] <= 0:
                sid2, s2 = start(kind)
                q[i], sess[sid2] = sid2, s2
    return out, cycle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1500, help="requests")
    p.add_argument("--cycle", default="warm:6,keep:3,hide:1", help="turns per cycle")
    for k, (n, pre, suf, turns) in {"warm": (12, 6144, 128, 1000), "keep": (56, 30720, 128, 1000),
                                    "hide": (18, 40960, 2560, 16), "dead": (0, 30720, 128, 2)}.items():
        p.add_argument(f"--{k}", type=int, default=n)
        p.add_argument(f"--{k}-prefix", type=int, default=pre)
        p.add_argument(f"--{k}-suffix", type=int, default=suf)
        p.add_argument(f"--{k}-turns", type=int, default=turns)
    p.add_argument("--static", default="warm,keep",
                   help="kinds whose every turn re-reads the same context (no growth)")
    p.add_argument("--output", type=int, default=16, help="output tokens per turn")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out, cycle = build(a)
    n = {k: sum(r["session_id"].startswith(k) for r in out) for k in KINDS}
    ws = {k: getattr(a, k) * (sum(r["input_length"] for r in out if r["session_id"].startswith(k))
                              / max(n[k], 1)) for k in KINDS}
    clen = sum(cycle.values())
    gaps = {k: getattr(a, k) * clen / cycle[k] for k in KINDS if cycle.get(k)}
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {a.out}: {len(out)} req, {len(set(r['session_id'] for r in out))} sessions, "
          f"prompt mean {sum(r['input_length'] for r in out) / len(out) / 1e3:.1f}K")
    print("  " + "  ".join(f"{k}: {n[k]} turns, 工作集 {ws[k] / 1e6:.2f}M, 复用间隔 {gaps.get(k, 0):.0f}"
                           for k in KINDS))


if __name__ == "__main__":
    main()

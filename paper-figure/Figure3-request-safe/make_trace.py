"""Per-request HBM/DRAM/SSD-safe traces (prefix>0 only).

    python3 make_trace.py toolagent # -> raw/toolagent/trace.jsonl
    python3 make_trace.py swe       # -> raw/swe/trace.jsonl
    python3 make_trace.py tracelab  # -> raw/tracelab/part{0,1}/trace.jsonl
    python3 make_trace.py swe_hbm1to2 --stop 6026  # -> raw/swe_hbm1to2/part{0,1}/
    python3 make_trace.py swe_hbm1to2_ssdmax --stop 2136  # -> raw/swe_hbm1to2_ssdmax/
    python3 make_trace.py swe_reqprio --stop 5252  # -> raw/swe_reqprio/part{0,1}/
    python3 make_trace.py tracelab_reqprio --stop 2267  # -> raw/tracelab_reqprio/part{0,1}/
    python3 make_trace.py swe --raw raw/ssdcap --stride 3
    python3 make_trace.py swe --start 2000 --stop 5140 --part part1
    python3 make_trace.py swe_blockprio_k160 --model qwen3.6 --raw raw/qwen3.6 --stop 1300
    python3 make_trace.py swe_blockprio_k160 --model qwen3.6-tp4 --raw raw/qwen3.6-tp4 --stop 1300
    python3 make_trace.py swe_blockprio_k160 --model qwen3.8 --raw raw/qwen3.8 --stop 1300

--raw puts the run under another root (one per loader configuration); --stride
keeps every Nth kept line (prefix and seen set still follow every line), so a
shorter run covers the same range of shapes. A stride that brings a PARTS
dataset under one SSD quota runs it as a single part. --start / --stop pick
source lines [start, stop) (prefixes still count every earlier line), and
--part writes them, unsplit, to <raw>/<name>/<part>/ so a dataset can be
measured in pieces on separate services.

Prefix = longest leading full-512 blocks already seen in an earlier line
(infinite pool), capped so ≥1 block is still computed, in whole vLLM blocks of
the model (--model; a hybrid model's hit ends at a state, one per block
boundary a prompt ends at, so warm's prompt is the prefix). Lines with prefix
0 are skipped entirely (not replayed, not classified). Their blocks still
enter the seen set so later lines keep the same theoretical prefix. Every row
records the model's block and KV layers for the source checks.

Only the lengths come from the trace: each line replays on fresh hash ids, so
its warm computes and stores the whole prefix, and [1] / [2] / [4] do not
depend on what earlier lines left in HBM, DRAM or SSD. Each line is warm, [1],
[2], [4] with nothing else; its [4] is sent after the next line's [2]: that
line's warm gives the prefix time to reach SSD, and [4] still follows a
request rather than idle time. Misses are retried (retry.py).

A dataset listed in PARTS writes more KV than one service's SSD quota
(MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES, 4 TiB per TP rank), so it is split
into contiguous parts of about equal KV, each replayed on a fresh service.
"""
import argparse
import json
from pathlib import Path

# Tokens per trace hash id.
BLOCK, LINES = 512, 2000
FRESH, HEAD = 10**9, 2 * 10**9
PARTS = {"tracelab": 2, "swe_hbm1to2": 2, "swe_reqprio": 2, "tracelab_reqprio": 2}
# TP ranks, vLLM block, KV layers, bf16 K+V bytes per token (all TP ranks;
# Qwen3.6's 2 KV heads are replicated at TP 4), and whether the prefill side
# computes the prompt but its last token (hybrid models under P/D: NIXL leaves
# the last token, and its state, to the decode side).
MODELS = {
    "qwen3-30b": {"tp": 2, "block": 512, "kv_layers": 48, "kv_bytes": 48 * 4 * 128 * 2 * 2,
                  "drops_last": False},
    "qwen3.6": {"tp": 2, "block": 1536, "kv_layers": 10, "kv_bytes": 10 * 2 * 256 * 2 * 2,
                "drops_last": True},
    "qwen3.6-tp4": {"tp": 4, "block": 1024, "kv_layers": 10, "kv_bytes": 10 * 4 * 256 * 2 * 2,
                    "drops_last": True},
    # Qwen3.8-Flash-Next: 12 QSA layers, 2 KV heads (replicated at TP 4), each
    # value carrying 64 B per token of indexer keys.
    "qwen3.8": {"tp": 4, "block": 1024, "kv_layers": 12,
                "kv_bytes": 12 * 4 * (256 * 2 * 2 + 64), "drops_last": True},
}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRACES = {
    "swe_blockprio_k160": ROOT / "datasets/block-priority-trace/swe_blockprio_k160.jsonl",
    "toolagent": ROOT / "datasets/mooncake-trace/toolagent_trace.jsonl",
    "swe": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_mooncake.jsonl",
    "swe_hbm1to2": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_hbm1to2.jsonl",
    "swe_hbm1to2_ssdmax": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_hbm1to2_ssdmax.jsonl",
    "tracelab": ROOT / "datasets/agent-traces/tracelab/tracelab_mooncake.jsonl",
    "swe_reqprio": ROOT / "datasets/req-priority-trace/swe_reqprio.jsonl",
    "tracelab_reqprio": ROOT / "datasets/req-priority-trace/tracelab_reqprio.jsonl",
}


def row(src, kind, tier, hids, length, p, model, delay_ms=0):
    r = {"timestamp": 0, "input_length": length, "output_length": 1, "hash_ids": hids,
         "src_line": src, "kind": kind, "tier": tier, "prefix": p, "tp": model["tp"],
         "block": model["block"], "kv_layers": model["kv_layers"],
         "drops_last": model["drops_last"]}
    if tier:
        r["extra_body"] = {"cache_use_control": [tier], "cache_use_max_tokens": p}
    if delay_ms:
        r["delay_ms"] = delay_ms
    return r


def lines(trace, model, stop=LINES):
    seen = set()
    block = model["block"]
    # The prefill side's hit stops a token short of what it computes.
    last = 2 if model["drops_last"] else 1
    for src, d in enumerate(map(json.loads, open(trace))):
        if src == stop:
            return
        n, ids = d["input_length"], d["hash_ids"]
        full = ids[: n // BLOCK]
        k = 0
        while k < len(full) and full[k] in seen:
            k += 1
        seen.update(full)
        yield src, ids, n, max(0, min(k * BLOCK // block, (n - last) // block)) * block


class Fresh:
    def __init__(self, start=FRESH):
        self.next = start

    def __call__(self, n):
        self.next += n
        return list(range(self.next - n, self.next))


def line_rows(src, n, p, tiers, fresh, heads, model, delay_ms=0):
    """Warm plus one test per tier, all on a new p-token prefix, each test
    with its own suffix of the same length; [4] after delay_ms.

    The first block draws from `heads`: the replay maps ids to tokens modulo
    the vocab, so ids repeat tokens after ~150K, and only a request's first
    block has a cache key that depends on its token alone.
    """
    pre = heads(1) + fresh(p // BLOCK - 1)
    nsuf = -(-(n - p) // BLOCK)
    if model["drops_last"]:
        # One token more, so the prefill side computes the prefix and leaves
        # its state at p.
        out = [row(src, "warm", 0, pre + fresh(1), p + 1, p, model)]
    else:
        out = [row(src, "warm", 0, pre, p, p, model)]
    for t in tiers:
        out.append(row(src, f"t{t}", t, pre + fresh(nsuf), n, p, model,
                       delay_ms if t == 4 else 0))
    return out


def defer_ssd(groups):
    """Each line's [4] right after the next line's [2]."""
    out, held = [], []
    for g in groups:
        out += [r for r in g if r["kind"] != "t4"] + held
        held = [r for r in g if r["kind"] == "t4"]
    return out + held


def kv_blocks(rows):
    return len({h for r in rows for h in r["hash_ids"]})


def split(groups, parts):
    """`parts` contiguous runs of line groups with about equal KV."""
    total, acc, out, cur = sum(map(kv_blocks, groups)), 0, [], []
    for g in groups:
        if cur and len(out) < parts - 1 and acc >= total * (len(out) + 1) / parts:
            out.append(cur)
            cur = []
        cur.append(g)
        acc += kv_blocks(g)
    return out + [cur]


def main(name, raw="raw", stride=1, start=0, stop=LINES, part=None, model="qwen3-30b"):
    out_dir = HERE / raw / name
    m = MODELS[model]
    fresh, heads, groups, skipped, kept = Fresh(FRESH), Fresh(HEAD), [], 0, 0
    for src, _ids, n, p in lines(TRACES[name], m, stop):
        if src < start:
            continue
        if not p:
            skipped += 1
            continue
        kept += 1
        if (kept - 1) % stride:
            continue
        groups.append(line_rows(src, n, p, (1, 2, 4), fresh, heads, m))
    if part:
        parts, dirs = [groups], [out_dir / part]
    else:
        parts = split(groups, max(1, -(-PARTS.get(name, 1) // stride)))
        dirs = [out_dir / f"part{i}" for i in range(len(parts))] if len(parts) > 1 else [out_dir]
    for d in set(dirs) if part else {out_dir, *dirs}:
        if any(d.glob("prefill-*.log*")) or (d / "benchmark.log").exists():
            raise SystemExit(f"{d} holds logs of an earlier run; move them aside first")
    if len(parts) > 1:
        (out_dir / "trace.jsonl").unlink(missing_ok=True)
    for d, part in zip(dirs, parts):
        d.mkdir(parents=True, exist_ok=True)
        rows = defer_ssd(part)
        with open(d / "trace.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tb = kv_blocks(rows) * BLOCK * m["kv_bytes"] / 1e12
        print(f"{d.relative_to(HERE)}: lines {len(part)}, requests {len(rows)}, "
              f"KV written ~{tb:.1f} TB")
    print(f"{name}: lines {start}-{stop}: kept {len(groups)} of {kept} (prefix>0, stride {stride}), "
          f"skipped {skipped} (prefix=0); next fresh id {fresh.next}, heads {heads.next - HEAD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(TRACES))
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=LINES)
    ap.add_argument("--part", help="write <raw>/<name>/<part>/ only, unsplit")
    ap.add_argument("--model", choices=list(MODELS), default="qwen3-30b")
    a = ap.parse_args()
    main(a.name, a.raw, a.stride, a.start, a.stop, a.part, a.model)

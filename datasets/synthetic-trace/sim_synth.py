"""Simulate a synthetic trace (make_synth.py) on the three models.

    python3 sim_synth.py <trace.jsonl> [models] [--arms lru,belady,exclb]

Capacity per model as Eval-EXP1: Qwen3-30B's defaults, the hybrids aligned to
the same tokens plus their sessions' state blocks (K = live sessions).
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datasets/block-priority-trace"))
import sim_placement as sp  # noqa: E402

CAP = {"qwen3-30b": (2361, 0, 100), "qwen3.6-fp8": (787, 3, 21), "qwen3.8": (1180, 4, 27)}
ARMS = ("lru", "belady", "incp", "exclu", "exclb")


BLOCK = 512


def next_suffix(rows):
    """Each turn's next turn's suffix: the tokens the next turn of its session
    computes past the blocks this turn leaves, i.e. past the hash ids the two
    share (the oracle signal the placements read)."""
    nxt = [None] * len(rows)
    last = {}
    for i, r in enumerate(rows):
        s = r["session_id"]
        if s in last:
            j, prev = last[s]
            shared = 0
            for a, b in zip(prev["hash_ids"], r["hash_ids"]):
                if a != b:
                    break
                shared += 1
            nxt[j] = max(0, r["input_length"] - shared * BLOCK)
        last[s] = (i, r)
    return nxt


def main():
    path = sys.argv[1]
    models = (sys.argv[2].split(",") if len(sys.argv) > 2 else list(CAP))
    arms = (sys.argv[3].split(",") if len(sys.argv) > 3 else ARMS)
    rows = [json.loads(x) for x in open(path)]
    nxt = next_suffix(rows)
    live = len(set(r["session_id"] for r in rows[: len(rows) // 4]))
    # Warm-up: until each kind's sessions alive from the start have all
    # computed their first turn, i.e. its first reuse (dead sessions keep
    # starting; theirs are steady state).
    seen, rep = set(), {}
    for i, r in enumerate(rows):
        k = r["session_id"].split("-")[0]
        if r["session_id"] in seen:
            rep.setdefault(k, i)
        seen.add(r["session_id"])
    warm = max(live, max(i for k, i in rep.items() if k != "dead"))
    print(f"{Path(path).name}: {len(rows)} requests, {len(set(r['session_id'] for r in rows))} sessions, "
          f"prompt mean {st.mean(r['input_length'] for r in rows) / 1e3:.1f}K, warm-up {warm}")
    for model in models:
        prof = sp.PROFILES[model]
        blocks, states, dram = CAP[model]
        M = sp.tm.TimeModel.load(str(prof["coeffs"]), prof["block"])
        M.thresh = 0.01
        B = prof["block"]
        reqs = sp.make_reqs(rows, nxt, prof)
        hbm = blocks + states * live
        db = int(sp.WATERMARK * dram / prof["block_gib"])
        base = None
        for arm in arms:
            recs = [r for r in sp.Sim(arm, reqs, M, hbm, db, 16, sp.default_cost(14.0, prof), prof).run()
                    if r["i"] >= warm]
            t = st.mean(r["ttft"] for r in recs)
            base = base or t
            tot = sum(r["hbm"] + r["dram"] + r["ssd"] for r in recs) or 1
            ex = st.mean(M.e2e(r["p"], r["s"], r["dram"] * B, r["ssd"] * B) - M.compute(r["p"], r["s"])
                         for r in recs)
            by = {}
            for r in recs:
                k = next((x for x in ("keep", "hide", "dead") if reqs[r["i"]]["sid"].startswith(x)), "?")
                by.setdefault(k, []).append((r["ssd"] * B, M.e2e(r["p"], r["s"], r["dram"] * B, r["ssd"] * B)
                                             - M.compute(r["p"], r["s"])))
            print(f"  {model:12s} {arm:7s} TTFT {t:7.1f} ({t / base * 100 - 100:+6.1f}%)  exposed {ex:6.1f} "
                  f"({ex / t * 100:4.1f}%)  prefix H/D/S "
                  + "/".join(f"{sum(r[k] for r in recs) / tot * 100:4.1f}" for k in ("hbm", "dram", "ssd"))
                  + "% | " + "  ".join(f"{k}: SSD {st.mean(x for x, _ in v) / 1e3:4.1f}K exposed "
                                       f"{st.mean(y for _, y in v):6.1f}" for k, v in sorted(by.items()))
                  + f" | {sum(r['blocked'] for r in recs) / 1e3:.0f}s blocked, "
                    f"{max(r['sent'] for r in recs) / 6e4:.0f}m")
        print()


if __name__ == "__main__":
    main()

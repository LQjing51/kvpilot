"""Eval-EXP1 results: per trace and model, each arm's TTFT and where its
prefix came from, and the model's required order checked pair by pair.

    python3 summarize.py [swe|tracelab ...]

Reads raw/<ds>/<model>/<arm>/ (run_exp1.sh). TTFT is the prefill worker's
end minus the client's send (utils/placement_ab/analyze.py); lines before the
trace's K (warm-up) are left out, and every arm is compared on the requests
all of them completed. A pair's difference comes with a 95% interval from
means of 50-request batches (the cache state carries over between requests).
"SSD first layer" is the p90 time to land layer 0 of the loads that read SSD:
a few ms when the disks serve only this run.
"""
import gzip
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "utils/placement_ab"))
import analyze  # noqa: E402

# Warm-up lines per trace (its K).
WARM = {"swe": 176, "tracelab": 250, "swe_rand": 144, "tl_rand": 160}
MODELS = {"q30": "Qwen3-30B-A3B (TP 2)", "q36": "Qwen3.6-35B-A3B (TP 4)",
          "q36f8": "Qwen3.6-35B-A3B-FP8 (TP 2)", "q38": "Qwen3.8-Flash-Next (TP 4)"}
ORDER = {"q30": ("lru", "belady", "incp", "exclu", "exclb"),
         "q36": ("lru", "belady", "exclb"), "q36f8": ("lru", "belady", "exclb"),
         "q38": ("lru", "belady", "exclb")}
# (slower, faster) pairs each model must show. Qwen3-30B: LRU > belady >
# blockwise and inclusive req > unblocking > blockwise (TTFT); belady and
# inclusive req are not compared.
CHAIN = (("lru", "belady"), ("belady", "exclb"))
PAIRS = {"q30": CHAIN + (("incp", "exclu"), ("exclu", "exclb")), "q36": CHAIN, "q36f8": CHAIN,
         "q38": CHAIN}
BATCH = 50


def ci(d):
    nb = len(d) // BATCH
    means = [statistics.mean(d[i * BATCH:(i + 1) * BATCH]) for i in range(nb)]
    return 1.96 * statistics.stdev(means) / nb ** 0.5 if nb > 1 else float("nan")


def ssd_first(d):
    pat = re.compile(r"KVPilot load req=(\d+)-\S+ .* ssd=([1-9]\d*) first=([\d.]+)ms")
    first = {}
    for line in gzip.open(d / "prefill-0.log.gz", "rt", errors="replace"):
        m = pat.search(line)
        if m:
            k = int(m[1])
            first[k] = max(first.get(k, 0.0), float(m[3]))
    return first


def check(runs, keys, model, name):
    """The model's required pairs; name maps an arm to the run used for it."""
    ok = done = True
    for slow, fast in PAIRS[model]:
        if slow not in name or fast not in name:
            done = False
            continue
        a, b = name[slow], name[fast]
        d = [runs[a]["ttft"][k] - runs[b]["ttft"][k] for k in keys]
        m, h = statistics.mean(d), ci(d)
        ok &= m > 0
        print(f"- {b} faster than {a} by {m:+.1f} ± {h:.1f} ms "
              f"({'significant' if m - h > 0 else 'within noise' if m > 0 else 'WRONG ORDER'})")
    if done:
        print(f"- required pairs (TTFT): {'hold' if ok else 'BROKEN'}")


def report(ds, model):
    base = HERE / "raw" / ds / model
    arms = [a for a in ORDER[model] if (base / a / "benchmark.log").exists()]
    if not arms:
        return
    # Repeats of an arm (run_exp1.sh REP=...): <arm>-<rep>/.
    reps = sorted(p.name for a in ORDER[model] for p in base.glob(f"{a}-*")
                  if (p / "benchmark.log").exists())
    runs = {a: analyze.load(base / a) for a in arms + reps}
    warm = WARM.get(ds, 0)
    keys = sorted(k for k in set.intersection(*(set(r["ttft"]) for r in runs.values())) if k >= warm)
    print(f"\n### {ds} / {MODELS[model]}: {len(keys)} requests (lines >= {warm})")
    # A repeat suffix with every arm is its own set (own table, vs its own
    # LRU); the other repeats go under the first runs.
    sets = {rep: [f"{a}-{rep}" for a in arms] for rep in sorted({r.split("-", 1)[1] for r in reps})}
    full = {rep: s for rep, s in sets.items() if all(a in runs for a in s)}
    extra = [r for r in reps if r.split("-", 1)[1] not in full]
    table(base, runs, keys, arms, arms + extra)
    check(runs, keys, model, {a: a for a in arms})
    for rep in sorted({r.split("-", 1)[1] for r in extra}):
        print(f"\nwith -{rep}:")
        check(runs, keys, model, {a: f"{a}-{rep}" if f"{a}-{rep}" in runs else a
                                  for a in ORDER[model] if f"{a}-{rep}" in runs or a in runs})
    for rep, s in full.items():
        print(f"\n#### all arms -{rep}")
        table(base, runs, keys, s, s)
        check(runs, keys, model, dict(zip(arms, s)))


def table(base, runs, keys, arms, rows):
    print("\n| arm | TTFT mean (ms) | vs LRU | p50 | p90 | prefix from HBM / DRAM / SSD | exposed load (s) "
          "| SSD first layer p90 (ms) | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    lru = statistics.mean(runs[arms[0]]["ttft"][k] for k in keys)
    for a in rows:
        r = runs[a]
        t = [r["ttft"][k] for k in keys]
        sf = ssd_first(base / a)
        sf = [sf[k] for k in keys if k in sf]
        hit = sum(r["pre"][k][0] for k in keys if k in r["pre"])
        loaded = sum(r["pre"][k][1] for k in keys if k in r["pre"])
        ssd = sum(min(r["ssd_tok"].get(k, 0), r["pre"][k][1]) for k in keys if k in r["pre"])
        tot = (hit + loaded) or 1
        err = sum(1 for line in open(base / a / "benchmark.log") if line.startswith("[ERR]"))
        print(f"| {a} | {statistics.mean(t):.1f} | {statistics.mean(t) / lru * 100 - 100:+.1f}% | "
              f"{analyze.pct(t, .5):.1f} | {analyze.pct(t, .9):.1f} | "
              f"{hit / tot * 100:.0f} / {(loaded - ssd) / tot * 100:.0f} / {ssd / tot * 100:.0f}% | "
              f"{sum(r['gpu'].get(k, 0.0) for k in keys) / 1e3:.1f} | "
              f"{analyze.pct(sf, .9) if sf else float('nan'):.1f} | {err} |")
    print()


def main():
    for ds in sys.argv[1:] or ["swe", "tracelab"]:
        for model in MODELS:
            report(ds, model)


if __name__ == "__main__":
    main()

"""Offline estimate of request-level priority eviction
(KVPILOT_PLACEMENT=inclusive-safe-reqwise) against LRU on a trace in send order: where each request's prefix loads from,
its estimated prefill e2e, and why DRAM-safe blocks still load from SSD.

    python3 dram_miss_sim.py [--lag 0]
    python3 dram_miss_sim.py --trace <interleaved.jsonl> --tiers model

The cache follows utils/trace_order/tier_sim.py, one request at a time (-p 1,
no queueing): partial write-through / allocate, block-wise eviction, HBM frees
a request last-block-first, DRAM evicts the oldest 10% past 90% and a lookup
refreshes the prompt front to back. Unlike tier_sim.py, an SSD hit is written
back to DRAM (load-through, as the service does outside exclusive).

Request priority: every block of a request shares one class, the request-safe
tier of the next request in its session (a table lookup, no prediction). HBM
and DRAM both keep four queues, evicted in order: no next request, SSD safe,
DRAM safe, HBM safe. Inside a queue the order is LRU, and a class change keeps
a block's last-access time. --lag N applies a request's verdict only after N
more requests have been sent (the decode of a -p 1 run ends later than the next
prefill starts); until then its blocks keep their previous class and new blocks
wait in the last-evicted queue.

Each loading request is scored against its own tier: a block from a slower
tier is a misread that exposes load time, from a faster one it spends faster
capacity. Its e2e is t1 / t2 / t4 when the prefix comes whole from HBM / DRAM /
SSD, otherwise max(L_ssd(p_ssd), L_dram(p_loaded), t1 + floor) with the fitted
request-safe time model; "over all-HBM" sums e2e - t1, the exposed load.
--tiers measured takes each request's tier and t1/t2/t4 from a request-safe
run (--raw, default the SWE one); --tiers model predicts them from (prefix,
suffix) with the fitted time model, for traces never measured. The profile line
gives what the remedies depend on: how often the safe tier changes between
turns, and the live blocks per next-turn tier against HBM / DRAM.

A block a DRAM-safe request reads from SSD is "harmful" when the request's
estimated e2e exceeds its all-HBM time by THRESH, otherwise "harmless". Each
harmful block is traced back to how it last left DRAM: evicted while its class
was SSD safe / no reuse, or evicted while DRAM safe, split by whether HBM held
it at the time.

Variants (on the request-priority arm unless lru):
  lru        no classes: the current LRU baseline
  exclusive  a DRAM copy of a block HBM holds is evicted right after SSD-safe
             ones; HBM evicting a DRAM- or HBM-safe block writes it back to DRAM
  dedup      exclusive without the write-back
  horizonN   a block's DRAM class is DRAM safe if any of the next N turns of its
             session is DRAM or HBM safe (HBM classes still use the next turn)
  watermark  evict 1% past 99% instead of 10% past 90%
  noalloc    a new block whose next request is SSD safe / no reuse skips DRAM
  opt        inside a class DRAM evicts the block used farthest in the future
             (Belady within class; a bound, needs the future)
  belady     no classes: HBM and DRAM both evict the block whose next reuse is
             farthest (classic Belady / MIN). Same capacities and watermark as
             lru; SSD hits still write back. Needs the future.
"""
import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG3 = ROOT / "paper-figure/Figure3-request-safe"
sys.path[:0] = [str(FIG3), str(ROOT / "utils/trace_order")]
import tier_sim  # noqa: E402
from fit_times import THRESH, times  # noqa: E402
from tier_logs import dataset  # noqa: E402

BLOCK = tier_sim.BLOCK
TRACE = ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_mooncake_interleaved.jsonl"
TIERS = ("HBM", "DRAM", "SSD")
HBM_CLASS = {None: 0, "SSD": 1, "DRAM": 2, "HBM": 3}
DRAM_CLASS = {None: 0, "SSD": 1, "DRAM": 2, "HBM": 3}
INF = float("inf")


def measured(raw=FIG3 / "raw/swe"):
    """src_line -> (prefix, input_length, tier, {t1, t2, t4}) from a request-safe run."""
    rows, res, _ = dataset(raw)
    out = {}
    for r in rows:
        e = {k: res.get((r["src_line"], k), (None, "m"))[0] for k in ("t1", "t2", "t4")}
        if r["kind"] == "t1" and None not in e.values():
            tier = ("SSD" if (e["t4"] - e["t1"]) / e["t1"] < THRESH
                    else "DRAM" if (e["t2"] - e["t1"]) / e["t1"] < THRESH else "HBM")
            out[r["src_line"]] = (r["prefix"], r["input_length"], tier, e)
    return out


def modeled(rows, k):
    """src_line -> (prefix, input_length, tier, {t1, t2, t4}) from the time model."""
    seen, out = set(), {}
    for r in rows:
        n_in = r["input_length"]
        full = r["hash_ids"][: n_in // BLOCK]
        hit = 0
        while hit < len(full) and full[hit] in seen:
            hit += 1
        seen.update(full)
        if hit and hit * BLOCK < n_in:
            th, td, ts = (float(x) for x in times(hit * BLOCK, n_in - hit * BLOCK, k))
            tier = "SSD" if (ts - th) / th < THRESH else "DRAM" if (td - th) / th < THRESH else "HBM"
            out[r["src_line"]] = (hit * BLOCK, n_in, tier, {"t1": th, "t2": td, "t4": ts})
    return out


def profile(rows, meas, fut, hbm_cap, dram_cap):
    tiers = Counter(m[2] for m in meas.values())
    flips = Counter()
    for r in rows:
        m, later = meas.get(r["src_line"]), fut[r["src_line"]]
        if m and m[2] == "SSD" and later:
            flips[later[0] != "SSD"] += 1
    nxt = {}
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["session_id"]].append((r["round_index"], i))
    for turns in by.values():
        turns.sort()
        for (_, i), later in zip(turns, turns[1:] + [None]):
            nxt[i] = later[1] if later else None
    delta = {t: np.zeros(len(rows) + 1) for t in ("HBM", "DRAM", "SSD", None)}
    gaps = []
    for i, r in enumerate(rows):
        j = nxt[i]
        if j is None:
            continue
        m = meas.get(rows[j]["src_line"])
        nb = r["input_length"] // BLOCK
        delta[m[2] if m else None][i + 1] += nb
        delta[m[2] if m else None][j + 1] -= nb
        gaps.append(j - i)
    live = {t: np.cumsum(d)[1:] for t, d in delta.items()}
    n = sum(tiers.values()) or 1
    allv = sum(live.values())
    spare = np.maximum(hbm_cap - live["HBM"], 0)
    print(f"{len(rows)} requests, {len(by)} sessions, turns of a session "
          f"{(np.median(gaps) if gaps else 0):.0f} requests apart; "
          f"tiers HBM/DRAM/SSD {100 * tiers['HBM'] / n:.0f}/{100 * tiers['DRAM'] / n:.0f}/"
          f"{100 * tiers['SSD'] / n:.0f}%; SSD-safe turn followed by DRAM/HBM "
          f"{flips[True]}/{sum(flips.values()) or 0}")
    print(f"live blocks (median/p90): all {np.median(allv):.0f}/{np.percentile(allv, 90):.0f}, "
          f"next HBM safe {np.median(live['HBM']):.0f}/{np.percentile(live['HBM'], 90):.0f}, "
          f"next DRAM safe {np.median(live['DRAM']):.0f}/{np.percentile(live['DRAM'], 90):.0f}; "
          f"DRAM-safe fits DRAM {100 * np.mean(live['DRAM'] <= dram_cap):.0f}%, "
          f"DRAM + spare HBM {100 * np.mean(live['DRAM'] <= dram_cap + spare):.0f}% of steps")


class Queues:
    """Blocks in per-class queues; eviction takes the lowest class first, then
    the smallest key. Default key is last-access time (LRU); Belady passes
    -next_use so the farthest reuse is smallest and leaves first. A class
    change keeps the block's key."""

    def __init__(self, n):
        self.heaps = [[] for _ in range(n)]
        self.cls, self.ts, self.tick = {}, {}, 0

    def __contains__(self, b):
        return b in self.cls

    def __len__(self):
        return len(self.cls)

    def _push(self, b):
        heapq.heappush(self.heaps[self.cls[b]], (self.ts[b], b))

    def put(self, b, c, key=None):
        self.tick += 1
        self.cls[b] = c
        self.ts[b] = self.tick if key is None else key
        self._push(b)

    def reclass(self, b, c):
        if b in self.cls and self.cls[b] != c:
            self.cls[b] = c
            self._push(b)

    def pop(self):
        for c, h in enumerate(self.heaps):
            while h:
                t, b = heapq.heappop(h)
                if self.cls.get(b) == c and self.ts[b] == t:
                    del self.cls[b], self.ts[b]
                    return b


class Pool:
    """Per-class queues; eviction takes the lowest class first, inside it the
    smallest key."""

    def __init__(self, n):
        self.heaps = [[] for _ in range(n)]
        self.cls, self.key = {}, {}
        self.count = [0] * n

    def __contains__(self, b):
        return b in self.cls

    def __len__(self):
        return len(self.cls)

    def put(self, b, c, key):
        if b in self.cls:
            self.count[self.cls[b]] -= 1
        self.count[c] += 1
        self.cls[b], self.key[b] = c, key
        heapq.heappush(self.heaps[c], (key, b))

    def reclass(self, b, c):
        if b in self.cls and self.cls[b] != c:
            self.put(b, c, self.key[b])

    def pop(self):
        for c, h in enumerate(self.heaps):
            while h:
                key, b = heapq.heappop(h)
                if self.cls.get(b) == c and self.key[b] == key:
                    del self.cls[b], self.key[b]
                    self.count[c] -= 1
                    return b, c
        return None, None


def estimate(k, t, kh, pd_, ps):
    if not pd_ and not ps:
        return t["t1"]
    if not kh and not ps:
        return t["t2"]
    if not kh and not pd_:
        return t["t4"]
    L = k["c_dram"] * (pd_ + ps) + k["f_dram"]
    fl = k["g_dram"] * pd_ + k["h_dram"]
    if ps:
        L = max(L, k["c_ssd"] * ps + k["f_ssd"])
        fl = k["g_ssd"] * ps + k["h_ssd"]
    return max(L, t["t1"] + max(0.0, fl))


def run(rows, meas, fut, k, opts, hbm_cap, dram_cap, lag, e2e=None, lease_ms=0):
    """e2e, if given, is filled with src_line -> estimated e2e (ms) of each scored request.

    lease_ms: Mooncake default_kv_lease_ttl. Lookup / put refresh a block's lease;
    BatchEvict skips blocks whose lease has not expired. Wall clock advances by
    each request's estimated prefill e2e (ms), matching the calibration against
    the 1P1D reqprio A/B. 0 disables leases (old behaviour).
    """
    high, evict = (0.99, 0.01) if "watermark" in opts else (tier_sim.HIGH, tier_sim.EVICT)
    excl = "exclusive" in opts or "dedup" in opts
    horizon = next((int(o[7:]) for o in opts if o.startswith("horizon")), 1)
    # Four DRAM classes matching HBM (None / SSD / DRAM / HBM). Exclusive inserts
    # a "HBM still holds a copy" tier between SSD and DRAM-safe.
    n_dram = 5 if excl else 4
    dup = 2 if excl else None
    prot = 3 if excl else 2  # DRAM-safe and above (for stats)
    nu = []
    last = {}
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        full = r["hash_ids"][: r["input_length"] // BLOCK]
        nu.append({b: last.get(b, INF) for b in full})
        for b in full:
            last[b] = i
    nu.reverse()
    hbm, dram, ssd = Queues(4), Pool(n_dram), set()
    want = {}
    evicted = {}
    promoted = {}
    pending = []
    lease_until = {}
    tick = 0
    wall = 0.0
    mis, why, harm = Counter(), Counter(), Counter()
    over, comp = 0.0, Counter()

    belady = "belady" in opts

    def key(i, b):
        return -nu[i].get(b, INF) if belady or "opt" in opts else tick

    def dclass(b):
        c = want.get(b, 3)
        if excl and c >= 2:
            return dup if b in hbm else c + 1
        return c

    def touch_lease(b):
        if lease_ms:
            lease_until[b] = wall + lease_ms

    for i, r in enumerate(rows):
        n_in = r["input_length"]
        full = r["hash_ids"][: n_in // BLOCK]
        kh = 0
        while kh < len(full) and full[kh] in hbm:
            kh += 1
        if kh * BLOCK >= n_in:
            kh -= 1
        hit = 0
        while hit < len(full) and (full[hit] in dram or full[hit] in ssd):
            hit += 1
        from_ssd = [b for b in full[kh:hit] if b not in dram]
        nd, ns = hit - kh - len(from_ssd) if hit > kh else 0, len(from_ssd)
        # Every prefix load, including cold-shape / unscored rows.
        if hit:
            for tier, nb in (("HBM", kh), ("DRAM", nd), ("SSD", ns)):
                comp[f"tok_{tier}"] += nb * BLOCK
            comp["tok_prefix"] += hit * BLOCK
        m = meas.get(r["src_line"])
        est = 0.0
        if hit and m and m[0] == hit * BLOCK and m[1] == n_in:
            req, t = m[2], m[3]
            for tier, nb in (("HBM", kh), ("DRAM", nd), ("SSD", ns)):
                mis[(req, tier)] += nb
            est = estimate(k, t, kh, nd * BLOCK, ns * BLOCK)
            if e2e is not None:
                e2e[r["src_line"]] = est
            over += est - t["t1"]
            comp["e2e_ms"] += est
            comp["scored"] += 1
            if req == "DRAM" and ns:
                bad = (est - t["t1"]) / t["t1"] >= THRESH
                harm[bad] += ns
                comp["reqs_ssd"] += 1
                comp["exposed_ssd"] += est - t["t1"]
                if bad:
                    for b in from_ssd:
                        why[evicted.get(b, ("never", None))] += 1
        later = fut[r["src_line"]]
        c_h = HBM_CLASS[later[0] if later else None]
        c_d = max((DRAM_CLASS[t] for t in later[:horizon]), default=0)
        if "lru" in opts or belady:
            # Single class each: lru uses tick keys; belady uses -next_use.
            c_h, c_d = 0, 0 if belady else 3
        for b in full:
            if lag:
                want.setdefault(b, 3)  # pending = HBM-safe class until verdict
            else:
                want[b] = c_d
        tick += 1
        for b in full:
            if b in dram:
                dram.put(b, dram.cls[b], key(i, b))
                touch_lease(b)
                if promoted.pop(b, None) is not None:
                    comp["promoted_used"] += 1
        for b in from_ssd:
            dram.put(b, dclass(b), key(i, b))
            touch_lease(b)
            promoted[b] = i
            comp["promoted"] += 1
        for b in full[max(kh, hit):]:
            ssd.add(b)
            if want[b] or "noalloc" not in opts:
                dram.put(b, dclass(b), key(i, b))
                touch_lease(b)
        for b in reversed(full):
            hk = key(i, b) if belady else None
            hbm.put(b, hbm.cls.get(b, 3) if lag else c_h, key=hk)
        while len(hbm) > hbm_cap:
            victim = hbm.pop()
            if excl and want.get(victim, 0) >= 2:
                # Write back / keep as protected DRAM-or-HBM class (shifted if excl).
                cls = want[victim] + 1
                if victim in dram:
                    dram.reclass(victim, cls)
                elif "exclusive" in opts:
                    dram.put(victim, cls, key(i, victim))
                    touch_lease(victim)
                    comp["written_back"] += 1
        if excl:
            for b in full:
                if b in dram and b in hbm and want.get(b, 0) >= 2:
                    dram.reclass(b, dup)
        # Prefill finishes before Mooncake's watermark eviction of this put.
        # Cold / unscored requests may already carry their e2e in the out-dict.
        if not est:
            if e2e is not None and r["src_line"] in e2e:
                est = e2e[r["src_line"]]
            elif m:
                est = m[3]["t1"]
        wall += est
        if len(dram) > high * dram_cap:
            need = int(evict * len(dram))
            comp["events"] += 1
            comp["dup"] += sum(b in hbm for b in dram.cls)
            comp["top"] += sum(c >= prot for c in dram.cls.values())
            comp["size"] += len(dram)
            if lease_ms:
                # Mooncake BatchEvict skips unexpired leases; take the oldest
                # expired objects in class order, up to need.
                free = [(dram.cls[b], dram.key[b], b) for b in dram.cls
                        if lease_until.get(b, 0) <= wall]
                comp["leased"] += len(dram) - len(free)
                free.sort()
                victims = free[:need]
                keep = [(b, dram.cls[b], dram.key[b]) for b in dram.cls
                        if b not in {x[2] for x in victims}]
                dram = Pool(n_dram)
                for b, c, kk in keep:
                    dram.put(b, c, kk)
                for _, c, b in victims:
                    # classes 0/1 = no-next / SSD; dup (excl) and DRAM/HBM are protected
                    evicted[b] = ("ssd-safe" if c <= 1 else "dram-safe", b in hbm)
                    lease_until.pop(b, None)
                    comp["evicted_top" if c >= prot else "evicted_low"] += 1
                    if promoted.pop(b, None) is not None:
                        comp["promoted_wasted"] += 1
            else:
                for _ in range(need):
                    b, c = dram.pop()
                    if b is None:
                        break
                    evicted[b] = ("ssd-safe" if c <= 1 else "dram-safe", b in hbm)
                    comp["evicted_top" if c >= prot else "evicted_low"] += 1
                    if promoted.pop(b, None) is not None:
                        comp["promoted_wasted"] += 1
        pending.append((full, c_h, c_d))
        while len(pending) > lag:
            blocks, ch, cd = pending.pop(0)
            for b in blocks:
                hbm.reclass(b, ch)
                want[b] = cd
                if b in dram:
                    dram.reclass(b, dclass(b))
    return mis, why, harm, over, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm", type=int, default=tier_sim.HBM_BLOCKS)
    ap.add_argument("--dram", type=int, default=tier_sim.DRAM_BLOCKS)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--trace", type=Path, default=TRACE)
    ap.add_argument("--tiers", choices=("measured", "model"), default="measured")
    ap.add_argument("--raw", default="raw/swe",
                    help="--tiers measured: request-safe run under Figure3-request-safe whose "
                         "src_line numbering the trace's src_line follows")
    ap.add_argument("--quiet", action="store_true", help="one line per variant")
    ap.add_argument("--lease-ms", type=float, default=0,
                    help="Model Mooncake TTL lease shielding (ms). 0 = off (default), "
                         "matching priority-eviction refcnt pins that only last for "
                         "the load. Use 5000 to replay the old TTL behaviour.")
    ap.add_argument("--variants", default="lru,none,horizon99,noalloc,opt,watermark,exclusive,"
                    "dedup+watermark,exclusive+watermark,exclusive+watermark+opt")
    a = ap.parse_args()
    k = json.loads((FIG3 / "fit_times_swe.json").read_text())
    rows = [json.loads(x) for x in open(a.trace)]
    meas = measured(FIG3 / a.raw) if a.tiers == "measured" else modeled(rows, k)
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session_id"]].append(r)
    fut = {}
    for turns in by_sess.values():
        turns.sort(key=lambda r: r["round_index"])
        tiers = [meas[r["src_line"]][2] if r["src_line"] in meas else None for r in turns]
        for j, r in enumerate(turns):
            fut[r["src_line"]] = tiers[j + 1:]
    rank = {t: j for j, t in enumerate(TIERS)}
    print(f"{a.trace.name}, tiers {a.tiers}; HBM {a.hbm} blocks, DRAM {a.dram} blocks, "
          f"verdict lag {a.lag}, lease {a.lease_ms:g} ms")
    profile(rows, meas, fut, a.hbm, a.dram)
    for v in a.variants.split(","):
        opts = set(v.split("+")) - {"none"}
        mis, why, harm, over, comp = run(rows, meas, fut, k, opts, a.hbm, a.dram, a.lag,
                                         lease_ms=a.lease_ms)
        total = sum(mis.values())
        low = sum(c for (rq, t), c in mis.items() if rank[t] > rank[rq])
        ev = max(comp["events"], 1)
        tok = max(comp["tok_prefix"], 1)
        hit_share = "/".join(f"{100 * comp[f'tok_{t}'] / tok:.1f}" for t in TIERS)
        scored_share = ("n/a" if not total else "/".join(
            f"{100 * sum(mis[(rq, t)] for rq in TIERS) / total:.1f}" for t in TIERS))
        print(f"{v:45s} low {100 * low / max(total, 1):4.1f}% | DRAM-safe from SSD {mis[('DRAM', 'SSD')]:6d} "
              f"(harmful {harm[True]:6d}) HBM-safe below HBM {mis[('HBM', 'DRAM')] + mis[('HBM', 'SSD')]:5d}"
              f" | over all-HBM {over / 1e3:5.1f} s, e2e mean "
              f"{comp['e2e_ms'] / max(comp['scored'], 1):6.1f} ms | written back to DRAM: SSD hits "
              f"{comp['promoted']}, HBM evictions {comp['written_back']} | "
              f"prefix tokens from HBM/DRAM/SSD {hit_share}% (scored blocks {scored_share}%)")
        if a.quiet:
            continue
        print(f"  DRAM-safe requests reading some SSD: {comp['reqs_ssd']}, their exposed load "
              f"{comp['exposed_ssd'] / 1e3:.1f} s")
        print("  harmful DRAM-safe-from-SSD blocks, by how they left DRAM: " + ", ".join(
            f"{w}{'' if h is None else (' (in HBM)' if h else ' (not in HBM)')} {c}"
            for (w, h), c in why.most_common()))
        print(f"  DRAM at eviction time: {comp['size'] / ev:.0f} blocks, "
              f"{100 * comp['dup'] / max(comp['size'], 1):.0f}% also in HBM, "
              f"{100 * comp['top'] / max(comp['size'], 1):.0f}% DRAM/HBM safe class; "
              f"{comp['events']} eviction rounds evicted {comp['evicted_low']} low-class and "
              f"{comp['evicted_top']} DRAM/HBM-safe blocks")
        if comp["promoted"]:
            print(f"  SSD hits written back {comp['promoted']}: used from DRAM {comp['promoted_used']}, "
                  f"evicted before use {comp['promoted_wasted']}")
        print("  blocks should/from " + "  ".join(
            f"{rq}->{t}:{mis[(rq, t)]}" for rq in TIERS for t in TIERS))


if __name__ == "__main__":
    main()

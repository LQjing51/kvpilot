"""Offline estimate of exclusive placement
(KVPILOT_PLACEMENT=exclusive-penalty-reqwise) against LRU and request
priority, on a trace in send order.

    python3 excl_sim.py --trace <trace.jsonl> --raw raw/swe_reqprio [--lag 10]

Exclusive placement keeps one copy of each block. A prefill leaves the whole
prompt in HBM, nothing is written below it. When the request's verdict arrives
(the request-safe tier of the next turn of its session, `--lag` requests
later), its blocks move to that tier: HBM safe stays, DRAM safe is written to
DRAM and leaves HBM, SSD safe is written to SSD and leaves HBM. With no next
turn a block keeps the copy it has below (a DRAM copy becomes first to spill),
or is written to SSD. A block already in the target tier is not rewritten; one
in another lower tier is moved. A block several sessions share sits in the
fastest tier any of them still needs (the newest turn of each session holds a
claim, pending counts as HBM).

Nothing is lost. Past HBM capacity a block with a copy below just leaves, else
the block whose next turn exposes least from DRAM (t2 - t1; pending last) is
written to DRAM. Past 90% DRAM spills 10% to SSD, least exposed by the move
(t4 - t2) first. SSD is unbounded.
--lossy is the first version: HBM drops pending / HBM-safe blocks oldest
first, DRAM past 95% drops the oldest 10%, no next turn drops everywhere.
e2e uses the same estimate as dram_miss_sim.py; a request whose prefix was cut
by a loss is timed with the fitted model at the shorter prefix.
"""
import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from dram_miss_sim import FIG3, TIERS, estimate, measured, modeled, run, tier_sim, times

IDX = {"HBM": 0, "DRAM": 1, "SSD": 2}


def run_excl(rows, meas, fut, nexp, k, hbm_cap, dram_cap, lag, lossy=False):
    """nexp: src_line -> the next turn's (t2 - t1, t4 - t2) ms, floored at 0:
    the load a move HBM -> DRAM / DRAM -> SSD exposes. lossy: the first
    version (drop on eviction, drop on no next turn)."""
    hbm: OrderedDict = OrderedDict()
    dram: OrderedDict = OrderedDict()
    ssd: set = set()
    dpen = {}  # DRAM block -> load exposed if it went to SSD (ms)
    cnt = defaultdict(lambda: [0, 0, 0])
    claimants = defaultdict(set)  # block -> requests claiming it in HBM
    head = {}  # session -> request index of its newest turn
    tier_of = {}  # request index -> verdict tier (absent = pending)
    exp_of = {}
    blocks_of = {}
    queue = []
    st = Counter()
    over = 0.0
    cap = hbm_cap

    def required(b):
        c = cnt[b]
        return "HBM" if c[0] else "DRAM" if c[1] else "SSD" if c[2] else None

    def hbm_pen(b):
        return max((float("inf") if j not in tier_of else exp_of.get(j, (0, 0))[0]
                    for j in claimants[b]), default=0.0)

    def to_dram(b, pen):
        if b not in dram:
            dram[b] = True
            st["write_dram"] += 1
        dpen[b] = pen
        if b in ssd:
            ssd.discard(b)
            st["drop_ssd"] += 1

    def reconcile(b, pen):
        req = required(b)
        if req == "HBM":
            if b in hbm:
                if dram.pop(b, None) is not None:
                    st["drop_dram"] += 1
                if b in ssd:
                    ssd.discard(b)
                    st["drop_ssd"] += 1
            return
        if req is None:
            if lossy:
                hbm.pop(b, None)
                dram.pop(b, None)
                ssd.discard(b)
            else:
                if b in dram:
                    dpen[b] = 0.0  # nobody needs it: first to spill to SSD
                if b in hbm:
                    if b not in dram and b not in ssd:
                        ssd.add(b)
                        st["write_ssd"] += 1
                    del hbm[b]
            return
        if b not in hbm:
            return
        if req == "DRAM":
            to_dram(b, pen)
        else:
            if b not in ssd:
                ssd.add(b)
                st["write_ssd"] += 1
            if dram.pop(b, None) is not None:
                st["drop_dram"] += 1
        del hbm[b]

    def claim(i, sign):
        t = tier_of.get(i, "P")
        if t is None:
            return
        slot = 0 if t in ("P", "HBM") else IDX[t]
        for b in blocks_of[i]:
            cnt[b][slot] += sign
            if slot == 0:
                (claimants[b].add if sign > 0 else claimants[b].discard)(i)

    for i, r in enumerate(rows):
        n_in = r["input_length"]
        full = r["hash_ids"][: n_in // tier_sim.BLOCK]
        kh = 0
        while kh < len(full) and full[kh] in hbm:
            kh += 1
        if kh * tier_sim.BLOCK >= n_in:
            kh -= 1
        hit = kh
        while hit < len(full) and (full[hit] in dram or full[hit] in ssd):
            hit += 1
        nd = sum(1 for b in full[kh:hit] if b in dram)
        ns = hit - kh - nd
        for b in full[kh:hit]:
            if b in dram:
                dram.move_to_end(b)
        st["tok_HBM"] += kh * tier_sim.BLOCK
        st["tok_DRAM"] += nd * tier_sim.BLOCK
        st["tok_SSD"] += ns * tier_sim.BLOCK
        m = meas.get(r["src_line"])
        if m and m[1] == n_in and m[0] > 0:
            req_tier, t = m[2], m[3]
            if m[0] == hit * tier_sim.BLOCK:
                est = estimate(k, t, kh, nd * tier_sim.BLOCK, ns * tier_sim.BLOCK)
                for tier, nb in (("HBM", kh), ("DRAM", nd), ("SSD", ns)):
                    st[(req_tier, tier)] += nb
            else:
                # Prefix cut short by a lost block: more of it is recomputed.
                st["short"] += 1
                th = float(times(max(hit, 1) * tier_sim.BLOCK,
                                 n_in - max(hit, 1) * tier_sim.BLOCK, k)[0])
                est = max(th, estimate(k, t, kh, nd * tier_sim.BLOCK, ns * tier_sim.BLOCK))
            over += est - t["t1"]
            st["e2e_ms"] += est
            st["scored"] += 1
        # The prompt is resident in HBM after prefill; this turn supersedes the
        # previous one of its session.
        for b in full:
            hbm[b] = True
            hbm.move_to_end(b)
        s = r["session_id"]
        if s in head:
            claim(head[s], -1)
        head[s] = i
        blocks_of[i] = full
        claim(i, +1)
        cur = set(full)
        while len(hbm) > cap:
            if lossy:
                b, _ = hbm.popitem(last=False)
                if b not in dram and b not in ssd and required(b) is not None:
                    st["lost_hbm"] += 1
                continue
            # A copy below makes the eviction free; else spill the block whose
            # next turn exposes least from DRAM.
            free = next((b for b in hbm if b not in cur and (b in dram or b in ssd)), None)
            if free is not None:
                del hbm[free]
                continue
            cands = [b for b in hbm if b not in cur]
            if not cands:
                break
            b = min(cands, key=hbm_pen)
            heads = claimants[b]
            to_dram(b, max((float("inf") if j not in tier_of else exp_of.get(j, (0, 0))[1]
                            for j in heads), default=0.0))
            del hbm[b]
            st["spill_hbm"] += 1
        queue.append(i)
        while queue and queue[0] <= i - lag:
            j = queue.pop(0)
            later = fut[rows[j]["src_line"]]
            tier = later[0] if later else None
            if head[rows[j]["session_id"]] != j:
                st["superseded"] += 1
                continue
            claim(j, -1)
            tier_of[j] = tier
            exp_of[j] = nexp.get(rows[j]["src_line"], (0.0, 0.0))
            claim(j, +1)
            for b in blocks_of[j]:
                reconcile(b, exp_of[j][1])
        st["peak_hbm_live"] = max(st["peak_hbm_live"], len(hbm))
        st["peak_dram"] = max(st["peak_dram"], len(dram))
        if lossy:
            if len(dram) > 0.95 * dram_cap:
                st["dram_evict_rounds"] += 1
                for _ in range(int(0.1 * len(dram))):
                    b, _ = dram.popitem(last=False)
                    if required(b) is not None:
                        st["lost_dram"] += 1
        elif len(dram) > 0.9 * dram_cap:
            st["dram_evict_rounds"] += 1
            for b in sorted(dram, key=lambda x: dpen.get(x, 0.0))[: int(0.1 * len(dram))]:
                del dram[b]
                ssd.add(b)
                st["spill_dram"] += 1
    return st, over


def estimated(rows):
    """src_line -> (prefix, input_length, tier, {t1, t2, t4}) from gen_trace's
    Estimator (nearest SWE request-safe measurements), for traces never measured."""
    sys.path.insert(0, str(FIG3.parents[1] / "datasets/req-priority-trace"))
    import gen_trace
    est, seen, out = gen_trace.Estimator(), set(), {}
    for r in rows:
        n_in = r["input_length"]
        full = r["hash_ids"][: n_in // tier_sim.BLOCK]
        hit = 0
        while hit < len(full) and full[hit] in seen:
            hit += 1
        seen.update(full)
        p = min(hit, (n_in - 1) // tier_sim.BLOCK) * tier_sim.BLOCK
        if p:
            tier, t1, t2, t4 = est(p, n_in - p)
            out[r["src_line"]] = (p, n_in, tier, {"t1": t1, "t2": t2, "t4": t4})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm", type=int, default=tier_sim.HBM_BLOCKS)
    ap.add_argument("--dram", type=int, default=tier_sim.DRAM_BLOCKS)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--raw", default="raw/swe_reqprio")
    ap.add_argument("--tiers", choices=("measured", "model", "estimate"), default="measured")
    ap.add_argument("--lines", type=int, default=0, help="first N requests only")
    ap.add_argument("--baselines", default="lru,none")
    ap.add_argument("--lossy", action="store_true",
                    help="the first version: drop on eviction and on no next turn")
    a = ap.parse_args()
    k = json.loads((FIG3 / "fit_times_swe.json").read_text())
    rows = [json.loads(x) for x in open(a.trace)]
    if a.lines:
        rows = rows[: a.lines]
    meas = (measured(FIG3 / a.raw) if a.tiers == "measured"
            else modeled(rows, k) if a.tiers == "model" else estimated(rows))
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session_id"]].append(r)
    fut, nexp = {}, {}
    for turns in by_sess.values():
        turns.sort(key=lambda r: r["round_index"])
        tiers = [meas[r["src_line"]][2] if r["src_line"] in meas else None for r in turns]
        for j, r in enumerate(turns):
            fut[r["src_line"]] = tiers[j + 1:]
            nm = meas.get(turns[j + 1]["src_line"]) if j + 1 < len(turns) else None
            if nm:
                t = nm[3]
                nexp[r["src_line"]] = (max(0.0, t["t2"] - t["t1"]), max(0.0, t["t4"] - t["t2"]))
    print(f"{a.trace.name}: HBM {a.hbm} blocks, DRAM {a.dram} blocks, verdict lag {a.lag}")
    for v in a.baselines.split(","):
        opts = set(v.split("+")) - {"none"}
        mis, _, _, over, comp = run(rows, meas, fut, k, opts, a.hbm, a.dram, a.lag)
        tok = max(comp["tok_prefix"], 1)
        total = sum(mis.values()) or 1
        rank = {t: j for j, t in enumerate(TIERS)}
        slow = sum(c for (rq, t), c in mis.items() if rank[t] > rank[rq])
        print(f"{v:8s} e2e {comp['e2e_ms'] / max(comp['scored'], 1):6.1f} ms  exposed "
              f"{over / 1e3:6.1f} s  from HBM/DRAM/SSD " + "/".join(
                  f"{100 * comp[f'tok_{t}'] / tok:.1f}" for t in TIERS)
              + f"%  slower than own tier {100 * slow / total:.1f}%")
    st, over = run_excl(rows, meas, fut, nexp, k, a.hbm, a.dram, a.lag, lossy=a.lossy)
    tok = max(st["tok_HBM"] + st["tok_DRAM"] + st["tok_SSD"], 1)
    total = sum(v for kk, v in st.items() if isinstance(kk, tuple)) or 1
    rank = {t: j for j, t in enumerate(TIERS)}
    slow = sum(v for kk, v in st.items() if isinstance(kk, tuple) and rank[kk[1]] > rank[kk[0]])
    print(f"{'excl':8s} e2e {st['e2e_ms'] / max(st['scored'], 1):6.1f} ms  exposed "
          f"{over / 1e3:6.1f} s  from HBM/DRAM/SSD " + "/".join(
              f"{100 * st[f'tok_{t}'] / tok:.1f}" for t in TIERS)
          + f"%  slower than own tier {100 * slow / total:.1f}%")
    print(f"  written DRAM/SSD {st['write_dram']}/{st['write_ssd']} blocks, dropped "
          f"DRAM/SSD {st['drop_dram']}/{st['drop_ssd']}; spilled HBM->DRAM {st['spill_hbm']} "
          f"DRAM->SSD {st['spill_dram']}; lost HBM {st['lost_hbm']} DRAM {st['lost_dram']} "
          f"({st['dram_evict_rounds']} rounds); peak HBM {st['peak_hbm_live']} "
          f"DRAM {st['peak_dram']}; prefix cut {st['short']}; superseded verdicts {st['superseded']}")


if __name__ == "__main__":
    main()

"""Block-level simulation of the KVPilot placements on a replayed trace, to
predict their prefill time without the GPU and pick traces that separate them.

    python3 sim_placement.py --trace swe_blockprio.jsonl \
        --table ../../utils/placement_ab/signal_swe_blockprio.json --hbm-blocks 600
    python3 sim_placement.py ... --logs lru=<dir> incp=<dir>   # compare with runs

Strategies: lru, belady, incl (inclusive-safe-reqwise), incp
(inclusive-penalty-reqwise), excl (exclusive-penalty-reqwise), exclu (its
unblocking variant), exclb (exclusive-penalty-blockwise-unblocking).

What is modeled (vllm/v1/core/kvpilot_*.py, the Mooncake master):
  * A block is its content with its prefix (the chain of trace hash ids), as
    vLLM chains block hashes. A request reads the longest run of its full
    prompt blocks from HBM, then goes on block by block in DRAM, else SSD,
    leaving at least one token to compute; its prefill time is the time model
    (kvpilot_time_model, the oracle placement's coefficients) for that split.
  * HBM holds hbm_blocks; a request takes a slot per block it does not hit.
    Inclusive placements evict free blocks by key: free order (lru), farthest
    next use (belady), next turn's tier class (incl), HBM -> DRAM penalty
    (incp); signal pending last. Exclusive placements pin a freed request's
    blocks until its signal, keep only blocks wanted in HBM (held), demote held
    ones to DRAM when a request is short (excl: the request waits; exclu /
    exclb: lent for its loaded prefix when that covers the shortfall, and the
    reserve refilled after each request).
  * DRAM holds dram_blocks (WATERMARK); above it the master removes
    (inclusive; SSD keeps a copy) or spills to SSD (exclusive; a block that
    kept its SSD copy is only removed) a tenth of it, lowest (class, penalty,
    release order, -position) first, so a request leaves from its tail.
    Inclusive writes every full prompt block to DRAM and SSD (write-through,
    SSD hits loaded through). Exclusive keeps one copy in HBM or DRAM; an SSD
    copy stays when a block moves up, and a turn with no next turn writes
    nothing.
  * A clock follows the replay (-p 1, released at the first token): a request
    starts when the previous one's first token is out (send, any wait for HBM,
    prefill, decode-side KV pull, one decode step) and not before its
    session's previous turn decoded. Its signal arrives once its decode ends
    (output tokens x --tpot); tier and penalties come from the suffix the
    table gives and the time model (kvpilot_blockwise). An exclusive request
    short of HBM with nothing to demote or lend waits for the next signal.

Exclusive writes below go through one serial writer, a job taking --job-ms
plus per block written to DRAM / SSD (fitted on the sweblock run's job lines);
its blocks leave HBM when it ends. A lend's copy rides on the load
(--lend-ms...). Not modeled: the master's offload rate and in-flight cap,
chunked prefill (one chunk), queueing on the prefill side.
"""
from __future__ import annotations

import argparse
import heapq
import importlib.util
import itertools
import json
import re
import statistics
import sys
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / "vllm-dev/vllm/v1/core"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ("vllm", "vllm.v1", "vllm.v1.core"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
tm = _load("vllm.v1.core.kvpilot_time_model", CORE / "kvpilot_time_model.py")
bw = _load("vllm.v1.core.kvpilot_blockwise", CORE / "kvpilot_blockwise.py")

# Tokens per trace hash id.
BLOCK = 512
# DRAM fill at which the master starts removing, as configured. In the
# Eval-EXP1 round-2 runs it evicted at used_ratio 0.81-0.84 (median), yet LRU's
# DRAM hits match these runs better at 0.9 (the master's order is not modeled).
WATERMARK = 0.9
COEFFS = ROOT / "paper-figure/Figure3-request-safe/fit_times2_swe.json"
# Per model, 1P1D at TP 2 unless named (default service settings, GPU_MEM_UTIL 0.9):
#   block      vLLM block = LMCache chunk, tokens
#   hbm        prefill KV pool, blocks
#   block_gib  one block on one TP rank in Mooncake DRAM, GiB
#   kv_layers  layers whose KV goes below HBM
#   hybrid     linear-attention states (Qwen3.6, Qwen3.8): never offloaded,
#              `states` blocks each in HBM (one per state group); a hit ends at
#              one (vllm/v1/core/kvpilot_states.py), and the prefill side
#              computes the prompt but its last token
#   excl_save  exclusive prefill below the time model (fitted under inclusive's
#              write-through): ms, ms per 1K tokens computed, per 1K prefix;
#              measured minus model of the Eval-EXP1 SWE runs, exclusive less
#              inclusive arms
#   job        exclusive writer job: ms, ms per DRAM block, per SSD block
PROFILES = {
    "qwen3-30b": {"block": 512, "coeffs": COEFFS, "hbm": 2361, "block_gib": 24 / 1024,
                  "kv_layers": 48, "hybrid": False, "excl_save": (0.0, 1.6, 0.0),
                  "job": (0.0, 1.6, 21.4)},
    "qwen3.6": {"block": 1536, "hbm": 3485, "block_gib": 15 / 1024, "kv_layers": 10,
                "hybrid": True, "states": 3,
                "coeffs": ROOT / "paper-figure/Figure3-request-safe/fit_times2_qwen3.6_k160.json"},
    # 1P1D at TP 4: the 2 KV heads are replicated, a rank's KV per token stays.
    "qwen3.6-tp4": {"block": 1024, "hbm": 6912, "block_gib": 10 / 1024, "kv_layers": 10,
                    "hybrid": True, "states": 3,
                    "coeffs": ROOT / "paper-figure/Figure3-request-safe/fit_times2_qwen3.6-tp4_k160.json"},
    # Qwen3.6 FP8 weights at TP 2: KV (BF16) and states as the BF16 TP 2 profile.
    # pre: send -> prefill admission as default_cost's, fitted on the
    # Eval-EXP1 round-2 SWE runs (lru / belady / exclb alike).
    "qwen3.6-fp8": {"block": 1536, "hbm": 3485, "block_gib": 15 / 1024, "kv_layers": 10,
                    "hybrid": True, "states": 3, "pre": (5.0, 0.27, 0.27, 0.27, 0.27),
                    "excl_save": (0.65, 0.83, 0.08), "job": (1.0, 0.61, 10.5),
                    "coeffs": ROOT / "paper-figure/Figure3-request-safe/fit_times2_qwen3.6-fp8_k160.json"},
    # Qwen3.8-Flash-Next at TP 4: 12 QSA layers whose values also carry the
    # indexer keys (64 KiB per block); states of 3 GDN groups and the PLE
    # short conv.
    "qwen3.8": {"block": 1024, "hbm": 3071, "block_gib": 12 * 1.0625 / 1024, "kv_layers": 12,
                "hybrid": True, "states": 4, "pre": (5.1, 0.37, 0.37, 0.37, 0.37),
                "excl_save": (0.0, 1.6, 0.0), "job": (0.0, 0.55, 13.3),
                "coeffs": ROOT / "paper-figure/Figure3-request-safe/fit_times2_qwen3.8-ep_k160.json"},
}
STRATEGIES = ("lru", "belady", "incl", "incp", "excl", "exclu", "exclb")
CLASS = {None: 1, "SSD": 2, "DRAM": 3, "HBM": 4}
RANK = {"HBM": 0, "DRAM": 1, "SSD": 2}
FAR = 1e9
PENDING = 1e8
INF = float("inf")


def load_trace(path: str, table_path: str | None, prof: dict = PROFILES["qwen3-30b"]) -> list[dict]:
    table = json.load(open(table_path)) if table_path else {}
    rows = [json.loads(line) for line in open(path)]
    return make_reqs(rows, [table.get(str(i), {}).get("next_suffix") for i in range(len(rows))],
                     prof)


def make_reqs(rows, next_suffix, prof: dict = PROFILES["qwen3-30b"]) -> list[dict]:
    """Trace rows in replay order and each one's oracle next suffix. A request's
    chains name its full blocks of computed prompt (the prefill side's; a
    hybrid model's leaves out the last token) by their prefix."""
    per, block = prof["block"] // BLOCK, prof["block"]
    reqs = []
    for i, (r, s) in enumerate(zip(rows, next_suffix)):
        ncomp = r["input_length"] - prof["hybrid"]
        chain, links = None, []
        for h in r["hash_ids"][: ncomp // BLOCK]:
            chain = hash((chain, int(h)))
            links.append(chain)
        chains = links[per - 1 :: per][: ncomp // block]
        reqs.append({"i": i, "sid": r["session_id"], "n": r["input_length"], "ncomp": ncomp,
                     "out": r["output_length"], "chains": chains, "next_suffix": s})
    last: dict[int, int] = {}
    for q in reversed(reqs):
        q["next_use"] = [last.get(c) for c in q["chains"]]
        for c in q["chains"]:
            last[c] = q["i"]
    return reqs


class Head:
    __slots__ = ("hid", "sid", "blocks", "tier", "time", "block_tiers", "layout",
                 "dram_pen", "dist")

    def __init__(self, hid, sid, blocks):
        self.hid, self.sid, self.blocks = hid, sid, blocks
        self.tier, self.time = "P", None
        self.block_tiers = self.layout = self.dram_pen = self.dist = None

    def tier_of(self, k):
        if self.tier == "P":
            return "HBM"
        return self.block_tiers[k] if self.tier == "B" else self.tier


class Sim:
    def __init__(self, strat, reqs, model, hbm_blocks, dram_blocks, reserve, cost,
                 prof: dict = PROFILES["qwen3-30b"]):
        self.s, self.reqs, self.M = strat, reqs, model
        self.B = prof["block"]
        self.hybrid = prof["hybrid"]
        self.state_blocks = prof.get("states", 0)
        # Hybrid: the state at a turn's end, by its last chain: "held" (out of
        # the free queue, with holders), "busy" (a running request's hit) or,
        # under LRU, "cached" in the free queue. Not held elsewhere means gone
        # (it sorts first in the class queue).
        self.sts: dict[int, str] = {}
        self.sholders: dict[int, int] = {}
        self.excl = strat in ("excl", "exclu", "exclb")
        self.blockwise = strat == "exclb"
        self.lend_ok = strat in ("exclu", "exclb")
        self.reserve = reserve if self.lend_ok else 0
        self.Hcap, self.Dcap, self.cost = hbm_blocks, dram_blocks, cost
        self.now = self.p_free = 0.0
        self.ready: dict[str, float] = {}
        # HBM: chain -> "cached" | "held" | "pinned" | "busy"; keys for eviction.
        self.hbm: dict[int, str] = {}
        self.hkey: dict[int, tuple] = {}
        self.hheap: list = []
        self.dram: dict[int, tuple] = {}
        self.dheap: list = []
        self.ssd: set[int] = set()
        # Blocks an exclusive job is writing to SSD -> its end (ms).
        self.ssd_pending: dict[int, float] = {}
        self.order: dict[int, tuple[int, int]] = {}
        self.seq = itertools.count()
        self.ord = itertools.count()
        self.heads: dict[int, Head] = {}
        self.claim: dict[int, set[int]] = defaultdict(set)
        self.sess: dict[str, int] = {}
        self.pending: list = []  # (arrival ms, head id)
        # Exclusive writes below, one job at a time: (end ms, seq, blocks,
        # those bound for SSD).
        self.writing: list = []
        self.writer_free = 0.0
        self.stats = defaultdict(int)

    # ---- claims -------------------------------------------------------------

    def claimants(self, c):
        return [self.heads[h] for h in self.claim.get(c, ())]

    def required(self, c):
        best = None
        for g in self.claimants(c):
            t = g.tier_of(g.blocks.index(c))
            if t is not None and (best is None or RANK[t] < RANK[best]):
                best = t
        return best

    def pending_claim(self, c):
        return any(g.tier == "P" for g in self.claimants(c))

    def hbm_pen(self, c):
        pens = [max(0.0, g.time[1] - g.time[0]) for g in self.claimants(c)
                if g.tier not in ("P", None, "B")]
        return max(pens, default=0.0)

    def dram_pen(self, c):
        pens = []
        for g in self.claimants(c):
            if g.tier == "B":
                v = g.dram_pen[g.blocks.index(c)]
                pens.append(v or 0.0)
            elif g.tier not in ("P", None):
                pens.append(max(0.0, g.time[2] - g.time[1]))
        return max(pens, default=0.0)

    def track(self, q, full, arrival):
        # The session's next turn supersedes its head once signalled
        # (kvpilot_claims); a head still waiting keeps its claims until then.
        old = self.sess.get(q["sid"])
        if old is not None and self.heads[old].tier != "P":
            self.drop_head(old)
        h = Head(q["i"], q["sid"], full)
        self.heads[h.hid] = h
        self.sess[q["sid"]] = h.hid
        for c in full:
            self.claim[c].add(h.hid)
        heapq.heappush(self.pending, (arrival, h.hid))
        return h

    def drop_head(self, hid):
        h = self.heads.pop(hid, None)
        if h is None:
            return
        for c in h.blocks:
            s = self.claim.get(c)
            if s:
                s.discard(hid)
                if not s:
                    del self.claim[c]
        if self.sess.get(h.sid) == hid:
            del self.sess[h.sid]

    # ---- HBM ----------------------------------------------------------------

    def hset(self, c, state, key=None):
        self.hbm[c] = state
        if key is not None:
            self.hkey[c] = key
            heapq.heappush(self.hheap, (key, c))

    def hdrop(self, c):
        self.hbm.pop(c, None)
        self.hkey.pop(c, None)

    def hfree(self):
        return self.Hcap - len(self.hbm) - self.state_blocks * len(self.sts)

    # ---- hybrid states -------------------------------------------------------

    def state_hold(self, c):
        """Hold the state ending chain c (one more holder)."""
        if self.sts.get(c) == "held":
            self.sholders[c] += 1
            return
        if c not in self.sts and self.hfree() < self.state_blocks:
            if self.excl:
                self.stats["overcommit"] += 1
            else:
                self.evict_cached(self.state_blocks)
        self.sts[c] = "held"
        self.sholders[c] = 1

    def state_let_go(self, c):
        """The state ending chain c is not held (any more): LRU queues it, the
        others drop it (first in their queue)."""
        if self.sts.get(c) == "held":
            self.sholders[c] -= 1
            if self.sholders[c]:
                return
            del self.sholders[c]
        if self.s == "lru":
            if c not in self.sts and self.hfree() < self.state_blocks:
                self.evict_cached(self.state_blocks)
            self.sts[c] = "cached"
            key = (0.0, next(self.seq))
            self.hkey[("S", c)] = key
            heapq.heappush(self.hheap, (key, ("S", c)))
        else:
            self.sts.pop(c, None)

    def evict_cached(self, n):
        out = 0
        while out < n and self.hheap:
            key, c = heapq.heappop(self.hheap)
            if isinstance(c, tuple):
                # An LRU-queued state (keys carry unique seqs: no ties).
                if self.sts.get(c[1]) == "cached" and self.hkey.get(c) == key:
                    del self.sts[c[1]]
                    del self.hkey[c]
                    out += self.state_blocks
                continue
            if self.hbm.get(c) != "cached" or self.hkey.get(c) != key:
                continue
            self.hdrop(c)
            out += 1
        return out

    def held_order(self, own):
        """Held blocks cheapest to give up first (excl / exclu)."""
        hs = [(self.hkey[c], c) for c, st in self.hbm.items() if st == "held" and c not in own]
        hs.sort()
        return [c for _, c in hs]

    def pick(self, n, own):
        """Up to n held blocks to give up, cheapest to expose first (exclb:
        each head's last HBM block, its layout and DRAM penalties updated)."""
        if not self.blockwise:
            return self.held_order(own)[:n]
        out, cands = [], []
        for g in self.heads.values():
            if g.tier == "B" and g.layout[0] > 0:
                cands.append((self.tail_key(g), next(self.seq), g.hid))
        heapq.heapify(cands)
        while len(out) < n and cands:
            _, _, hid = heapq.heappop(cands)
            g = self.heads.get(hid)
            if g is None or g.tier != "B" or g.layout[0] == 0:
                continue
            c = g.blocks[g.layout[0] - 1]
            if c in own or self.hbm.get(c) != "held":
                continue
            if any(o.tier == "B" and o is not g and o.block_tiers[o.blocks.index(c)] == "HBM"
                   and o.layout[0] - 1 != o.blocks.index(c) for o in self.claimants(c)):
                continue
            for o in self.claimants(c):
                if o.tier == "B" and o.block_tiers[o.blocks.index(c)] == "HBM":
                    o.block_tiers[o.blocks.index(c)] = "DRAM"
                    o.layout = [o.layout[0] - 1, o.layout[1] + 1, o.layout[2]]
                    self.set_dram_pen(o)
            out.append(c)
            if g.layout[0] > 0:
                heapq.heappush(cands, (self.tail_key(g), next(self.seq), g.hid))
        return out

    def write_job(self, n_dram, n_ssd, blocks, at, to_ssd=()):
        """The serial writer writes blocks below; they leave HBM when done,
        and those bound for SSD are on SSD only then."""
        c = self.cost
        start = max(at, self.writer_free)
        end = (start + c["job_ms"] + c["job_ms_per_dram"] * n_dram
               + c["job_ms_per_ssd"] * n_ssd)
        self.writer_free = end
        for b in blocks:
            self.hbm[b] = "writing"
            self.hkey.pop(b, None)
        heapq.heappush(self.writing, (end, next(self.seq), blocks, to_ssd))
        for b in to_ssd:
            self.ssd_pending[b] = end
        return end

    def settle(self, t):
        while self.writing and self.writing[0][0] <= t:
            _, _, blocks, to_ssd = heapq.heappop(self.writing)
            self.ssd.update(to_ssd)
            for b in to_ssd:
                self.ssd_pending.pop(b, None)
            for b in blocks:
                if self.hbm.get(b) == "writing":
                    self.hdrop(b)

    def n_writing(self):
        return sum(1 for st in self.hbm.values() if st == "writing")

    def demote(self, n, own, at):
        """Write n held blocks to DRAM; returns when they are written."""
        picked = self.pick(n, own)
        for c in picked:
            self.dput(c, self.dram_pen(c))
        self.denforce()
        return self.write_job(len(picked), 0, picked, at) if picked else at, len(picked)

    def tail_key(self, g):
        return bw.hbm_penalty(self.M, len(g.blocks), g.dist, g.layout[1], g.layout[2])

    def set_dram_pen(self, g):
        h, d, x = g.layout
        pens = bw.dram_penalties(self.M, len(g.blocks), g.dist, d, x)
        g.dram_pen = [None] * h + pens + [None] * x
        for k in range(h, h + d):
            c = g.blocks[k]
            if c in self.dram:
                self.dput(c, self.dram_pen(c))

    # ---- DRAM ---------------------------------------------------------------

    def dkey(self, c, rank):
        ordv, pos = self.order.get(c, (1 << 40, 0))
        return (rank, ordv, -pos)

    def dput(self, c, rank):
        key = self.dkey(c, rank)
        self.dram[c] = key
        heapq.heappush(self.dheap, (key, c))

    def drank(self, c):
        """The master's rank of a DRAM block for the inclusive placements."""
        s = self.s
        if s == "lru":
            return 0.0
        if s == "belady":
            return FAR - self.order_next[c] if self.order_next.get(c) is not None else 0.0
        if s == "incl":
            return CLASS[self.required(c)] if c in self.claim else CLASS[None]
        if s == "incp":
            if self.pending_claim(c):
                return INF
            return self.dram_pen(c)
        raise AssertionError(s)

    def denforce(self):
        over = len(self.dram) - self.Dcap
        if over <= 0:
            return
        # The master drops (inclusive) or spills / drops (exclusive) a tenth
        # of DRAM at once.
        target = over + len(self.dram) // 10
        out = 0
        while out < target and self.dheap:
            key, c = heapq.heappop(self.dheap)
            if self.dram.get(c) != key:
                continue
            del self.dram[c]
            if self.excl:
                self.stats["dropped_on_ssd" if c in self.ssd else "spilled"] += 1
                self.ssd.add(c)
            out += 1

    # ---- signals ------------------------------------------------------------

    def deliver(self, upto):
        while self.pending and self.pending[0][0] <= upto:
            at, hid = heapq.heappop(self.pending)
            self.signal(hid, at)

    def signal(self, hid, at):
        g = self.heads.get(hid)
        if g is None or g.tier != "P":
            return
        q = self.reqs[hid]
        s = q["next_suffix"]
        if s is None or self.s in ("lru", "belady"):
            tier, time = None, None
        else:
            tier, time = bw.reqwise(self.M, len(g.blocks), [[1.0, s]])
        if self.blockwise and tier is not None:
            h, d, x = bw.place(self.M, len(g.blocks), [[1.0, s]])
            g.tier, g.time = "B", time
            g.block_tiers = ["HBM"] * h + ["DRAM"] * d + ["SSD"] * x
            g.layout, g.dist = [h, d, x], [[1.0, s]]
            g.dram_pen = [None] * len(g.blocks)
            self.set_dram_pen(g)
        else:
            g.tier, g.time = tier, time
        self.stats[f"signal_{g.tier}"] += 1
        if self.excl:
            self.reconcile(g, at)
        else:
            for k, c in enumerate(g.blocks):
                if self.hbm.get(c) == "cached":
                    self.rekey_hbm(c)
                if c in self.dram:
                    self.dput(c, self.drank(c))
        if tier is None:
            self.drop_head(hid)
            if self.hybrid and g.blocks:
                self.state_let_go(g.blocks[-1])

    def rekey_hbm(self, c):
        _, seq = self.hkey[c]
        if self.s == "incl":
            r = CLASS[self.required(c)] if c in self.claim else CLASS[None]
        else:
            r = PENDING if self.pending_claim(c) else (self.hbm_pen(c) if c in self.claim else 0.0)
        self.hset(c, "cached", (r, seq))

    def reconcile(self, g, at):
        down_dram, down_ssd = [], []
        # Tail first, as a request frees: within a key the tail goes first.
        for c in reversed(g.blocks):
            if self.hbm.get(c) not in ("pinned", "held"):
                continue
            if self.pending_claim(c):
                continue
            want = self.required(c)
            if want == "HBM":
                self.dram.pop(c, None)
                key = (self.hbm_pen(c), next(self.seq)) if not self.blockwise else (0.0, next(self.seq))
                self.hset(c, "held", key)
                continue
            if want is None:
                # No next turn: nothing written; a DRAM copy demotes first.
                if c in self.dram:
                    self.dput(c, 0.0)
                self.hdrop(c)
                continue
            if (want == "DRAM" and c in self.dram) or (want == "SSD" and c in self.ssd):
                if want == "SSD":
                    self.dram.pop(c, None)
                self.hdrop(c)
                continue
            if want == "DRAM":
                self.dput(c, self.dram_pen(c))
                down_dram.append(c)
            else:
                self.dram.pop(c, None)
                down_ssd.append(c)
        if down_dram or down_ssd:
            self.stats["write_dram"] += len(down_dram)
            self.stats["write_ssd"] += len(down_ssd)
            self.write_job(len(down_dram), len(down_ssd), down_dram + down_ssd, at, down_ssd)
        self.denforce()

    # ---- a request ----------------------------------------------------------

    def step(self, q):
        j = q["i"]
        cost = self.cost
        chains, n = q["chains"], q["n"]
        # Sent at the previous request's first token, or when its session's
        # previous round ends; the prefill side admits it after the frontend
        # and lookup (pre) once the previous prefill is done (queue).
        sent = max(self.now, self.ready.get(q["sid"], 0.0))
        # The signal of the session's previous round (its end + signal_ms)
        # reaches the prefill side before this request's lookup does.
        self.deliver(sent + cost["signal_ms"])
        self.settle(sent)
        B = self.B
        F = len(chains)
        cap = min(F, (q["ncomp"] - 1) // B)
        H = 0
        while H < cap and chains[H] in self.hbm:
            H += 1
        P = H
        # A block still being written to SSD is not there yet: the prefill
        # waits for the write (lmcache "KVPilot hybrid / exclusive wait").
        while P < cap and (chains[P] in self.dram or chains[P] in self.ssd
                           or chains[P] in self.ssd_pending):
            P += 1
        if self.hybrid:
            # The hit ends at a state in HBM; HBM's KV past it is no use.
            while P and chains[P - 1] not in self.sts:
                P -= 1
            H = min(H, P)
            if P and self.sts[chains[P - 1]] == "cached":
                self.sts[chains[P - 1]] = "busy"
        wait_until = max((self.ssd_pending[c] for c in chains[H:P]
                          if c in self.ssd_pending and c not in self.dram), default=0.0)
        Dn = sum(1 for c in chains[H:P] if c in self.dram)
        Xn = P - H - Dn
        a, b_new, b_hbm, b_dram, b_ssd = cost["pre"]["excl" if self.excl else "incl"]
        pre = a + (b_new * (n - P * B) + (b_hbm * H + b_dram * Dn + b_ssd * Xn) * B) / 1000
        start = max(sent + pre, self.p_free, wait_until)
        queue = start - sent - pre
        self.deliver(start)
        self.settle(start)
        own = set(chains[:H])
        for c in chains[:H]:
            if self.hbm.get(c) == "cached":
                self.hbm[c] = "busy"
        # Plus the running state (hybrid).
        need = -(-q["ncomp"] // B) - H + self.state_blocks
        blocked = extra = 0.0
        short = need - self.hfree()
        if short > 0:
            if not self.excl:
                self.evict_cached(short)
            else:
                picked = []
                if self.lend_ok and short <= P - H:
                    picked = self.pick(short, own)
                if len(picked) == short:
                    # Lent to the loaded prefix: written within the load.
                    for c in picked:
                        self.hdrop(c)
                        self.dput(c, self.dram_pen(c))
                    self.denforce()
                    extra += cost["lend_ms"] + cost["lend_ms_per_block"] * short
                    self.stats["lend"] += 1
                    self.stats["lent"] += short
                    picked = None
                elif picked:
                    # Too few to lend (exclb takes only each head's last HBM
                    # block): written below like a demote.
                    for c in picked:
                        self.dput(c, self.dram_pen(c))
                    self.write_job(len(picked), 0, picked, start)
                    self.denforce()
            if self.excl and picked is not None and self.hfree() < need:
                # Waits: demote what is short (not counting writes in flight)
                # and wait for writes and signals until it fits.
                t = start
                self.stats["blocked"] += 1
                while self.hfree() < need:
                    more = need - self.hfree() - self.n_writing()
                    if more > 0:
                        end, moved = self.demote(more, own, t)
                        self.stats["demoted"] += moved
                    nxt = min(self.writing[0][0] if self.writing else INF,
                              self.pending[0][0] if self.pending else INF)
                    if nxt == INF:
                        self.stats["overcommit"] += 1
                        break
                    t = max(t, nxt)
                    self.deliver(t)
                    self.settle(t)
                blocked = t - start
        p = P * B
        e2e = self.M.e2e(p, n - p, Dn * B, Xn * B) + extra
        if self.excl:
            a, b_new, b_pre = cost["excl_save"]
            e2e -= a + (b_new * (n - p) + b_pre * p) / 1000
        done = start + blocked + e2e
        self.p_free = done
        first = done + cost["pull_ms"] + cost["pull_ms_per_ktok"] * n / 1000 + cost["tpot"]
        end = first + max(0, q["out"] - 1) * cost["tpot"]
        self.now = first + cost["dispatch_ms"]
        self.ready[q["sid"]] = end
        self.release(q, F, end + cost["signal_ms"])
        if self.reserve:
            short = self.reserve - self.hfree() - self.n_writing()
            if short > 0:
                _, moved = self.demote(short, set(), first)
                self.stats["reserve_demote"] += moved
        return {"i": j, "hbm": H, "dram": Dn, "ssd": Xn, "p": p, "s": n - p,
                "e2e": e2e, "blocked": blocked, "queue": queue, "ttft": done - sent,
                "sent": sent}

    def release(self, q, F, arrival):
        full = q["chains"][:F]
        ordv = next(self.ord)
        for k, c in enumerate(full):
            self.order[c] = (ordv, k)
        if self.s == "belady":
            for k, c in enumerate(full):
                self.order_next[c] = q["next_use"][k]
        if not self.excl:
            if self.s in ("incl", "incp"):
                self.track(q, full, arrival)
            for k in range(F - 1, -1, -1):
                c = full[k]
                seq = next(self.seq)
                if self.s == "lru":
                    r = 0.0
                elif self.s == "belady":
                    u = q["next_use"][k]
                    r = -FAR if u is None else -float(u)
                elif self.s == "incl":
                    r = CLASS["HBM"]
                else:
                    r = PENDING
                if c not in self.hbm and self.hfree() <= 0:
                    self.evict_cached(1)
                self.hset(c, "cached", (r, seq))
                self.dput(c, self.drank(c))
                self.ssd.add(c)
            self.denforce()
            self.release_states(q, full)
            return
        self.track(q, full, arrival)
        for c in full:
            if c not in self.hbm and self.hfree() <= 0:
                self.stats["overcommit"] += 1
            self.hset(c, "pinned")
            self.hkey.pop(c, None)
            if c in self.dram:
                self.dput(c, self.dram[c][0])
        self.release_states(q, full)

    def release_states(self, q, full):
        """Hybrid: the request supersedes the states inside its prompt and
        leaves its own at its end, held while a next turn comes
        (kvpilot_states): LRU and Belady while the trace reuses the prompt
        (run_ab.sh gives LRU the trace), the others until their signal."""
        if not self.hybrid or not full:
            return
        for c in full[:-1]:
            if c in self.sts:
                self.state_let_go(c)
        end = full[-1]
        if self.s in ("lru", "belady") and q["next_use"][len(full) - 1] is None:
            self.state_let_go(end)
        else:
            self.state_hold(end)

    def run(self):
        if self.s == "belady":
            self.order_next = {}
        return [self.step(q) for q in self.reqs]


# ---- measured runs ----------------------------------------------------------

OK = re.compile(r"^\[ok \] idx=(\d+) .*? sent=([\d.]+)")
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ .*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms"
                 r".*? done=([\d.]+)(?: blocked=([\d.]+)ms)?")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ .*?ssd_tokens=(\d+)")


def measured(d: str) -> dict[int, dict]:
    d = Path(d)
    sent = {int(m[1]): float(m[2]) for l in open(d / "benchmark.log") if (m := OK.match(l))}
    out, ssd = {}, defaultdict(int)
    for line in open(d / "prefill-0.log", errors="replace"):
        if m := PRE.search(line):
            i = int(m[1])
            out[i] = {"hit": int(m[2]), "dram": int(m[3]), "e2e": float(m[4]),
                      "ttft": (float(m[5]) - sent[i]) * 1e3 if i in sent else None,
                      "blocked": float(m[6] or 0)}
        elif m := RET.search(line):
            ssd[int(m[1])] = max(ssd[int(m[1])], int(m[2]))
    for i, r in out.items():
        r["ssd"] = min(ssd.get(i, 0), r["dram"])
        r["dram"] -= r["ssd"]
    return out


def summary(name, recs, skip):
    rs = [r for r in recs if r["i"] >= skip]
    tot = [sum(r[k] for r in rs) for k in ("hbm", "dram", "ssd")]
    allt = sum(tot) or 1
    ttft = sorted(r["ttft"] for r in rs)
    return (f"{name:7s} n={len(rs):5d}  TTFT mean {statistics.mean(ttft):6.1f} p50 "
            f"{ttft[len(ttft) // 2]:6.1f} p90 {ttft[int(len(ttft) * 0.9)]:6.1f}  e2e "
            f"{statistics.mean(r['e2e'] for r in rs):6.1f}  blocked "
            f"{sum(r['blocked'] for r in rs) / 1e3:5.1f} s  queue "
            f"{sum(r.get('queue', 0) for r in rs) / 1e3:4.1f} s  prefix HBM/DRAM/SSD "
            + "/".join(f"{t / allt * 100:4.1f}" for t in tot) + "%")


def default_cost(tpot, prof: dict = PROFILES["qwen3-30b"]):
    """Request timing outside the prefill, fitted on the 2026-11-22 sweblock
    (HBM 600: lru / incp / exclu) and sweblock160 (HBM 2361: all five) runs
    of Qwen3-30B-A3B; another model's decode-side pull scales with its KV
    bytes per token (Qwen3.6 not measured yet)."""
    ref = PROFILES["qwen3-30b"]
    kv_ratio = (prof["block_gib"] / prof["block"]) / (ref["block_gib"] / ref["block"])
    return {
        # Client send -> prefill admission (frontend, lookup): ms, then ms per
        # 1K tokens computed / hit in HBM / DRAM / SSD. Inclusive hits cost
        # ~0.4 ms per 1K tokens more in HBM and DRAM, at both HBM sizes. A
        # profile's own "pre" is the same for both.
        "pre": ({"incl": prof["pre"], "excl": prof["pre"]} if "pre" in prof else
                {"incl": (6.18, 0.975, 1.264, 1.43, 1.114),
                 "excl": (6.75, 0.927, 0.862, 0.987, 1.103)}),
        # Prefill done -> client first token: decode-side KV pull, one step.
        "pull_ms": 52.9, "pull_ms_per_ktok": 0.97 * kv_ratio, "tpot": tpot,
        # First token -> the replay's next send.
        "dispatch_ms": 0.6, "signal_ms": 5.0,
        # Exclusive writer jobs (TP0 job lines; a profile's own "job" from its
        # Eval-EXP1 round-2 exclb run) and a lend's copy.
        **dict(zip(("job_ms", "job_ms_per_dram", "job_ms_per_ssd"), prof.get("job", (4.9, 1.0, 25.0)))),
        "lend_ms": 3.0, "lend_ms_per_block": 0.15,
        "excl_save": prof.get("excl_save", (0.0, 0.0, 0.0)),
    }


def paired(sims, skip, base="lru"):
    """Mean TTFT difference against base over lines >= skip, its sd, and the
    steady requests for a 95% interval of +-5 ms."""
    b = {r["i"]: r["ttft"] for r in sims[base] if r["i"] >= skip}
    bm = statistics.mean(b.values())
    out = {}
    for s, recs in sims.items():
        if s == base:
            continue
        d = [r["ttft"] - b[r["i"]] for r in recs if r["i"] in b]
        sd = statistics.stdev(d)
        out[s] = (statistics.mean(d), statistics.mean(d) / bm, sd, int((1.96 * sd / 5) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(HERE / "swe_blockprio.jsonl"))
    ap.add_argument("--table", help="default: utils/placement_ab/signal_<trace>.json")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--model", choices=list(PROFILES), default="qwen3-30b")
    ap.add_argument("--hbm-blocks", type=int, help="default: the model's at GPU_MEM_UTIL 0.9")
    ap.add_argument("--dram-gib", type=float, default=100.0, help="per TP rank")
    ap.add_argument("--watermark", type=float, default=WATERMARK)
    ap.add_argument("--tpot", type=float, default=14.0,
                    help="decode ms per token (sweblock runs: 12.5 lru, 17.5 incp, 14.7 exclu)")
    ap.add_argument("--reserve", type=int, default=16)
    ap.add_argument("--skip", type=int, default=1970, help="warm-up lines left out")
    ap.add_argument("--logs", nargs="*", default=[], help="strategy=<log dir> to compare")
    a = ap.parse_args()

    prof = PROFILES[a.model]
    hbm_blocks = a.hbm_blocks or prof["hbm"]
    suffix = f"_b{prof['block']}" if prof["hybrid"] else ""
    table = a.table or str(ROOT / f"utils/placement_ab/signal_{Path(a.trace).stem}{suffix}.json")
    reqs = load_trace(a.trace, table, prof)
    M = tm.TimeModel.load(str(prof["coeffs"]), prof["block"])
    dram_blocks = int(a.watermark * a.dram_gib / prof["block_gib"])
    cost = default_cost(a.tpot, prof)
    print(f"{Path(a.trace).name} on {a.model}: {len(reqs)} requests; HBM {hbm_blocks} blocks, DRAM "
          f"{dram_blocks} blocks of {prof['block']} tokens, tpot {a.tpot} ms; steady from line {a.skip}")
    sims = {}
    for s in a.strategies.split(","):
        sim = Sim(s, reqs, M, hbm_blocks, dram_blocks, a.reserve, cost, prof)
        sims[s] = sim.run()
        print("sim  " + summary(s, sims[s], a.skip) + "  "
              + " ".join(f"{k}={v}" for k, v in sorted(sim.stats.items())
                         if not k.startswith("signal")))
    if "lru" in sims:
        print(f"\npaired TTFT vs lru (lines >= {a.skip}): mean diff, sd, steady requests for +-5 ms (95%)")
        for s, (m, rel, sd, n) in paired(sims, a.skip).items():
            print(f"  {s:6s} {m:+7.1f} ms ({rel * 100:+5.1f}%)  sd {sd:6.1f}  n {n}")
    real = {}
    for spec in a.logs:
        s, d = spec.split("=", 1)
        meas = measured(d)
        B = prof["block"]
        real[s] = [{"i": i, "hbm": r["hit"] // B, "dram": r["dram"] // B,
                    "ssd": r["ssd"] // B, "e2e": r["e2e"], "blocked": r["blocked"],
                    "ttft": r["ttft"]} for i, r in sorted(meas.items()) if r["ttft"] is not None]
        print("real " + summary(s, real[s], a.skip))
        if s in sims:
            sm = {r["i"]: r for r in sims[s]}
            for k in ("e2e", "ttft"):
                err = [(sm[r["i"]][k] - r[k]) / r[k] for r in real[s] if r["i"] >= a.skip and r["i"] in sm]
                print(f"     per-request {k} sim vs real: median error {statistics.median(err) * 100:+.1f}%, "
                      f"|err| p50 {statistics.median(abs(e) for e in err) * 100:.1f}%")
    if "lru" in real and len(real) > 1:
        print(f"\nreal paired TTFT vs lru (lines >= {a.skip})")
        for s, (m, rel, sd, n) in paired(real, a.skip).items():
            print(f"  {s:6s} {m:+7.1f} ms ({rel * 100:+5.1f}%)  sd {sd:6.1f}  n {n}")


if __name__ == "__main__":
    main()

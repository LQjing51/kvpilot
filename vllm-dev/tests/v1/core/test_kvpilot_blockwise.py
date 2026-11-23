# SPDX-License-Identifier: Apache-2.0
"""KVPilot blockwise: time model, placement, penalties, HBM tail picks."""

import json

import pytest

from vllm.v1.core import kvpilot_blockwise as bw
from vllm.v1.core import kvpilot_heap
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kvpilot_exclusive import ExclusivePlacement
from vllm.v1.core.kvpilot_lru import LruPlacement
from vllm.v1.core.kvpilot_time_model import TimeModel

BS = 16
# Round numbers: per prefix token 1 us from DRAM, 10 us from SSD; compute 10 ms
# plus 0.01 ms per suffix token (one chunk, no cudagraph); no first-layer or
# fixed load cost; a sharp max of load and compute.
K = {"version": 2, "chunk": 1 << 20, "capture": [], "thresh": 0.05,
     "d_graph": 0.0, "d_eager": 10.0, "b_graph": 0.0, "b_eager": 0.01,
     "a": 0.0, "e": 0.0, "c_dram": 0.001, "f_dram": 0.0, "c_ssd": 0.01,
     "f_ssd": 0.0, "g_dram": 0.0, "h_dram": 0.0, "g_ssd": 0.0, "h_ssd": 0.0,
     "k": 0.0, "F": 0.0, "omega": 0.0, "w0": 1e-9}


def _model(tmp_path, **over):
    path = tmp_path / "coeffs.json"
    path.write_text(json.dumps({**K, **over}))
    return TimeModel.load(str(path), 512)


def test_time_model_rejects_old_coefficients(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"a": 0.0, "b": 0.01, "d": 10.0, "thresh": 0.05}))
    with pytest.raises(ValueError):
        TimeModel.load(str(path), 512)


def test_time_model_chunks_and_cudagraph(tmp_path):
    m = _model(tmp_path, chunk=8192, capture=[64, 256], d_graph=5.0, b_graph=0.02,
               a=1e-6, e=1e-4)
    # 100 tokens run as a 256-token graph; attention on the real tokens.
    assert m.compute(1000, 100) == pytest.approx(
        5.0 + 0.02 * 256 + 1e-6 * 100 * (1000 + 50) + 1e-4 * 1000)
    # 10000 tokens: an 8192 chunk, then 1808 after 8192 more context.
    assert m.compute(0, 10000) == pytest.approx(
        2 * 10.0 + 0.01 * 10000 + 1e-6 * (8192 * 4096 + 1808 * (8192 + 904))
        + 1e-4 * 8192)


def test_smooth_overlap(tmp_path):
    m = _model(tmp_path, F=5.0, omega=0.1, w0=1.0)
    c = m.compute(8192, 1000)
    assert c == pytest.approx(20.0)  # so w = 3 ms
    # Far from the knee (load + F = compute, at 15000 DRAM tokens) it is the
    # max of compute and load + F; at the knee, w * ln 2 more.
    e = [m.e2e(8192, 1000, d, 0) for d in (1024, 8192, 15000, 40000)]
    assert e[0] == pytest.approx(c, abs=0.05)
    assert c < e[1] < e[2]
    assert e[2] == pytest.approx(c + 3 * 0.693147, abs=1e-3)
    assert e[3] == pytest.approx(40.0 + 5.0, abs=0.01)


@pytest.fixture
def model(tmp_path):
    return _model(tmp_path)


def test_safe_suffix_takes_a_value_95pct_reach():
    assert bw.safe_suffix([[1.0, 640]]) == 640
    assert bw.safe_suffix([[0.9, 100], [0.1, 200]]) == 100
    assert bw.safe_suffix([[0.02, 50], [0.98, 300]]) == 300
    assert bw.safe_suffix([[0.06, 50], [0.94, 300]]) == 50


def test_mixed_load_hides_dram_behind_ssd(model):
    # 4096 DRAM + 1024 SSD tokens: the SSD read (10.24 ms) hides the copies
    # of all 5120 (5.12 ms).
    assert model.e2e(8192, 1, 4096, 1024) == pytest.approx(10.24)
    assert model.e2e(8192, 1, 4096, 0) == pytest.approx(10.01)  # compute-bound


def test_place_fewest_hbm_then_most_ssd(model):
    # 20 blocks (10240 tokens), suffix 100: compute 11 ms, safe below 11.55;
    # all in DRAM loads in 10.24 ms, and two SSD blocks read in 10.24 ms too.
    h, d, x = bw.place(model, 20, [[1.0, 100]])
    assert (h, d, x) == (0, 18, 2)
    assert model.e2e(20 * 512, 100, d * 512, x * 512) < 11.55
    assert model.e2e(20 * 512, 100, (d - 1) * 512, (x + 1) * 512) >= 11.55
    # Short compute: HBM keeps the head.
    h, d, x = bw.place(model, 40, [[1.0, 1]])
    assert h > 0 and h + d + x == 40
    assert model.e2e(40 * 512, 1, (d + x + 1) * 512, 0) >= 10.01 * 1.05


def test_reqwise_tier_and_times(model):
    # 20 blocks, suffix 100: compute 11 ms; all in DRAM loads in 10.24 ms
    # (hidden), all on SSD in 102.4 ms (not).
    tier, t = bw.reqwise(model, 20, [[1.0, 100]])
    assert tier == "DRAM" and t == pytest.approx([11.0, 11.0, 102.4])
    assert bw.reqwise(model, 40, [[1.0, 1]])[0] == "HBM"
    assert bw.reqwise(model, 2, [[1.0, 2000]])[0] == "SSD"
    # The tier at the suffix 95% reach, the times over the whole distribution.
    tier, t = bw.reqwise(model, 20, [[0.9, 100], [0.1, 2000]])
    assert tier == "DRAM"
    assert t == pytest.approx([0.9 * 11 + 0.1 * 30, 0.9 * 11 + 0.1 * 30, 102.4])


def test_reqwise_exclusive_takes_tier_and_penalty_from_the_model(monkeypatch, model):
    pool, excl, conn = _make(monkeypatch, model, blockwise=False)
    _run(pool, excl, "a", [b"a%d" % i for i in range(20)])
    with pytest.raises(RuntimeError):
        excl.apply("a", "HBM", [1.0, 2.0, 3.0])  # no suffix
    out = excl.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 100]])
    assert out["tier"] == "DRAM" and out["write_dram"] == 20
    (job,) = conn.jobs
    assert job["dram_pen"] == pytest.approx([102.4 - 11.0] * 20)
    assert excl.stats["model_HBM_to_DRAM"] == 1


def test_penalties_grow_toward_the_head(model):
    dist = [[0.5, 100], [0.5, 900]]
    pens = bw.dram_penalties(model, 30, dist, 20, 0)
    assert len(pens) == 20
    assert all(a >= b - 1e-9 for a, b in zip(pens, pens[1:]))
    assert bw.hbm_penalty(model, 30, dist, 20, 0) >= 0


class Conn:
    def __init__(self):
        self.jobs, self.evicts, self.orders = [], [], []

    def kvpilot_chunk_hashes(self, request_id, token_ids, n):
        return [2000 + i for i in range(n)]

    def kvpilot_group_order(self, lmc):
        self.orders.append(list(lmc))

    def kvpilot_exclusive_submit(self, job):
        self.jobs.append(job)

    def kvpilot_exclusive_evict(self, request_id, evict):
        self.evicts.append((request_id, evict))


def _make(monkeypatch, model, blocks=32, blockwise=True):
    monkeypatch.setenv("KVPILOT_PLACEMENT", "exclusive-penalty-blockwise-unblocking")
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)
    pool = BlockPool(num_gpu_blocks=blocks, enable_caching=True, hash_block_size=BS)
    excl = ExclusivePlacement(pool, BS, reserve=2, model=model, blockwise=blockwise)
    conn = Conn()
    excl.bind(conn)
    return pool, excl, conn


def _run(pool, excl, rid, hashes, hits=()):
    """A request that hits `hits` (cached blocks) and computes `hashes`."""
    pool.touch(list(hits))
    blocks = list(hits)
    for h in hashes:
        b = pool.get_new_blocks(1)[0]
        b._block_hash = h
        pool.cached_block_hash_to_block.insert(h, b)
        blocks.append(b)
    pool.free_blocks(list(reversed(blocks)))
    excl.track(rid, [blocks], list(range(BS * len(blocks))))
    return blocks


def _done(excl, conn):
    for job in conn.jobs:
        excl.on_job_done(job["id"])
    conn.jobs.clear()


def test_signal_splits_a_turn_across_tiers(monkeypatch, model):
    pool, excl, conn = _make(monkeypatch, model)
    a = _run(pool, excl, "a", [b"a%d" % i for i in range(6)])
    assert conn.orders == [[2000 + i for i in range(6)]]
    # Layout forced by the time model: stub place() to 3 HBM, 2 DRAM, 1 SSD.
    monkeypatch.setattr(bw, "place", lambda m, n, dist: (3, 2, 1))
    out = excl.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 200]])
    assert out["layout"] == [3, 2, 1]
    (job,) = conn.jobs
    assert job["dram"] == [3, 4] and job["ssd"] == [5]
    # DRAM penalties grow toward the head (block 3 dearer than block 4).
    assert job["dram_pen"][0] >= job["dram_pen"][1]
    _done(excl, conn)
    assert all(b.kvp_held for b in a[:3])
    assert all(b.block_hash is None for b in a[3:])


def test_hbm_gives_up_a_turn_from_its_tail(monkeypatch, tmp_path):
    # Compute 1 ms plus 0.01 ms per suffix token, 2.048 ms per DRAM block.
    model = _model(tmp_path, d_eager=1.0, c_dram=0.004)
    pool, excl, conn = _make(monkeypatch, model)
    monkeypatch.setattr(bw, "place", lambda m, n, dist: (n, 0, 0))
    a = _run(pool, excl, "a", [b"a%d" % i for i in range(4)])
    b = _run(pool, excl, "b", [b"b%d" % i for i in range(4)])
    excl.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 1]])
    excl.apply("b", "HBM", [1.0, 2.0, 3.0], [[1.0, 4000]])
    conn.jobs.clear()
    # b's next turn computes long enough to hide its loads, a's does not: b
    # goes first, always its last HBM block next.
    excl.demote(3)
    (job,) = conn.jobs
    assert job["block_ids"] == [b[3].block_id, b[2].block_id, b[1].block_id]
    assert excl._claims.heads["b"].layout == [1, 3, 0]
    assert excl._claims.heads["b"].block_tiers == ["HBM", "DRAM", "DRAM", "DRAM"]
    _done(excl, conn)
    # A lend takes from the tails too, and skips the request's own hits.
    pool.get_new_blocks(pool.get_num_free_blocks())
    lent = excl.lend("r", 2, 2, [b[0]])
    assert lent == 2
    ((_, ev),) = conn.evicts
    assert ev["block_ids"] == [a[3].block_id, a[2].block_id]
    head = excl._claims.heads["a"]
    assert head.layout == [2, 2, 0]
    # a gives up one more: its DRAM blocks' penalties change with it, and
    # go to the master.
    excl.demote(1, keep=[b[0]])
    repen, job = conn.jobs
    assert head.layout == [1, 3, 0]
    assert job["block_ids"] == [a[1].block_id]
    assert repen["repen"] == [[2, head.dram_pen[2]], [3, head.dram_pen[3]]]
    assert head.dram_pen[1] >= head.dram_pen[2] >= head.dram_pen[3]


def test_shared_block_keeps_fastest_tier_and_dearest_penalty(monkeypatch, tmp_path):
    model = _model(tmp_path, d_eager=1.0, c_dram=0.004)
    pool, excl, conn = _make(monkeypatch, model)
    lmc = {"a": [3000, 3001, 3002], "c": [3000, 3101]}
    conn.kvpilot_chunk_hashes = lambda rid, toks, n: lmc[rid][:n]
    a = _run(pool, excl, "a", [b"s0", b"a1", b"a2"])
    c = _run(pool, excl, "c", [b"c1"], hits=[a[0]])
    layouts = {3: (3, 0, 0), 2: (0, 2, 0)}
    monkeypatch.setattr(bw, "place", lambda m, n, dist: layouts[n])
    # a's next turn hides its loads (its DRAM penalties are small), c's
    # does not.
    excl.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 1000]])
    excl.apply("c", "HBM", [1.0, 2.0, 3.0], [[1.0, 1]])
    (job,) = conn.jobs
    # c wants its prompt in DRAM, but a keeps the shared block in HBM.
    assert job["req"] == "c" and job["dram"] == [1]
    _done(excl, conn)
    assert a[0].block_hash == b"s0" and c[1].block_hash is None
    excl.demote(3)
    (job,) = conn.jobs
    assert job["block_ids"] == [a[2].block_id, a[1].block_id, a[0].block_id]
    ha, hc = excl._claims.heads["a"], excl._claims.heads["c"]
    assert hc.dram_pen[0] > ha.dram_pen[0]
    assert job["dram_pen"][2] == hc.dram_pen[0]


def test_lru_sends_the_release_order(monkeypatch):
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=BS)
    lru = LruPlacement(BS)
    conn = Conn()
    lru.bind(conn)
    blocks = pool.get_new_blocks(3)
    for i, b in enumerate(blocks):
        b._block_hash = b"x%d" % i
    lru.track("r", [blocks], list(range(BS * 2 + 5)))
    assert conn.orders == [[2000, 2001]]

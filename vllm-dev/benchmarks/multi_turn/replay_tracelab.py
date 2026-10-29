#!/usr/bin/env python3
"""Replay the TraceLab trace against vLLM/Dynamo as synthetic-token requests.

TraceLab records how long every prompt was, not what it said, so each round is
replayed as `prefix + append` fabricated tokens sent to `/v1/completions` (that
endpoint applies no chat template, so a token array reaches the engine as sent).

KV reuse is the point, so tokens are not random:

    token[i] = pool[H(session_id, epoch, i // block_size) % len(pool)]

Deriving each block from the session id makes round N+1 a byte-prefix of round
N — its `prefix` really hits in cache — while keeping distinct sessions from
colliding, which would inflate the hit rate. Hashing per block rather than per
position matches how vLLM compares prefixes.

Input is `tracelab_turns.jsonl` from `prepare_tracelab_trace.py`.

`--forced-output` additionally pins what decode emits, rendering the round's
tool calls as `<function=NAME>` so the online dram-safe judge can recognise
them. Without it the judge parses random tokens, finds no call, and rules every
turn "session ended" — the verdict becomes a constant. It also needs the
response-length table built from *this* trace (`KVPILOT_TOOL_RESP_LEN_PATH`);
TraceLab and SWE-smith share no labels at all.

Usage:
  python3 replay_tracelab.py \
      --url http://localhost:8000 --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --max-inflight-req 32 --max-inflight-session 32 \
      --session-arrival-rate 1.0 --time-scale 100
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import uuid
from collections import defaultdict

# Must equal the engine's --block-size. A mismatch does not fail: it silently
# lowers the hit rate, which reads like a bad placement policy.
BLOCK_SIZE = 512

DEFAULT_INPUT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "datasets", "agent-traces", "tracelab",
        "tracelab_turns.jsonl",
    )
)


DEFAULT_SCHEDULE = os.path.join(os.path.dirname(DEFAULT_INPUT),
                                "tracelab_schedule.jsonl")


# ---------------------------------------------------------------- token making


def block_token(session_id: str, epoch: int, block_index: int,
                pool_size: int) -> int:
    """Index into the token pool for one block of one session.

    blake2b, not hash(): the latter is salted per interpreter, making replays
    unreproducible. `epoch` separates content that occupied the same position
    before and after a context compaction.
    """
    h = hashlib.blake2b(
        f"{session_id}:{epoch}:{block_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % pool_size


def plan_session(turns, block_size):
    """Assign every block position the round that last wrote it.

    A pure function of position would make every longer prompt a byte-prefix of
    every shorter one, always hitting the full shared length. That is wrong where
    the agent compacted its context and the trace reports `prefix` dropping
    (294,446 -> 33,997 in one session): reuse would be overstated ~9x.

    Each record already says how much was reused, so stamping every block at or
    past `prefix // block_size` with this round's index is enough — no notion of
    a compaction event is needed. Reproduces the trace's prefix on 97.6% of
    rounds in compacting sessions, 99.1% elsewhere; the rest are rounds whose
    `prefix` exceeds the previous `input`, because Anthropic also caches model
    output that this replay never sends.
    """
    block_epoch = {}          # block index -> index of the round that wrote it
    plans = []
    for i, t in enumerate(turns):
        n_blocks = -(-(t["prefix"] + t["append"]) // block_size)
        # Floor is deliberate: a block straddling the boundary holds new
        # content, so only wholly-cached blocks keep their stamp.
        for b in range(t["prefix"] // block_size, n_blocks):
            block_epoch[b] = i
        plans.append({"block_epoch": dict(block_epoch)})
    return plans


def build_prompt(session_id, n_tokens, pool, block_size, block_epoch=None):
    """The prompt for one round, each block drawn from the round that holds it."""
    ids = []
    for start in range(0, n_tokens, block_size):
        b = start // block_size
        e = (block_epoch or {}).get(b, 0)
        tid = pool[block_token(session_id, e, b, len(pool))]
        ids.extend([tid] * min(block_size, n_tokens - start))
    return ids


def load_pool(model, token_offset, vocab_limit):
    """Token ids that survive a streaming completion unharmed.

    Dynamo's incremental detokenizer slices UTF-8 on byte offsets, so a token
    decoding to a partial character kills the SSE stream; pure ASCII cannot.
    Special ids are dropped as they confuse templating and stop conditions.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    special = set(tok.all_special_ids or [])
    special.update((tok.get_added_vocab() or {}).values())
    limit = min(vocab_limit or tok.vocab_size, tok.vocab_size)
    pool = []
    for tid in range(token_offset, limit):
        if tid in special:
            continue
        try:
            text = tok.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if text and all(0x20 <= ord(c) <= 0x7E for c in text):
            pool.append(tid)
    if not pool:
        raise RuntimeError(
            f"vocab [{token_offset}, {limit}) 里没有纯 ASCII token，无法构造 pool"
        )
    return pool, tok


# ------------------------------------------------------------- forced output


def render_tool_calls(labels):
    """This round's calls as the text the online predictor parses.

    The decode-side judge finds call types with `<function=([^>\\s]+)>`
    (kvpilot_predict.RX_INLINE_CALL), so the wire format has to match the
    SWE-smith path exactly or the regex finds nothing and every turn is judged
    "session ended". Only the name is read, so the body stays empty.
    """
    return "".join(f"<function={name}></function>" for name in labels)


def build_forced_output(labels, out_len, tok, pool):
    """The exact token sequence decode must emit for this round.

    Length comes from the trace: `out_len` is what the real model generated and
    it becomes the judge's `decode_len`, so the calls go first and the rest is
    padded from the ASCII pool.

    The call set is never trimmed to fit. Dropping a trailing call would change
    the label itself (`Read+Bash:ls+Bash:ls` becomes `Read+Bash:ls`, a different
    category), and truncating mid-token would leave `Bash:fi`, which is absent
    from the lookup table and so silently answered with the global median. Both
    corrupt the quantity being measured. On the 1.6% of rounds whose `output` is
    shorter than its own rendered calls the budget is overrun instead, making
    `decode_len` a few tokens longer than the trace — the smallest of the three
    distortions, and the only one that keeps the label exact.
    """
    if not labels:
        return [pool[0]] * max(1, out_len)
    ids = tok(render_tool_calls(labels), add_special_tokens=False)["input_ids"]
    if len(ids) >= out_len:
        return ids
    return ids + [pool[0]] * (out_len - len(ids))


# ------------------------------------------------------------------- the trace


def load_turns(path, max_prompt_tokens=0):
    """Group turns into sessions, ordered by round_index, and cut the overlong.

    An overlong round truncates its session rather than being skipped. Skipping
    would leave the next round with only its grandparent's blocks warm, and that
    shortfall reads as a placement failure rather than an artifact of filtering.
    A session left with nothing is dropped.
    """
    by_session = defaultdict(list)
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            # Physical line number, carried through so the request_id can name
            # this exact turn — the analyzer keys everything off that prefix.
            t["line_no"] = lineno
            by_session[t["session_id"]].append(t)

    n_cut_sessions = n_cut_rounds = 0
    for turns in by_session.values():
        turns.sort(key=lambda t: t["round_index"])
        if not max_prompt_tokens:
            continue
        for i, t in enumerate(turns):
            if t["prefix"] + t["append"] > max_prompt_tokens:
                n_cut_sessions += 1
                n_cut_rounds += len(turns) - i
                del turns[i:]
                break
    if n_cut_rounds:
        print(f"超过 --max-prompt-tokens 而截断: {n_cut_sessions:,} 个 session，"
              f"丢弃其尾部共 {n_cut_rounds:,} 轮")

    # A session whose very first round is already too long has nothing left.
    by_session = {k: v for k, v in by_session.items() if v}
    keys = sorted(by_session)  # sorted first: dict order must not leak in
    return [(key, by_session[key]) for key in keys]


def request_id_for(line_no):
    """A UUID whose leading 8 chars are this turn's line number in the trace.

    Dynamo rejects request ids that do not parse as a UUID, so the mapping from
    a block-pool log line back to a turn has to live inside the UUID. Decimal,
    not hex: analyze_lp_ttft_breakdown.py reads that segment with `int(.., 10)`
    and its `\\d{8}` pattern skips any id carrying a-f, so a hex encoding would
    make every turn past the 9th invisible to the analyzer. Same convention as
    mooncake_trace_replay and benchmark_serving_multi_turn — one idx, one
    request, numbered by its line in the input file.
    """
    return (
        f"{min(line_no, 99999999):08d}-"
        + str(uuid.UUID(uuid.uuid4().hex))[9:]
    )


def schedule_sessions(sessions, arrival_rate, time_scale, seed):
    """Assign every round a target offset in ms from the start of the replay.

    Two timescales, handled separately because one `time_scale` cannot serve
    both. Within a session the trace's own gaps are kept (divided by
    `time_scale`): that is the real think time between a tool result arriving
    and the next request. Across sessions the real start times are discarded
    and redrawn as a Poisson process at `arrival_rate` sessions/sec, because
    the trace spans 283 days — replayed faithfully the GPU would sit idle
    99.9% of the time, and no amount of uniform scaling fixes that without
    also flattening the intra-session gaps (p50 is 10.9s) into nothing.

    `arrival_rate <= 0` keeps the trace's own session starts, scaled like
    everything else. Rounds with no arrival timestamp get their session's
    start, i.e. they are issued as soon as their turn comes up.
    """
    rng = random.Random(seed)
    plans = []
    clock_ms = 0.0
    for _, turns in sessions:
        stamps = [t.get("arrival_ms") for t in turns]
        known = [s for s in stamps if s is not None]
        base = known[0] if known else None
        if arrival_rate > 0:
            # Exponential inter-arrivals: the gaps of a Poisson process.
            clock_ms += rng.expovariate(arrival_rate) * 1000.0
            start_ms = clock_ms
        else:
            start_ms = (base or 0) / max(time_scale, 1e-9)
        offsets = []
        for stamp in stamps:
            if stamp is None or base is None:
                offsets.append(start_ms)
            else:
                offsets.append(start_ms + (stamp - base) / max(time_scale, 1e-9))
        plans.append(offsets)
    if arrival_rate <= 0 and plans:
        # Faithful mode: shift so the earliest round lands at zero.
        first = min(o for offs in plans for o in offs)
        plans = [[o - first for o in offs] for offs in plans]
    return plans


def dump_schedule(sessions, plans, path, params):
    """Write the timetable the replay follows, and report its shape.

    Always written: the schedule is what the run replayed, and reconstructing
    it afterwards from the log is guesswork. The first line records the
    parameters that produced it, so a file can be matched to its run.
    """
    rows = []
    for (sid, turns), offsets in zip(sessions, plans):
        for t, off in zip(turns, offsets):
            rows.append(
                {
                    "line_no": t["line_no"],
                    "session_id": sid,
                    "round_index": t["round_index"],
                    "target_ms": int(off),
                    "input": t["prefix"] + t["append"],
                    "prefix": t["prefix"],
                    "append": t["append"],
                    "output": t["output"],
                    "labels": t.get("labels") or [],
                }
            )
    rows.sort(key=lambda r: r["target_ms"])
    with open(path, "w") as fh:
        fh.write(json.dumps({"params": params}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Time-weighted concurrency: what a random instant would see. The peak
    # alone would flatter the schedule, since one burst can carry it.
    spans = [(o[0], o[-1]) for o in plans if o]
    events = []
    for start, end in spans:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    cur = peak = 0
    area = 0.0
    prev = None
    for stamp, delta in events:
        if prev is not None:
            area += cur * (stamp - prev)
        cur += delta
        peak = max(peak, cur)
        prev = stamp
    total_ms = max(e for _, e in spans) - min(s for s, _ in spans) if spans else 0
    mean_conc = area / total_ms if total_ms else 0.0
    starts = sorted(s for s, _ in spans)
    gaps = [b - a for a, b in zip(starts, starts[1:])]

    print(f"-> {path}: {len(rows):,} 轮 / {len(sessions):,} session")
    print(f"   回放总时长      {total_ms / 1000:,.1f}s "
          f"（{total_ms / 60000:,.1f}min）")
    print(f"   并发 session    平均 {mean_conc:.1f}   峰值 {peak}")
    if gaps:
        gaps.sort()
        print(f"   session 起点间隔 p50={gaps[len(gaps) // 2] / 1000:.3f}s "
              f"p90={gaps[int(0.9 * (len(gaps) - 1))] / 1000:.3f}s")
    return rows


# ------------------------------------------------------------------ the replay


class RequestBudget:
    """The shared -n cap, and the run's stop signal.

    Every session gets a coroutine up front and each waits for its own target
    instant, so the cap cannot be a bare counter checked once per round: every
    session asleep when the last slot is taken would still wake and send (500
    sessions against -n 20 sent 519), and the sessions scheduled beyond that
    would keep the run alive, idle, until the whole timetable had played out.
    Claiming a slot before sending and waking every sleeper fixes both.
    """

    def __init__(self, limit):
        self._limit = limit or 0  # 0 = unlimited, as -n documents
        self._used = 0
        self._spent = asyncio.Event()
        # Rounds never issued because the cap was reached while they waited,
        # reported at the end so a short run is not mistaken for failures.
        self.dropped_waiting = 0

    def take(self) -> bool:
        if self._limit and self._used >= self._limit:
            return False
        self._used += 1
        if self._limit and self._used >= self._limit:
            self._spent.set()
        return True

    @property
    def exhausted(self) -> bool:
        return bool(self._limit) and self._used >= self._limit

    async def sleep_unless_spent(self, seconds: float) -> bool:
        """Sleep until due, or until the cap is reached. True = stop now."""
        if self.exhausted:
            self.dropped_waiting += 1
            return True
        try:
            await asyncio.wait_for(self._spent.wait(), timeout=seconds)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        self.dropped_waiting += 1
        return True


async def send_one(session, args, prompt_ids, out_len, request_id,
                   forced_ids=None):
    """Stream one completion, returning per-request timings and usage.

    Streaming (as in mooncake_trace_replay) is what makes TTFT observable: a
    non-streamed response only yields a total latency, and TTFT is the number
    this whole setup exists to measure. `include_usage` keeps the final chunk
    carrying `usage`, so the cached-token count survives the switch.
    """
    body = {
        "model": args.served_model_name or args.model,
        "prompt": prompt_ids,
        # Only output length is in the trace, not its content.
        "max_tokens": max(1, out_len),
        "min_tokens": max(1, out_len),
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if forced_ids:
        # Dynamo passes this through to vLLM's ForcedOutputTokensLogitsProcessor
        # and re-pins max/min_tokens to its length. It is dropped from the
        # prefill worker's params (that worker's single token is discarded), so
        # it shapes exactly the decode output the online judge reads.
        body["forced_output_tokens"] = forced_ids
    headers = {
        "Content-Type": "application/json",
        "request-id": request_id,
        "x-dynamo-request-id": request_id,
    }
    t0 = time.perf_counter()
    ttft = None
    num_chunks = 0
    usage = {}
    async with session.post(
        f"{args.url.rstrip('/')}/v1/completions", json=body, headers=headers
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"ok": False, "status": resp.status, "body": text[:300]}
        async for raw in resp.content:
            raw = raw.strip()
            if not raw:
                continue
            chunk = raw.decode("utf-8").removeprefix("data: ")
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except ValueError:
                continue
            choices = data.get("choices") or []
            if choices and choices[0].get("text"):
                num_chunks += 1
                if ttft is None:
                    # First chunk carrying text: prefill is done.
                    ttft = time.perf_counter() - t0
            if data.get("usage"):
                usage = data["usage"]
    latency = time.perf_counter() - t0
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "ok": True,
        "latency": latency,
        # No text ever arrived (e.g. out_len collapsed to nothing): report the
        # whole request as prefill rather than claiming a zero TTFT.
        "ttft": ttft if ttft is not None else latency,
        "num_chunks": num_chunks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # The number this whole token construction exists to make meaningful.
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "expected_output": max(1, out_len),
        "actual_output": completion,
    }


async def replay(args, sessions, pool, tok):
    import aiohttp

    sem = asyncio.Semaphore(args.max_inflight_req)
    # 0 = 不限，保持原行为：每个 session 到点就开跑。
    session_sem = (
        asyncio.Semaphore(args.max_inflight_session)
        if args.max_inflight_session > 0
        else None
    )
    results = []
    # One timeline for every target offset, mirroring mooncake_trace_replay's
    # t0_ms.
    t0_ms = time.perf_counter_ns() // 1_000_000
    plans = schedule_sessions(
        sessions, args.session_arrival_rate, args.time_scale, args.seed
    )
    # Stop issuing once spent, rather than trimming the session list, so every
    # session keeps its prefix chain intact.
    budget = RequestBudget(args.max_num_requests)

    async def run_session(session_id, turns, http, offsets):
        """One session's rounds strictly in order: N+1 only hits if N ran."""
        plans = plan_session(turns, BLOCK_SIZE)
        # Taken lazily, on the first round that actually sends, and held until
        # the session ends — think time between rounds included. Acquiring
        # before the round-0 due wait would let a session that is not due yet
        # occupy a slot; releasing between rounds would defeat the cap, since
        # what it bounds is how many distinct prefixes the server must keep
        # warm, and a session mid-think still owns its prefix.
        holding = False
        try:
            for t, plan, target_ms in zip(turns, plans, offsets):
                # Wait until this round is due, waking early if the cap is
                # reached. Sleeping per round rather than per session keeps the
                # trace's own think time between turns; a round already overdue
                # (the server is slower than the trace) is issued at once, which
                # is what makes the replay an open loop.
                due_ms = target_ms - (time.perf_counter_ns() // 1_000_000 - t0_ms)
                if due_ms > 0 and await budget.sleep_unless_spent(due_ms / 1000.0):
                    return
                if session_sem is not None and not holding:
                    await session_sem.acquire()
                    holding = True
                # Claimed before the request is built, so concurrent sessions
                # cannot each pass the check and then all send.
                if not budget.take():
                    return
                n_prompt = t["prefix"] + t["append"]
                prompt_ids = build_prompt(
                    session_id, n_prompt, pool, BLOCK_SIZE,
                    block_epoch=plan["block_epoch"],
                )
                # The tool calls go only into this round's output. They are
                # absent from the next round's prompt on purpose: measured over
                # the trace, prefix[N+1] tracks input[N] and not
                # input[N]+output[N] (92.4% vs 0.9% of consecutive pairs),
                # matching a decode side with no LMCache. Splicing the call text
                # into the next prompt would also shift every block boundary
                # plan_session() lines up, quietly lowering the hit rate this
                # replay exists to measure.
                forced_ids = (
                    build_forced_output(t.get("labels") or [], t["output"], tok, pool)
                    if args.forced_output
                    else None
                )
                idx = t["line_no"]
                async with sem:
                    r = await send_one(
                        http, args, prompt_ids,
                        t["output"], request_id_for(idx),
                        forced_ids=forced_ids,
                    )
                r.update(session=session_id, round_index=t["round_index"],
                         expected_prefix=t["prefix"], sent=n_prompt,
                         line_no=idx,
                         label="+".join(t.get("labels") or []) or "-")
                results.append(r)
                # Parsed by analyze_lp_ttft_breakdown.py. `idx` must match the
                # decimal in the request_id's leading 8 chars — that is the join
                # key to the server logs. `hashes` is a mooncake-only field.
                print(
                    f"[req] idx={idx} ts={int(target_ms)}ms"
                    f" in={n_prompt} out={t['output']} hashes=0"
                )
                if r["ok"]:
                    tpot = ((r["latency"] - r["ttft"]) / max(r["actual_output"] - 1, 1)
                            if r["actual_output"] > 1 else 0.0)
                    print(
                        f"[ok ] idx={idx}"
                        f" out={r['actual_output']}/{r['expected_output']}"
                        f" ttft={r['ttft'] * 1000:.1f}ms tpot={tpot * 1000:.2f}ms"
                        f" e2e={r['latency'] * 1000:.1f}ms chunks={r['num_chunks']}"
                    )
                else:
                    print(
                        f"[ERR] idx={idx} status={r.get('status')}"
                        f" in={n_prompt} out={t['output']}"
                    )
        finally:
            if holding:
                session_sem.release()

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_sec)
    conn = aiohttp.TCPConnector(limit=args.max_inflight_req * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as http:
        tasks = [
            run_session(sid, turns, http, offsets)
            for (sid, turns), offsets in zip(sessions, plans)
        ]
        await asyncio.gather(*tasks)
    if budget.dropped_waiting:
        print(f"到达 -n={args.max_num_requests}，"
              f"{budget.dropped_waiting:,} 个 session 在等待中放弃剩余轮次")
    return results


def summarize(results):
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"\n请求 {len(results):,}   成功 {len(ok):,}   失败 {len(bad):,}")
    if bad:
        first = bad[0]
        print(f"  首个失败: status={first.get('status')} "
              f"{str(first.get('body'))[:200]}")
    if not ok:
        return
    lat = sorted(r["latency"] for r in ok)
    print(f"延迟 p50 {lat[len(lat)//2]:.2f}s  "
          f"p90 {lat[int(len(lat)*0.9)]:.2f}s  max {lat[-1]:.2f}s")

    have = [r for r in ok if r.get("cached_tokens") is not None]
    if have:
        sent = sum(r["sent"] for r in have)
        cached = sum(r["cached_tokens"] for r in have)
        expect = sum(r["expected_prefix"] for r in have)
        print(f"prompt token 合计 {sent:,}   服务端报告命中 {cached:,} "
              f"({100*cached/max(sent,1):.1f}%)")
        print(f"trace 记录的 prefix 合计 {expect:,} "
              f"({100*expect/max(sent,1):.1f}%)  "
              f"-> 实测/期望 = {100*cached/max(expect,1):.1f}%")
    else:
        print("服务端未返回 cached_tokens（该字段需 vLLM 开启 prefix caching）")


def main():
    ap = argparse.ArgumentParser()
    # Flag names follow benchmark_serving_multi_turn.py.
    ap.add_argument("-i", "--input-file", default=DEFAULT_INPUT,
                    help=f"prepare_tracelab_trace.py 的产物（默认 {DEFAULT_INPUT}）")
    ap.add_argument("-u", "--url", default="http://localhost:8000")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--served-model-name", default=None,
                    help="请求体里的 model 字段，默认同 -m")
    ap.add_argument("--tokenizer", default=None,
                    help="构造 token pool 用，默认同 -m")
    ap.add_argument("--max-inflight-req", type=int, default=8,
                    help="并发请求数（同时在飞的请求数）")
    ap.add_argument("--max-inflight-session", type=int, default=0,
                    help="同时活跃的 session 数上限，0 = 不限。session 从第一轮"
                         "开始占位、到最后一轮结束才释放（含轮间 think time），"
                         "所以它约束的是服务端要同时保温多少条不同前缀，而不是在飞"
                         "请求数。不设时，高 --session-arrival-rate 会放进远多于 "
                         "--max-inflight-req 能服务的 session，绝大多数只发了首轮："
                         "请求摊薄在大量冷前缀上。设成接近 --max-inflight-req 可让"
                         "在服务的请求数和活跃 session 数相匹配，后续轮（有 KV 要"
                         "加载）占比更高")
    ap.add_argument("-n", "--max-num-requests", type=int, default=0,
                    help="总请求数上限，0 = 不限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--session-arrival-rate", type=float, default=0.0,
                    help="每秒新开 session 数（泊松到达）。session 起点重排到这个"
                         "速率，session 内部仍走 trace 的真实间隔。0 = 用 trace 自己"
                         "的起点，但它横跨 283 天，几乎全程空闲。并发 session 数约为 "
                         "rate x 184s（session 平均时长）/ --time-scale")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="session 内轮间隔的压缩倍数（>1 更快，同 mooncake）。"
                         "不影响 --session-arrival-rate 决定的 session 起点")
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="跳过 prompt 超过此长度的轮，防止超 --max-model-len")
    ap.add_argument("--token-offset", type=int, default=1000)
    ap.add_argument("--vocab-size", type=int, default=0,
                    help="扫 vocab 的上界，0 = 整个 vocab（同 mooncake）")
    ap.add_argument("--request-timeout-sec", type=float, default=600.0)
    ap.add_argument("--forced-output", action="store_true",
                    help="把每轮的 tool call 渲染成 <function=NAME> 并强制 decode "
                         "按它输出。在线 dram-safe 判据靠正则从 decode 明文里认 "
                         "call 类型，不开这个开关它永远认不到，会把每轮都判成"
                         "「会话结束」，判据等于关掉")
    ap.add_argument("--schedule-out", default=DEFAULT_SCHEDULE, metavar="PATH",
                    help=f"调度时刻表的写出位置（默认 {DEFAULT_SCHEDULE}）。"
                         "每轮一行，含 target_ms 与长度；首行是产生它的参数")
    args = ap.parse_args()

    sessions = load_turns(args.input_file, args.max_prompt_tokens)
    if not sessions:
        sys.exit(f"{args.input_file} 没有数据，先跑 prepare_tracelab_trace.py")
    n_turns = sum(len(t) for _, t in sessions)
    print(f"session {len(sessions):,}   轮 {n_turns:,}   "
          f"block_size {BLOCK_SIZE}（写死，须与 vLLM --block-size 一致）")

    # Written before the server and tokenizer checks, so the timetable exists
    # even if the run cannot start.
    dump_schedule(
        sessions,
        schedule_sessions(
            sessions, args.session_arrival_rate, args.time_scale, args.seed
        ),
        args.schedule_out,
        {
            "seed": args.seed,
            "session_arrival_rate": args.session_arrival_rate,
            "time_scale": args.time_scale,
            "max_prompt_tokens": args.max_prompt_tokens,
            "num_sessions": len(sessions),
            "num_rounds": n_turns,
        },
    )

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        sys.exit("需要 aiohttp 才能发请求：pip3 install --user aiohttp")

    # After the import check: scanning the vocab takes ~20s.
    loaded = load_pool(args.tokenizer or args.model, args.token_offset,
                       args.vocab_size)
    if loaded is None:
        sys.exit("需要 transformers 构造 token pool（只用 tokenizer，不需要 torch）")
    pool, tok = loaded
    print(f"token pool {len(pool):,} 个纯 ASCII token")
    if args.forced_output:
        n_calls = sum(len(t.get("labels") or []) for _, ts in sessions for t in ts)
        print(f"forced output: 已开启，{n_turns:,} 轮共 {n_calls:,} 个 tool call "
              f"会以 <function=NAME> 出现在 decode 输出里")
    else:
        print("forced output: 未开启 —— 在线 dram-safe 判据将认不出 tool call，"
              "每轮都会判成会话结束（要测判据请加 --forced-output）")

    t0 = time.time()
    results = asyncio.run(replay(args, sessions, pool, tok))
    print(f"\n耗时 {time.time()-t0:.1f}s")
    summarize(results)


if __name__ == "__main__":
    main()

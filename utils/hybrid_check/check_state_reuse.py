"""Hybrid model (Qwen3.6) prefix reuse check: a prompt's greedy output must not
change when its prefix is reused instead of computed -- the full-attention KV
from HBM, DRAM or SSD, the linear-attention state at the block boundary always
from HBM (states are never offloaded).

    python3 check_state_reuse.py --model <path> [--url http://localhost:8000]

cache_use_control picks the KV's tier (1 HBM, 2 DRAM, 4 SSD); [3] (remote
DRAM, absent on one host) reuses nothing, a cold prefill. Each case prints the
hit the prefill log should show and how much of the output matches the cold
run.
"""
import argparse
import json
import random
import time
import urllib.request
import uuid


def complete(url: str, model: str, prompt: list[int], control: list[int], n: int) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": n,
        "min_tokens": n,
        "ignore_eos": True,
        "temperature": 0.0,
        "guided_regex": "[ -~]*",
        "cache_use_control": control,
    }
    rid = str(uuid.uuid4())
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-dynamo-request-id": rid},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    return out["choices"][0]["text"]


def common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--block", type=int, default=1536)
    ap.add_argument("--tokens", type=int, default=7000, help="first turn's prompt")
    ap.add_argument("--more", type=int, default=3000, help="next turn's new tokens")
    ap.add_argument("--out", type=int, default=32)
    ap.add_argument("--ssd-wait", type=float, default=8.0, help="s for the SSD write-through")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    turn1 = [rng.randrange(100, a.vocab - 100) for _ in range(a.tokens)]
    turn2 = turn1 + [rng.randrange(100, a.vocab - 100) for _ in range(a.more)]
    hit1 = (a.tokens - 1) // a.block * a.block
    hit2 = (len(turn2) - 1) // a.block * a.block

    cold1 = complete(a.url, a.model, turn1, [3], a.out)
    print(f"turn 1 ({a.tokens} tokens) cold: {cold1!r}")
    cases = [("turn 1, DRAM", turn1, [2], cold1, hit1), ("turn 1, HBM", turn1, [1], cold1, hit1)]
    results = []
    for name, prompt, control, ref, hit in cases:
        text = complete(a.url, a.model, prompt, control, a.out)
        results.append((name, hit, text, ref))
    time.sleep(a.ssd_wait)
    results.append(("turn 1, SSD", hit1, complete(a.url, a.model, turn1, [4], a.out), cold1))

    cold2 = complete(a.url, a.model, turn2, [3], a.out)
    print(f"turn 2 ({len(turn2)} tokens) cold: {cold2!r}")
    results.append(("turn 2, cold again", 0, complete(a.url, a.model, turn2, [3], a.out), cold2))
    # Turn 2 cold saved its own boundary too: the HBM and DRAM cases hit the
    # longer one.
    hbm2 = complete(a.url, a.model, turn2, [1], a.out)
    results.append(("turn 2, HBM", hit2, hbm2, cold2))
    dram2 = complete(a.url, a.model, turn2, [2], a.out)
    results.append(("turn 2, DRAM", hit2, dram2, cold2))
    results.append(("turn 2, DRAM vs HBM", hit2, dram2, hbm2))
    # Another continuation of turn 1: its KV past turn 1 comes from its own
    # cold run, the state from that run's end.
    turn3 = turn1 + [rng.randrange(100, a.vocab - 100) for _ in range(a.more)]
    cold3 = complete(a.url, a.model, turn3, [3], a.out)
    hbm3 = complete(a.url, a.model, turn3, [1], a.out)
    dram3 = complete(a.url, a.model, turn3, [2], a.out)
    results.append(("turn 2', HBM", hit2, hbm3, cold3))
    results.append(("turn 2', DRAM, turn 1's KV shared", hit2, dram3, cold3))
    results.append(("turn 2', DRAM vs HBM", hit2, dram3, hbm3))
    ok = True
    for name, hit, text, ref in results:
        same = common_prefix(text, ref)
        ok &= same == len(ref)
        print(f"{name:34s} expected hit {hit:6d} tokens | output matches cold for "
              f"{same}/{len(ref)} chars | {text!r}")
    print("ALL MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()

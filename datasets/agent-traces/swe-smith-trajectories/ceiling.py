#!/usr/bin/env python3
"""Upper bound on any predictor built from the features the judge can see.

Before tuning constants or percentiles it is worth knowing whether the target is
reachable. The online judge sees prefix, decode_len and the tool-call label, and
turns them into one number (an estimated append) that feeds a monotone
threshold. So the best any such predictor can do is:

  - the best single threshold on the model's own load/compute ratio, and
  - the best achievable if each label could pick its own ideal response length
    (an oracle over the one knob the lookup table controls).

If the oracle is far below the target, the gap is in the features, not in the
constants, and no amount of sweeping closes it.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../vllm-dev/vllm/v1/core"
    ),
)
from kvpilot_predict import derive_constants, is_dram_safe  # noqa: E402


def best_threshold(rows):
    """Best accuracy over every threshold on the model ratio."""
    ordered = sorted(rows, key=lambda r: r[0])
    n = len(ordered)
    total_safe = sum(1 for r in ordered if r[1])
    # Predict safe below the cut, unsafe above it.
    best = (0.0, None)
    safe_below = 0
    for i, (ratio, truth) in enumerate(ordered):
        # cut after position i
        safe_below += truth
        unsafe_above = (n - i - 1) - (total_safe - safe_below)
        acc = (safe_below + unsafe_above) / n
        if acc > best[0]:
            best = (acc, ratio)
    # Also consider the degenerate all-unsafe cut.
    acc_all_unsafe = (n - total_safe) / n
    if acc_all_unsafe > best[0]:
        best = (acc_all_unsafe, float("-inf"))
    return best


def main():
    measured = json.load(open(sys.argv[1]))["requests"]
    ratio = json.load(open(sys.argv[2]))["requests"]
    k = derive_constants("h20")

    shared = sorted(set(measured) & set(ratio), key=int)
    rows = [
        (ratio[x]["ratio"], bool(measured[x]["dram_safe"]), ratio[x]["label"],
         ratio[x]["prefix"], ratio[x]["decode_len"])
        for x in shared if ratio[x].get("ratio") is not None
    ]
    n = len(rows)
    n_safe = sum(1 for r in rows if r[1])
    print(f"{n:,} rows, measured safe {100*n_safe/n:.1f}%")
    print(f"majority baseline: {100*max(n_safe, n-n_safe)/n:.1f}%\n")

    acc, cut = best_threshold([(r[0], r[1]) for r in rows])
    print(f"best single threshold on model ratio: {100*acc:.1f}% (cut at {cut:.4f})")
    print("  -- the ceiling for any monotone rule on this ratio\n")

    # Per-label oracle: choose each label's response length to maximize its own
    # accuracy. This is the best the lookup table could ever do.
    by_label = defaultdict(list)
    for _, truth, label, prefix, dlen in rows:
        by_label[label].append((truth, prefix, dlen))

    hit = 0
    for label, items in by_label.items():
        # Candidate response lengths: enough to span the decision boundary.
        cands = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                 16384, 65536]
        best_label = max(
            sum(1 for truth, p, d in items
                if is_dram_safe(p, d + c, k) == truth)
            for c in cands
        )
        hit += best_label
    print(f"per-label oracle response length: {100*hit/n:.1f}%")
    print("  -- the ceiling for tuning tool_resp_len.json / --predict-percentile")


if __name__ == "__main__":
    main()

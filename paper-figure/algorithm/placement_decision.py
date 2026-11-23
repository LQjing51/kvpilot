"""Placement safe-probability from tool type + prefix + decode length.

    P_hbm, P_dram, P_ssd = decide(tool_type, prefix, decode_len,
                                  dist, coeffs, prev_input_len)

Length identity for consecutive turns (block size B=512):

    input_i  = input_{i-1} + decode_{i-1} + tool_resp
    prefix_i = align_B(input_{i-1}) = input_{i-1} - input_{i-1} % B
    suffix_i = input_i - prefix_i
             = (input_{i-1} % B) + decode_{i-1} + tool_resp

So the random append compared to safe-suffix boundaries is NOT
decode+tool_resp; it is shifted by the previous turn's partial block.

1. Look up the tool-response length histogram for `tool_type`.
2. Shift every bucket by rem+decode, rem = prev_input_len % 512.
3. Integrate mass over [0, dram_b) / [dram_b, ssd_b) / [ssd_b, ∞).
   A boundary that cuts a width-10 bucket splits that bucket's probability
   linearly by length.

    python3 placement_decision.py --dataset swe \\
        --tool 'bash:python' --prefix 16384 --decode-len 100 --prev-input-len 16700
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resp_length_distribution import dist_path, load as load_dist, label_of
from safe_boundary_fitting import load_coeffs, predict_boundaries

HERE = Path(__file__).resolve().parent
BLOCK = 512


def block_remainder(prev_input_len):
    return int(prev_input_len) % BLOCK


def suffix_shift(prev_input_len, decode_len):
    """Deterministic part of suffix: (input_{i-1} % 512) + decode."""
    return block_remainder(prev_input_len) + int(decode_len)


def _allocate_interval(lo, hi, mass, dram_b, ssd_b):
    """Split probability `mass` on append ∈ [lo, hi) across the three tiers.

    Cuts at dram_b / ssd_b inside the interval get a length-proportional share
    (linear split of a bucket that a boundary crosses).
    """
    if hi <= lo or mass <= 0:
        return 0.0, 0.0, 0.0
    cuts = [lo]
    for b in (dram_b, ssd_b):
        if lo < b < hi:
            cuts.append(b)
    cuts.append(hi)
    width = hi - lo
    hbm = dram = ssd = 0.0
    for a, b in zip(cuts, cuts[1:]):
        frac = (b - a) / width
        # Assign by which side of the boundaries the open interval sits on.
        # Use the left endpoint: [a,b) is wholly inside one tier between cuts.
        if a < dram_b:
            hbm += mass * frac
        elif a < ssd_b:
            dram += mass * frac
        else:
            ssd += mass * frac
    return hbm, dram, ssd


def _iter_buckets(hist):
    """Yield (lo, hi, prob) for each non-zero tool-resp bucket."""
    if "buckets" in hist:
        for lo, hi, p in hist["buckets"]:
            yield lo, hi, p
        return
    edges, probs = hist["edges"], hist["probs"]
    for i, p in enumerate(probs):
        if p > 0:
            yield edges[i], edges[i + 1], p


def safe_probs_from_hist(hist, shift, dram_b, ssd_b):
    """(P_hbm, P_dram, P_ssd); append = shift + tool_resp.

    `shift` should be (prev_input_len % 512) + decode_len.
    """
    if hist is None:
        raise ValueError("missing histogram")
    sh = int(shift)
    tot = [0.0, 0.0, 0.0]
    for lo0, hi0, p in _iter_buckets(hist):
        h, dr, s = _allocate_interval(lo0 + sh, hi0 + sh, p, dram_b, ssd_b)
        tot[0] += h
        tot[1] += dr
        tot[2] += s
    ssum = sum(tot)
    if ssum <= 0:
        return 0.0, 0.0, 0.0
    return tot[0] / ssum, tot[1] / ssum, tot[2] / ssum


def decide(tool_type, prefix, decode_len, dist_for_dataset, coeffs, prev_input_len):
    """Return (P_hbm, P_dram, P_ssd, dram_b, ssd_b).

    `prev_input_len` is input_{i-1}; together with decode_len it sets
    suffix = (prev_input_len % 512) + decode_len + tool_resp. `prefix` should
    be align_512(prev_input_len); a mismatch only warns.
    """
    if isinstance(tool_type, (list, tuple)):
        tool_type = label_of(tool_type)
    hist = dist_for_dataset.get(tool_type)
    if hist is None:
        raise KeyError(f"unknown tool type {tool_type!r}; "
                       f"{len(dist_for_dataset)} labels in table")
    prev = int(prev_input_len)
    align = prev - (prev % BLOCK)
    if int(prefix) != align:
        print(f"warning: prefix={prefix} != align512(prev_input={prev})={align}",
              file=sys.stderr)
    dram_b, ssd_b = predict_boundaries(int(prefix), coeffs)
    shift = suffix_shift(prev, decode_len)
    probs = safe_probs_from_hist(hist, shift, dram_b, ssd_b)
    return (*probs, dram_b, ssd_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("swe", "tracelab"), required=True)
    ap.add_argument("--tool", required=True, help="joined label, e.g. bash:python or Read+Read")
    ap.add_argument("--prefix", type=int, required=True)
    ap.add_argument("--decode-len", type=int, required=True)
    ap.add_argument("--prev-input-len", type=int, required=True,
                    help="input length of the previous turn (for %%512 remainder)")
    ap.add_argument("--dist", type=Path, default=None,
                    help="default resp_length_dist_<dataset>.json")
    ap.add_argument("--coeffs", type=Path, default=HERE / "safe_boundary_coeffs.json")
    ap.add_argument("--demo-split", action="store_true",
                    help="print the README example (0.3,0.2,0.5) with bounds 20/1000")
    args = ap.parse_args()

    if args.demo_split:
        hist = {"bucket_width": 10, "n": 0, "max_resp": 30,
                "buckets": [[0, 10, 0.3], [10, 20, 0.2], [20, 30, 0.5]]}
        for dram_b, ssd_b, note in (
            (20, 1000, "bounds on bucket edges"),
            (15, 1000, "dram bound splits [10,20)"),
        ):
            ph, pd, ps = safe_probs_from_hist(hist, 0, dram_b, ssd_b)
            print(f"{note}: dram_b={dram_b} ssd_b={ssd_b} -> "
                  f"HBM={ph:.3f} DRAM={pd:.3f} SSD={ps:.3f}")
        return

    args.dist = args.dist or dist_path(args.dataset)
    if not args.dist.exists():
        sys.exit(f"missing {args.dist}; run resp_length_distribution.py first")
    if not args.coeffs.exists():
        sys.exit(f"missing {args.coeffs}; run safe_boundary_fitting.py first")

    dist = load_dist(args.dist)
    coeffs = load_coeffs(args.coeffs)
    rem = block_remainder(args.prev_input_len)
    shift = suffix_shift(args.prev_input_len, args.decode_len)
    ph, pd, ps, db, sb = decide(
        args.tool, args.prefix, args.decode_len, dist, coeffs, args.prev_input_len)
    print(f"tool={args.tool!r} prefix={args.prefix} decode_len={args.decode_len} "
          f"prev_input={args.prev_input_len} rem={rem} shift={shift}")
    print(f"boundaries: dram={db}  ssd={sb}")
    print(f"safe probs: HBM={ph:.4f}  DRAM={pd:.4f}  SSD={ps:.4f}")


if __name__ == "__main__":
    main()

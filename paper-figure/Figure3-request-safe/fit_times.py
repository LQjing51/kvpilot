"""Fit (prefix, suffix) -> (t_hbm, t_dram, t_ssd) from request-safe e2e,
then invert to DRAM / SSD safe-suffix boundaries.

    python3 fit_times.py --datasets swe --out fit_times_swe.json

Model (layerwise overlap + first-layer residual that never fully hides):

    t_hbm(p, s) = a * s * (p + s/2) + b * s + d
    L(p)        = c * p + f                 # load-bound e2e plateau
    floor(p)    = max(0, g * p + h)         # residual once compute-bound
    t_dram      = max(L_dram(p), t_hbm + floor_dram(p))
    t_ssd       = max(L_ssd(p),  t_hbm + floor_ssd(p))

Safe (exposed / t_hbm < 5%) iff
    t_hbm > max(L / 1.05, floor / 0.05).

L is the per-prefix median tier e2e over suffix ≤ 256 (load fully exposed).
floor is the per-prefix median of (t_tier - t_hbm) among compute-bound
points (t_hbm > 1.1 L, suffix ≥ 512). Compute coeffs use suffix ≤ 4K.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tier_logs import dataset

HERE = Path(__file__).resolve().parent
THRESH = 0.05
DATASETS = ("toolagent", "swe", "tracelab")
COMPUTE_SUFFIX_MAX = 4096
L_SUFFIX_MAX = 256


def load_points(raw="raw", datasets=DATASETS):
    pts = []
    for name in datasets:
        rows, res, _ = dataset(HERE / raw / name)
        meta = {}
        for r in rows:
            if r["kind"] in ("t1", "t2", "t4"):
                meta[r["src_line"]] = (r["prefix"], r["input_length"] - r["prefix"])
        for line, (p, s) in meta.items():
            e = {k: res.get((line, k), (None, "m"))[0] for k in ("t1", "t2", "t4")}
            if None in e.values() or p == 0 or s <= 0:
                continue
            pts.append((p, s, e["t1"], e["t2"], e["t4"]))
    return np.asarray(pts, dtype=np.float64)


def wls(X, y, w=None):
    y = np.asarray(y, dtype=np.float64)
    if w is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    sw = np.sqrt(np.asarray(w, dtype=np.float64))
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def ape(pred, y):
    return np.abs(pred - y) / np.maximum(y, 1e-9)


def per_prefix_median(p, values, mask, min_n=1):
    by = defaultdict(list)
    for pi, v, keep in zip(p, values, mask):
        if keep:
            by[int(pi)].append(float(v))
    xs, ys = [], []
    for pi, vs in sorted(by.items()):
        if len(vs) < min_n:
            continue
        xs.append(float(pi))
        ys.append(float(np.median(vs)))
    return np.asarray(xs), np.asarray(ys)


def fit_line(xs, ys, relative=False):
    X = np.column_stack([xs, np.ones(len(xs))])
    w = (1.0 / np.maximum(ys, 1.0) ** 2) if relative else None
    return wls(X, ys, w)


def fit(pts):
    p, s, t1, t2, t4 = pts.T
    m = s <= COMPUTE_SUFFIX_MAX
    Xc = np.column_stack([s[m] * (p[m] + s[m] / 2.0), s[m], np.ones(m.sum())])
    cc = wls(Xc, t1[m], 1.0 / np.maximum(t1[m], 1.0) ** 2)

    xd, yd = per_prefix_median(p, t2, s <= L_SUFFIX_MAX, min_n=1)
    xs, ys = per_prefix_median(p, t4, s <= L_SUFFIX_MAX, min_n=1)
    cd, cs = fit_line(xd, yd, relative=True), fit_line(xs, ys, relative=True)

    def floor_points(tier, cL):
        L = cL[0] * p + cL[1]
        mask = (t1 > 1.1 * L) & (s >= 512)
        return per_prefix_median(p, np.maximum(0.0, tier - t1), mask, min_n=3)

    xfd, yfd = floor_points(t2, cd)
    xfs, yfs = floor_points(t4, cs)
    # Unweighted: SSD residual grows with prefix; relative WLS would crush it.
    fd = fit_line(xfd, yfd, relative=False) if len(xfd) >= 3 else np.array([0.0, 0.0])
    fs = fit_line(xfs, yfs, relative=False) if len(xfs) >= 3 else np.array([0.0, 0.0])

    return {
        "a": float(cc[0]), "b": float(cc[1]), "d": float(cc[2]),
        "c_dram": float(cd[0]), "f_dram": float(cd[1]),
        "c_ssd": float(cs[0]), "f_ssd": float(cs[1]),
        "g_dram": float(fd[0]), "h_dram": float(fd[1]),
        "g_ssd": float(fs[0]), "h_ssd": float(fs[1]),
        "n": int(len(pts)),
        "n_compute": int(m.sum()),
        "n_L_dram": int(len(xd)), "n_L_ssd": int(len(xs)),
        "n_floor_dram": int(len(xfd)), "n_floor_ssd": int(len(xfs)),
        "compute_suffix_max": COMPUTE_SUFFIX_MAX,
        "L_suffix_max": L_SUFFIX_MAX,
        "thresh": THRESH,
    }


def times(p, s, k):
    """(t_hbm, t_dram, t_ssd) in ms; p, s scalar or ndarray."""
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    th = k["a"] * s * (p + s / 2.0) + k["b"] * s + k["d"]
    Ld = k["c_dram"] * p + k["f_dram"]
    Ls = k["c_ssd"] * p + k["f_ssd"]
    fd = np.maximum(0.0, k["g_dram"] * p + k["h_dram"])
    fs = np.maximum(0.0, k["g_ssd"] * p + k["h_ssd"])
    return th, np.maximum(Ld, th + fd), np.maximum(Ls, th + fs)


def tier_of(th, td, ts):
    r2 = (td - th) / th
    r4 = (ts - th) / th
    out = np.full(np.shape(th), "HBM safe", dtype=object)
    out[r2 < THRESH] = "DRAM safe"
    out[r4 < THRESH] = "SSD safe"
    return out


def _need(p, medium, k):
    if medium == "dram":
        L = k["c_dram"] * p + k["f_dram"]
        fl = max(0.0, k["g_dram"] * p + k["h_dram"])
    else:
        L = k["c_ssd"] * p + k["f_ssd"]
        fl = max(0.0, k["g_ssd"] * p + k["h_ssd"])
    need = L / (1.0 + THRESH)
    if fl > 0:
        need = max(need, fl / THRESH)
    return need


def bound_suffix(p, medium, k, hi=200_000):
    """Smallest suffix whose predicted t_hbm hides that medium under 5%."""
    need = _need(p, medium, k)
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        th, _, _ = times(p, mid, k)
        if float(np.asarray(th)) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bounds(p, k):
    d = bound_suffix(p, "dram", k)
    s = max(bound_suffix(p, "ssd", k), d)
    return d, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default="fit_times.json")
    args = ap.parse_args()
    pts = load_points(args.raw, args.datasets)
    k = fit(pts)
    p, s, t1, t2, t4 = pts.T
    th, td, ts = times(p, s, k)

    print(f"points {k['n']}  compute n={k['n_compute']} (s≤{COMPUTE_SUFFIX_MAX})  "
          f"L n={k['n_L_dram']}/{k['n_L_ssd']} (s≤{L_SUFFIX_MAX})  "
          f"floor n={k['n_floor_dram']}/{k['n_floor_ssd']}")
    print(f"t_hbm = {k['a']:.6e}*s*(p+s/2) + {k['b']:.6e}*s + {k['d']:.3f}")
    print(f"L_dram = {k['c_dram']:.6e}*p + {k['f_dram']:.3f}  "
          f"({k['c_dram']*1024:.3f} ms/1K)   "
          f"floor_dram = max(0, {k['g_dram']:.6e}*p + {k['h_dram']:.3f})")
    print(f"L_ssd  = {k['c_ssd']:.6e}*p + {k['f_ssd']:.3f}  "
          f"({k['c_ssd']*1024:.3f} ms/1K)   "
          f"floor_ssd  = max(0, {k['g_ssd']:.6e}*p + {k['h_ssd']:.3f})")
    print("t_dram = max(L_dram, t_hbm + floor_dram)")
    print("t_ssd  = max(L_ssd,  t_hbm + floor_ssd)")

    near = s <= COMPUTE_SUFFIX_MAX
    for name, pred, y in (("t_hbm", th, t1), ("t_dram", td, t2), ("t_ssd", ts, t4)):
        a = ape(pred, y)
        print(f"  {name} medAPE all={np.median(a)*100:.1f}%  "
              f"s≤{COMPUTE_SUFFIX_MAX}={np.median(a[near])*100:.1f}%")

    tm, tp = tier_of(t1, t2, t4), tier_of(th, td, ts)
    mid = (p >= 4096) & (p <= 61440)
    print(f"3-tier agreement {(tm == tp).mean()*100:.1f}%  "
          f"4K–60K {(tm[mid] == tp[mid]).mean()*100:.1f}%  "
          f"HBM-vs-not {((tm == 'HBM safe') == (tp == 'HBM safe')).mean()*100:.1f}%")

    fig2 = HERE.parent / "Figure2-safe-suffix-boundary" / "raw"
    curve = {"dram": {}, "ssd": {}}
    for d in (fig2 / "1p1d", fig2 / "1p1d-ext"):
        cf = d / "confirm" / "search.jsonl"
        if not cf.exists():
            continue
        for r in map(json.loads, open(cf)):
            if "boundary" in r and r["boundary"] and (
                    not r.get("note") or str(r["note"]).startswith("safe down")):
                curve[r["medium"]][r["prefix"]] = r["boundary"]

    print("\nprefix -> dram_bound, ssd_bound  (fit | Figure2 confirm)")
    err = {"dram": [], "ssd": []}
    prefixes = sorted(set(range(4096, 65537, 4096)) | set(curve["dram"]) | set(curve["ssd"]))
    for pp in prefixes:
        db, sb = bounds(pp, k)
        g2d, g2s = curve["dram"].get(pp), curve["ssd"].get(pp)
        extra = f"  | fig2 {g2d}/{g2s}" if g2d is not None or g2s is not None else ""
        print(f"  p={pp:6d}  dram={db:5d}  ssd={sb:5d}{extra}")
        if g2d:
            err["dram"].append(abs(db - g2d) / g2d)
        if g2s:
            err["ssd"].append(abs(sb - g2s) / g2s)
    if err["dram"]:
        print(f"medAPE vs Figure2 boundary: dram={np.median(err['dram'])*100:.1f}%  "
              f"ssd={np.median(err['ssd'])*100:.1f}%")

    keys = ("a", "b", "d", "c_dram", "f_dram", "c_ssd", "f_ssd",
            "g_dram", "h_dram", "g_ssd", "h_ssd",
            "n", "n_compute", "n_L_dram", "n_L_ssd",
            "n_floor_dram", "n_floor_ssd",
            "compute_suffix_max", "L_suffix_max", "thresh")
    path = HERE / args.out
    path.write_text(json.dumps({key: k[key] for key in keys}, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(HERE)}")
    print("API: times(p, s, k), bounds(p, k), bound_suffix(p, medium, k)")


if __name__ == "__main__":
    main()

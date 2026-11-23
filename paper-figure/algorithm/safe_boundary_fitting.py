"""Safe-suffix boundary from measured (prefix, suffix, t_hbm, t_dram, t_ssd).

Two entry points:

    fit(points) -> coeffs
        points: iterable of (prefix, suffix, t_hbm, t_dram, t_ssd) in ms
    predict_boundaries(prefix, coeffs) -> (dram_bound, ssd_bound)
        smallest suffix that hides DRAM / SSD load under the 5% threshold

Also: times(prefix, suffix, coeffs) -> (t_hbm, t_dram, t_ssd).

    python3 safe_boundary_fitting.py                  # fit Figure3 request-safe
    python3 safe_boundary_fitting.py --prefix 16384   # print two bounds
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THRESH = 0.05
COMPUTE_SUFFIX_MAX = 4096
L_SUFFIX_MAX = 256

# Coeff keys persisted to JSON / returned by fit().
KEYS = (
    "a", "b", "d",
    "c_dram", "f_dram", "c_ssd", "f_ssd",
    "g_dram", "h_dram", "g_ssd", "h_ssd",
    "thresh",
)


def _wls(X, y, w=None):
    y = np.asarray(y, dtype=np.float64)
    if w is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    sw = np.sqrt(np.asarray(w, dtype=np.float64))
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def _per_prefix_median(p, values, mask, min_n=1):
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


def _fit_line(xs, ys, relative=False):
    X = np.column_stack([xs, np.ones(len(xs))])
    w = (1.0 / np.maximum(ys, 1.0) ** 2) if relative else None
    return _wls(X, ys, w)


def fit(points):
    """Fit time curves from measured request-safe points.

    Model (ms):
        t_hbm(p, s) = a*s*(p + s/2) + b*s + d
        L(p)        = c*p + f
        floor(p)    = max(0, g*p + h)
        t_dram      = max(L_dram(p), t_hbm + floor_dram(p))
        t_ssd       = max(L_ssd(p),  t_hbm + floor_ssd(p))

    Safe when exposed / t_hbm < 5%, i.e. t_hbm > max(L/1.05, floor/0.05).
    """
    pts = np.asarray(list(points), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 5:
        raise ValueError("points must be Nx5: prefix, suffix, t_hbm, t_dram, t_ssd")
    p, s, t1, t2, t4 = pts.T

    m = s <= COMPUTE_SUFFIX_MAX
    Xc = np.column_stack([s[m] * (p[m] + s[m] / 2.0), s[m], np.ones(int(m.sum()))])
    cc = _wls(Xc, t1[m], 1.0 / np.maximum(t1[m], 1.0) ** 2)

    xd, yd = _per_prefix_median(p, t2, s <= L_SUFFIX_MAX, min_n=1)
    xs, ys = _per_prefix_median(p, t4, s <= L_SUFFIX_MAX, min_n=1)
    if len(xd) < 2 or len(xs) < 2:
        raise ValueError("need ≥2 prefixes with suffix ≤ L_SUFFIX_MAX for L fit")
    cd, cs = _fit_line(xd, yd, relative=True), _fit_line(xs, ys, relative=True)

    def floor_pts(tier, cL):
        L = cL[0] * p + cL[1]
        return _per_prefix_median(
            p, np.maximum(0.0, tier - t1), (t1 > 1.1 * L) & (s >= 512), min_n=3)

    xfd, yfd = floor_pts(t2, cd)
    xfs, yfs = floor_pts(t4, cs)
    fd = _fit_line(xfd, yfd) if len(xfd) >= 3 else np.array([0.0, 0.0])
    fs = _fit_line(xfs, yfs) if len(xfs) >= 3 else np.array([0.0, 0.0])

    return {
        "a": float(cc[0]), "b": float(cc[1]), "d": float(cc[2]),
        "c_dram": float(cd[0]), "f_dram": float(cd[1]),
        "c_ssd": float(cs[0]), "f_ssd": float(cs[1]),
        "g_dram": float(fd[0]), "h_dram": float(fd[1]),
        "g_ssd": float(fs[0]), "h_ssd": float(fs[1]),
        "thresh": THRESH,
        "n": int(len(pts)),
    }


def times(prefix, suffix, coeffs):
    """(t_hbm, t_dram, t_ssd) in ms; prefix/suffix scalar or ndarray."""
    p = np.asarray(prefix, dtype=np.float64)
    s = np.asarray(suffix, dtype=np.float64)
    th = coeffs["a"] * s * (p + s / 2.0) + coeffs["b"] * s + coeffs["d"]
    Ld = coeffs["c_dram"] * p + coeffs["f_dram"]
    Ls = coeffs["c_ssd"] * p + coeffs["f_ssd"]
    fd = np.maximum(0.0, coeffs["g_dram"] * p + coeffs["h_dram"])
    fs = np.maximum(0.0, coeffs["g_ssd"] * p + coeffs["h_ssd"])
    return th, np.maximum(Ld, th + fd), np.maximum(Ls, th + fs)


def _need(prefix, medium, coeffs):
    thr = coeffs.get("thresh", THRESH)
    if medium == "dram":
        L = coeffs["c_dram"] * prefix + coeffs["f_dram"]
        fl = max(0.0, coeffs["g_dram"] * prefix + coeffs["h_dram"])
    else:
        L = coeffs["c_ssd"] * prefix + coeffs["f_ssd"]
        fl = max(0.0, coeffs["g_ssd"] * prefix + coeffs["h_ssd"])
    need = L / (1.0 + thr)
    if fl > 0:
        need = max(need, fl / thr)
    return need


def bound_suffix(prefix, medium, coeffs, hi=200_000):
    need = _need(prefix, medium, coeffs)
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        th, _, _ = times(prefix, mid, coeffs)
        if float(np.asarray(th)) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo


def predict_boundaries(prefix, coeffs):
    """(dram_safe_bound, ssd_safe_bound): min suffix that is DRAM- / SSD-safe."""
    d = bound_suffix(prefix, "dram", coeffs)
    s = max(bound_suffix(prefix, "ssd", coeffs), d)
    return int(d), int(s)


def load_coeffs(path):
    return json.loads(Path(path).read_text())


def save_coeffs(coeffs, path):
    out = {k: coeffs[k] for k in KEYS if k in coeffs}
    Path(path).write_text(json.dumps(out, indent=2) + "\n")


def load_figure3_points():
    """Optional: pull (p,s,t1,t2,t4) from Figure3-request-safe raw logs."""
    fig3 = HERE.parent / "Figure3-request-safe"
    sys.path.insert(0, str(fig3))
    from tier_logs import dataset  # noqa: WPS433

    pts = []
    for name in ("toolagent", "swe", "tracelab"):
        if not (fig3 / "raw" / name).is_dir():
            continue
        rows, res, _ = dataset(fig3 / "raw" / name)
        meta = {}
        for r in rows:
            if r["kind"] in ("t1", "t2", "t4"):
                meta[r["src_line"]] = (r["prefix"], r["input_length"] - r["prefix"])
        for line, (p, s) in meta.items():
            e = {k: res.get((line, k), (None, "m"))[0] for k in ("t1", "t2", "t4")}
            if None in e.values() or p == 0 or s <= 0:
                continue
            pts.append((p, s, e["t1"], e["t2"], e["t4"]))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true", help="fit from Figure3 request-safe (default if no --prefix)")
    ap.add_argument("--coeffs", type=Path, default=HERE / "safe_boundary_coeffs.json")
    ap.add_argument("--prefix", type=int, default=None, help="predict two boundaries for this prefix")
    ap.add_argument("--points", type=Path, default=None,
                    help="Nx5 JSON list [[p,s,t1,t2,t4], ...] instead of Figure3 logs")
    args = ap.parse_args()

    if args.prefix is not None and args.coeffs.exists() and not args.fit and args.points is None:
        coeffs = load_coeffs(args.coeffs)
        d, s = predict_boundaries(args.prefix, coeffs)
        print(f"prefix={args.prefix}  dram_bound={d}  ssd_bound={s}")
        return

    if args.points:
        points = json.loads(args.points.read_text())
    else:
        print("loading Figure3-request-safe points …", flush=True)
        points = load_figure3_points()
    coeffs = fit(points)
    save_coeffs(coeffs, args.coeffs)
    print(f"fit n={coeffs['n']} -> {args.coeffs}")
    print(f"t_hbm = {coeffs['a']:.6e}*s*(p+s/2) + {coeffs['b']:.6e}*s + {coeffs['d']:.3f}")
    print(f"L_dram = {coeffs['c_dram']:.6e}*p + {coeffs['f_dram']:.3f}  "
          f"floor_dram = max(0,{coeffs['g_dram']:.6e}*p + {coeffs['h_dram']:.3f})")
    print(f"L_ssd  = {coeffs['c_ssd']:.6e}*p + {coeffs['f_ssd']:.3f}  "
          f"floor_ssd  = max(0,{coeffs['g_ssd']:.6e}*p + {coeffs['h_ssd']:.3f})")
    if args.prefix is not None:
        d, s = predict_boundaries(args.prefix, coeffs)
        print(f"prefix={args.prefix}  dram_bound={d}  ssd_bound={s}")
    else:
        for p in (4096, 16384, 32768, 65536):
            d, s = predict_boundaries(p, coeffs)
            print(f"  p={p:6d}  dram={d:5d}  ssd={s:5d}")


if __name__ == "__main__":
    main()

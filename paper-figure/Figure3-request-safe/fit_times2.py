"""Fit the prefill time model (vllm/v1/core/kvpilot_time_model.py) to the
request-safe runs: each request's prefill e2e with its prefix in HBM [1],
DRAM [2], SSD [4], and the loaders' load / first-layer times.

    python3 fit_times2.py --out fit_times2_swe.json
    python3 fit_times2.py --points points.json   # reuse collect_points.py output

Steps, each on its own measurement:
  compute      relative least squares on the [1] e2e
  load         DRAM: load time vs tokens, suffix <= 256 (heavier compute
               stretches the copies, but then the load is hidden anyway);
               SSD: load time vs tokens
  first layer  first-layer time vs tokens, per tier
  overlap      k, F, omega, w0 by robust least squares on the [2] / [4] e2e

Points whose suffix spans more than one chunk are left out: their logged e2e
misses the last chunk (fixed in the scheduler on 2026-11-22, after these runs).
Prints the fit's errors per dataset, and a held-out check: fit on the first
dataset only, errors on the others.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from collect_points import collect

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "kvpilot_time_model",
    HERE.parents[1] / "vllm-dev/vllm/v1/core/kvpilot_time_model.py")
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)

BLOCK = 512
# The runs' scheduler: max_num_batched_tokens and prefill cudagraph sizes.
CHUNK = 8192
CAPTURE = [64, 128, 256, 512, 1024]
THRESH = 0.05
DATASETS = ("swe", "swe_reqprio", "swe_hbm1to2")


def fit(pts: list) -> dict:
    pts = [x for x in pts if x["s"] <= CHUNK]
    X = np.array([np.add(*tm.compute_features(x["p"], x["s"], CHUNK, CAPTURE))
                  for x in pts])
    y = np.array([x["t1"] for x in pts])
    comp = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)[0]
    p = np.array([x["p"] for x in pts], float)
    short = np.array([x["s"] <= 256 for x in pts])
    col = lambda key: np.array([x[key] for x in pts], float)  # noqa: E731
    c_dram, f_dram = np.polyfit(p[short], col("l2")[short], 1)
    c_ssd, f_ssd = np.polyfit(p, col("l4"), 1)
    g_dram, h_dram = np.polyfit(p, col("f2"), 1)
    g_ssd, h_ssd = np.polyfit(p, col("f4"), 1)
    k = {"version": tm.VERSION, "chunk": CHUNK, "capture": CAPTURE, "thresh": THRESH,
         **dict(zip(tm.COMPUTE, map(float, comp))),
         "c_dram": c_dram, "f_dram": f_dram, "c_ssd": c_ssd, "f_ssd": f_ssd,
         "g_dram": g_dram, "h_dram": h_dram, "g_ssd": g_ssd, "h_ssd": h_ssd,
         "k": 0.02, "F": 10.0, "omega": 0.1, "w0": 2.0}
    k = {a: (float(b) if isinstance(b, np.floating) else b) for a, b in k.items()}

    def resid(v):
        m = tm.TimeModel({**k, "k": v[0], "F": v[1], "omega": v[2], "w0": v[3]}, BLOCK)
        r = []
        for x in pts[::2]:
            r.append(m.e2e(x["p"], x["s"], x["p"], 0) / x["t2"] - 1)
            r.append(m.e2e(x["p"], x["s"], 0, x["p"]) / x["t4"] - 1)
        return r

    res = least_squares(resid, [0.02, 10.0, 0.1, 2.0],
                        bounds=([0, -50, 1e-4, 0.1], [1, 100, 1, 50]),
                        loss="soft_l1", f_scale=0.05)
    k.update(k=float(res.x[0]), F=float(res.x[1]), omega=float(res.x[2]),
             w0=float(res.x[3]), n=len(pts))
    return k


def report(k: dict, pts: list, names) -> None:
    m = tm.TimeModel(k, BLOCK)
    fmt = lambda e: (f"{np.median(np.abs(e))*100:4.1f}% / "  # noqa: E731
                     f"{np.percentile(np.abs(e), 90)*100:5.1f}% / {np.median(e)*100:+5.1f}%")
    print(f"{'':24s} |err| p50 / p90 / bias")
    for name in names:
        d = [x for x in pts if x["ds"] == name and x["s"] <= CHUNK]
        print(f"{name} (n={len(d)})")
        for tier, key, dram, ssd in (("HBM", "t1", 0, 0), ("DRAM", "t2", 1, 0),
                                     ("SSD", "t4", 0, 1)):
            y = np.array([x[key] for x in d])
            e = np.array([m.e2e(x["p"], x["s"], dram * x["p"], ssd * x["p"])
                          for x in d]) / y - 1
            print(f"  {tier:4s} all              {fmt(e)}")
            if key == "t1":
                continue
            u = np.array([x["l2" if dram else "l4"] / x["t1"] for x in d])
            for band, mask in (("compute-bound", u < .7), ("near", (u >= .7) & (u <= 1.3)),
                               ("load-bound", u > 1.3)):
                if mask.sum() >= 10:
                    print(f"       {band:13s} n={mask.sum():5d}  {fmt(e[mask])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--points", help="collect_points.py output instead of the raw logs")
    ap.add_argument("--out", default="fit_times2_swe.json")
    args = ap.parse_args()
    pts = json.loads(Path(args.points).read_text()) if args.points else collect(args.datasets)
    pts = [x for x in pts if x["ds"] in args.datasets]

    held = fit([x for x in pts if x["ds"] == args.datasets[0]])
    print(f"== held out: fit on {args.datasets[0]} only ==")
    report(held, pts, args.datasets[1:])

    k = fit(pts)
    print(f"\n== fit on {' '.join(args.datasets)} ==")
    report(k, pts, args.datasets)
    print("compute " + "  ".join(f"{a}={k[a]:.4g}" for a in tm.COMPUTE))
    print(f"load    DRAM {k['c_dram']*1024:.3f} ms/1K + {k['f_dram']:.2f}   "
          f"SSD {k['c_ssd']*1024:.3f} ms/1K + {k['f_ssd']:.2f}")
    print(f"first   DRAM {k['g_dram']*1024:.4f} ms/1K + {k['h_dram']:.2f}   "
          f"SSD {k['g_ssd']*1024:.3f} ms/1K + {k['h_ssd']:.2f}")
    print(f"overlap k={k['k']:.3f} F={k['F']:.1f} omega={k['omega']:.3f} w0={k['w0']:.2f}")
    (HERE / args.out).write_text(json.dumps(k, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

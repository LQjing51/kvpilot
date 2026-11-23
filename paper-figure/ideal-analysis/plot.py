"""Cost model: prefill compute time vs KV load time against prefix length, on
H20 x 8. Analytic, no measurement. Writes context_time.pdf.

    python3 plot.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

# H20 x 8, dense tensor-core peaks (FLOP/s) and bandwidths (B/s).
FLOPS = {"bf16": 1.184e15, "fp8": 2.368e15}
BW = {"HBM": 3.20e13, "DRAM": 5.04e11, "RNIC": 4.00e11}
# name, precision, a (FLOP/token^2), b (FLOP/token), KV bytes/token, max prefix
MODELS = [
    ("Qwen3-30B-A3B", "bf16", 786_432, 5_460_983_808, 98_304, 20_000, [0, 10_000, 20_000]),
    ("DeepSeek-V3.2-671B", "fp8", 999_424, 107_894_800_384, 43_188, 500_000, [0, 250_000, 500_000]),
]
SUFFIXES = [100, 200, 300]
LOAD_PATHS = ["DRAM", "RNIC", "HBM"]

# White halo so a label stays legible where it crosses another line.
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]


def label(ax, x, y, text, color, frac):
    """Text just above the line, at `frac` of the way along it, tilted to its slope."""
    i = int(frac * (len(x) - 1))
    p0, p1 = ax.transData.transform([(x[i - 1], y[i - 1]), (x[i + 1], y[i + 1])])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(text, (x[i], y[i]), xytext=(0, 1.5), textcoords="offset points",
                fontsize=5.5, color=color, ha="center", va="bottom",
                rotation=angle, rotation_mode="anchor", path_effects=HALO)


ms_fmt = FuncFormatter(lambda v, _: f"{v:g}")
k_fmt = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v else "0")

fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4, 1.55))
for ax, (name, prec, a, b, kv, xmax, xticks) in zip(axes, MODELS):
    x = np.linspace(0, xmax, 200)
    f = FLOPS[prec]
    curves = []
    print(name)
    for sf in SUFFIXES:
        y = (a * sf * (x + sf / 2) + b * sf) / f * 1e3
        ax.plot(x, y, color=COMPUTE, label="Compute")
        curves.append((y, f"suffix {sf}", COMPUTE, 0.48 if sf == 200 else 0.62))
        print(f"  compute suffix={sf}: {y[0]:.2f} -> {y[-1]:.2f} ms")
    for path in LOAD_PATHS:
        y = kv * x / BW[path] * 1e3
        ax.plot(x, y, color=LOAD, label="Load")
        curves.append((y, path, LOAD, 0.86))
        print(f"  load {path}: {y[-1]:.2f} ms at prefix {xmax}")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(k_fmt)
    ax.yaxis.set_major_formatter(ms_fmt)
    ax.locator_params(axis="y", nbins=4)
    ax.set_title(name, pad=2)
    for y, text, c, frac in curves:     # after limits are set, so the slope angle is final
        label(ax, x, y, text, c, frac)
axes[0].set_ylabel("Time (ms)")
fig.supxlabel("Prefix length", y=-0.06, fontsize=7.5)
fig.subplots_adjust(wspace=0.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend([h[1], h[3]], ["Compute", "Load"], loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, max(ax.get_position().y1 for ax in axes) + 0.1))
fig.savefig(HERE / "context_time.pdf")
print("wrote context_time.pdf")

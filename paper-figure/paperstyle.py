"""Shared matplotlib style for KVPilot paper figures.

Every paper-figure/<name>/plot.py does `from paperstyle import *` (the script
adds the repo's paper-figure/ dir to sys.path). Rationale for each choice is in
docs/evalution/figure-guidelines.md; change it there and here together.
"""
import matplotlib as mpl

COL = 3.33      # single-column width, inches (USENIX two-column)
FULL = 7.0      # full text width, inches
PANEL_H = 1.4   # typical panel height

# Categorical slots, fixed order, never cycled. Validated (light, on white):
# adjacent CVD dE >= 9.1, normal-vision >= 19.6. Slots 3, 4, 5 are below 3:1
# contrast on white, so a chart using them must direct-label or hatch.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
# Ours = slot 1 (blue), always solid. Baselines take the following slots, hatched.
OURS = BLUE
BASELINES = SERIES[1:]
HATCHES = ["//", "\\\\", "xx", "..", "--"]
MARKERS = ["o", "s", "^", "D", "v"]

# Meaning-fixed colors, used across every figure so a reader learns them once.
WAIT = "#e34948"      # KV load time, and GPU idle waiting on it (exposed load)
LOAD = WAIT
COMPUTE = BLUE        # prefill compute
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.linestyle": "--", "grid.linewidth": 0.4, "grid.color": GRID,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.0,
    "savefig.format": "pdf", "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def top_legend(fig, axes_or_handles=None, ncol=None, pad=0.02):
    """One shared, frameless, single-row legend just above the tallest panel."""
    if axes_or_handles is None:
        axes_or_handles = fig.axes
    y = max(ax.get_position().y1 for ax in fig.axes) + pad
    handles, labels = [], []
    for ax in axes_or_handles:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(labels))


def panel_label(ax, text):
    """'(a) ...' under a panel, for multi-panel figures without LaTeX subcaptions."""
    ax.set_title(text, y=-0.42, fontsize=7.5)

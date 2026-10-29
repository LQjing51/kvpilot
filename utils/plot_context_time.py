#!/usr/bin/env python3
"""
Context-time plot generator.

This version separates SYSTEM configuration (GPU FLOPs, bandwidths, HBM capacity)
from MODEL architecture constants (a_flops, b_flops, KV bytes per token).

Per-model timing constants are derived at runtime:
    a_us      = a_flops      / F_compute * 1e6
    b_us      = b_flops      / F_compute * 1e6
    c_hbm_us  = KV_per_token / BW_hbm    * 1e6
    c_dram_us = KV_per_token / BW_dram   * 1e6
    c_nic_us  = KV_per_token / BW_nic    * 1e6

F_compute is selected by the model's precision:
    "fp8"  -> SYSTEM.fp8_flops
    "fp16" -> SYSTEM.bf16_flops   (same tensor core path as BF16)
    "bf16" -> SYSTEM.bf16_flops

All SYSTEM FLOPs values are DENSE tensor-core peaks, not the "with sparsity"
numbers datasheets headline. Prefill is dense GEMM, so the sparse figures (2x)
are unreachable here and would halve every derived a/b constant.

Edit the SYSTEM block below (or override via env vars) to switch machines.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# ------------------------------ SYSTEM CONFIG (editable) ------------------------------
@dataclass
class SystemConfig:
    name: str = "DGX H200 x 8 GPU"

    # All FLOPs figures are 8-GPU aggregates of the vendor's DENSE tensor-core
    # peak. LLM prefill is dense GEMM and cannot use the 2:4 structured
    # sparsity path, so the sparse numbers printed on datasheets (2x these)
    # would overstate real compute.
    #
    # Per-GPU dense / sparse, from the official datasheets:
    #   H200 SXM  BF16  989 / 1,979    FP8 1,979 / 3,958
    #   A100 SXM  BF16  312 /   624    FP8 n/a (Ampere has no FP8 path)
    #   H20       BF16  148 /   296*   FP8   296 /   592
    # bf16_flops: float = 7.912e15 # H200 (989/GPU dense)
    bf16_flops: float = 1.184e15 # H20  (148/GPU dense)
    # bf16_flops: float = 2.496e15 # A100 (312/GPU dense)

    # fp8_flops:  float = 1.5832e16 # H200 (1979/GPU dense)
    fp8_flops:  float =  2.368e15 # H20 (296/GPU dense)
    # A100 has no FP8 tensor core; use bf16_flops for FP16/BF16 models instead.

    # bw_hbm:    float = 3.84e13 # H200
    bw_hbm:    float =  3.20e13 # H20
    # bw_hbm:    float =  1.548e13 # A100
    bw_nvlink: float = 3.60e12

    # bw_dram:   float = 5.04e11 # H200
    bw_dram:  float =  5.04e11  # H20
    # bw_dram:   float =  2.52e11  # A100

    bw_pcie:   float = 5.04e11
    bw_nic:    float = 4.00e11
    bw_ssd:    float = 6.40e10

    # hbm_capacity_gb: float = 1128.0
    hbm_capacity_gb: float =  768.0 # H20

    default_load_path: str = "dram"

    def f_compute(self, precision: str) -> float:
        p = precision.lower()
        if p == "fp8":
            return self.fp8_flops
        if p in ("fp16", "bf16"):
            return self.bf16_flops
        raise ValueError(f"Unsupported precision: {precision}")

    def bw(self, path: str) -> float:
        return {
            "hbm":    self.bw_hbm,
            "nvlink": self.bw_nvlink,
            "dram":   self.bw_dram,
            "pcie":   self.bw_pcie,
            "nic":    self.bw_nic,
            "ssd":    self.bw_ssd,
        }[path.lower()]


def _system_from_env(base: SystemConfig) -> SystemConfig:
    def _get(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    return SystemConfig(
        name=os.environ.get("SYS_NAME", base.name),
        bf16_flops=_get("SYS_BF16_FLOPS", base.bf16_flops),
        fp8_flops=_get("SYS_FP8_FLOPS", base.fp8_flops),
        bw_hbm=_get("SYS_BW_HBM", base.bw_hbm),
        bw_nvlink=_get("SYS_BW_NVLINK", base.bw_nvlink),
        bw_dram=_get("SYS_BW_DRAM", base.bw_dram),
        bw_pcie=_get("SYS_BW_PCIE", base.bw_pcie),
        bw_nic=_get("SYS_BW_NIC", base.bw_nic),
        bw_ssd=_get("SYS_BW_SSD", base.bw_ssd),
        hbm_capacity_gb=_get("SYS_HBM_GB", base.hbm_capacity_gb),
        default_load_path=os.environ.get("SYS_LOAD_PATH", base.default_load_path),
    )


SYSTEM = _system_from_env(SystemConfig())


# ------------------------------ MODEL CONFIG (editable) -------------------------------
@dataclass
class ModelSpec:
    name: str
    precision: str
    a_flops: float
    b_flops: float
    kv_bytes: float
    appends: List[float] = field(default_factory=lambda: [1000.0])
    x_min_b: float = 0.0
    x_max_b: float = 20 * 1024.0

# Sparse / compressed attention (DeepSeek V3.2 and V4) does not fit the dense
# reading of a and b, so the split is by GROWTH not by operator:
#   a_flops -> everything whose per-new-token cost still scales with prefix
#   b_flops -> everything fixed per new token
# For those models the main attention has a bounded top-k budget, so it lands in
# b and a carries only the selector (V3.2 indexer, V4 CSA indexer) plus V4's
# uncapped HCA layers. b therefore holds only once the budget is saturated:
#   V3.2 prefix > 2,048   V4-Pro prefix > 4,096   V4-Flash prefix > 2,048
# Below that, both models attend to the full context and a is understated.
# V4's per-token kv_bytes counts only the growing compressed cache; each layer
# also pins a fixed 128-token sliding window (Pro 3.8 MiB, Flash 2.7 MiB total).

# H200
# MODELS: List[ModelSpec] = [
#     ModelSpec("Qwen3-30B-A3B",          "bf16", 786_432,    5_460_983_808,    98_304,  [100.0, 500.0, 1000.0], 0, 20 * 1024.0),
#     ModelSpec("Llama-3-70B",            "fp16", 2_621_440,  136_902_082_560,  327_680, [100.0, 500.0, 1000.0], 0, 20 * 1024.0),
#     ModelSpec("MiniMax-M2-230B",        "fp8",  1_523_712,  80_354_476_032,   126_976, [100.0, 500.0, 1000.0], 0, 20 * 1024.0),
#     ModelSpec("DeepSeek-V3-671B",       "fp8",  16_990_208, 71_182_974_976,   35_136,  [50.0, 100.0, 200.0],   0, 20 * 1024.0),
#     ModelSpec("DeepSeek-V3.2-671B",     "fp8",  999_424,    107_894_800_384,  43_188,  [50.0, 100.0, 200.0],   0, 20 * 1024.0),
#     ModelSpec("DeepSeek-V4-Pro-1.6T",   "fp8",  186_368,    105_983_057_920,  4_444,   [50.0, 100.0, 200.0],   0, 20 * 1024.0),
#     ModelSpec("DeepSeek-V4-Flash-284B", "fp8",  106_496,    27_623_456_768,   3_104,   [50.0, 100.0, 200.0],   0, 20 * 1024.0),
#     ModelSpec("MiMo-V2-Flash-309B",     "fp8",  368_640,    28_595_716_096,   111_360, [100.0, 500.0, 1000.0], 0, 20 * 1024.0),
# ]

# H20
MODELS: List[ModelSpec] = [
    ModelSpec("Qwen3-30B-A3B",          "bf16", 786_432,    5_460_983_808,    98_304,  [100.0, 200.0, 500.0], 0, 20 * 1024.0),
    ModelSpec("MiMo-V2-Flash-309B",     "fp8",  368_640,    28_595_716_096,   111_360, [100.0, 200.0, 500.0], 0, 20 * 1024.0),
    # ModelSpec("Llama-3-70B",            "fp16", 2_621_440,  136_902_082_560,  327_680, [50.0, 100.0, 200.0], 0, 20 * 1024.0),
    # ModelSpec("MiniMax-M2-230B",        "fp8",  1_523_712,  80_354_476_032,   126_976, [50.0, 100.0, 200.0], 0, 20 * 1024.0),
    # ModelSpec("DeepSeek-V3-671B",       "fp8",  16_990_208, 71_182_974_976,   35_136,  [50.0, 100.0, 200.0], 0, 20 * 1024.0),
    ModelSpec("DeepSeek-V3.2-671B",     "fp8",  999_424,    107_894_800_384,  43_188,  [100.0, 200.0, 500.0], 0, 500 * 1024.0),
    # ModelSpec("DeepSeek-V4-Pro-1.6T",   "fp8",  186_368,    105_983_057_920,  4_444,   [50.0, 100.0, 200.0], 0, 20 * 1024.0),
    # ModelSpec("DeepSeek-V4-Flash-284B", "fp8",  106_496,    27_623_456_768,   3_104,   [50.0, 100.0, 200.0], 0, 20 * 1024.0),
]

@dataclass(frozen=True)
class ModelConstants:
    name: str
    a_us: float
    b_us: float
    c_hbm_us: float
    c_dram_us: float
    c_nic_us: float
    appends: Tuple[float, ...]
    x_min_b: float
    x_max_b: float


def derive_constants(model: ModelSpec, sys: SystemConfig) -> ModelConstants:
    F = sys.f_compute(model.precision)
    return ModelConstants(
        name=model.name,
        a_us=(model.a_flops / F) * 1e6,
        b_us=(model.b_flops / F) * 1e6,
        c_hbm_us=(model.kv_bytes / sys.bw_hbm) * 1e6,
        c_dram_us=(model.kv_bytes / sys.bw_dram) * 1e6,
        c_nic_us=(model.kv_bytes / sys.bw_nic) * 1e6,
        appends=tuple(model.appends),
        x_min_b=model.x_min_b,
        x_max_b=model.x_max_b,
    )


# ---------------------------------- PLOT CONSTANTS ------------------------------------
X_STEP_B = 1024.0
OUTPUT_PDF_NAME = "context_time.pdf"

SUBPLOT_WIDTH_PT = 520
SUBPLOT_HEIGHT_PT = 420
PAGE_MARGIN_L_PT = 36
PAGE_MARGIN_R_PT = 24
PAGE_MARGIN_B_PT = 28
SUBPLOT_GAP_PT = 18
TOP_AREA_H_PT = 12
MODEL_TITLE_GAP_PT = -15
CHART_MARGIN_L_PT = 64
CHART_MARGIN_R_PT = 18
CHART_MARGIN_T_PT = 34
CHART_MARGIN_B_PT = 52
# Axis title placement. Y is the distance from the subplot's left edge to the
# rotated baseline (larger = further right, toward the plot); X is the distance
# from the subplot's bottom edge up to the label baseline.
Y_LABEL_X_PT = 20
X_LABEL_Y_PT = 10
FONT_PT = 20.0
FORMULA_FONT_PT = 18.0
FORMULA_LINE_GAP_PT = 4.0
FORMULA_AREA_TOP_GAP_PT = 6.0
FORMULA_AREA_H_PT = 0.0


@dataclass(frozen=True)
class Series:
    name: str
    color_rgb: Tuple[float, float, float]
    ys: List[float]


def _frange(x_min: float, x_max: float, x_step: float) -> List[float]:
    if x_step <= 0:
        raise ValueError("x_step must be > 0")
    n = int(math.floor((x_max - x_min) / x_step)) + 1
    if n <= 0:
        return []
    xs = [x_min + i * x_step for i in range(n)]
    xs[-1] = min(xs[-1], x_max)
    return xs


def _nice_ticks(vmin: float, vmax: float, count: int) -> List[float]:
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or count <= 1:
        return []
    if vmax == vmin:
        return [vmin]
    span = vmax - vmin
    raw_step = span / (count - 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    for m in [1, 2, 5, 10]:
        step = m * mag
        if step >= raw_step:
            break
    start = math.floor(vmin / step) * step
    ticks: List[float] = []
    v = start
    for _ in range(count * 3):
        if v >= vmin - 1e-12 and v <= vmax + 1e-12:
            ticks.append(v)
        v += step
        if v > vmax + step:
            break
    if not ticks:
        ticks = [vmin, vmax]
    return ticks


def _format_si(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"{int(round(value / 1e9))}G"
    if abs_v >= 1e6:
        return f"{int(round(value / 1e6))}M"
    if abs_v >= 1e3:
        return f"{int(round(value / 1e3))}K"
    return str(int(round(value)))


def _pdf_escape_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _approx_text_width_pt(s: str, font_size: float) -> float:
    if not s:
        return 0.0
    ascii_count = sum(1 for ch in s if ord(ch) < 128)
    non_ascii_count = len(s) - ascii_count
    return ascii_count * font_size * 0.56 + non_ascii_count * font_size * 1.0


class _Pdf:
    def __init__(self) -> None:
        self._objects: List[bytes] = []

    def add_obj(self, data: bytes) -> int:
        self._objects.append(data)
        return len(self._objects)

    def build(self, root_obj_num: int) -> bytes:
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        offsets: List[int] = [0]
        out = bytearray(header)
        for i, obj in enumerate(self._objects, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("ascii"))
            out.extend(obj)
            if not obj.endswith(b"\n"):
                out.extend(b"\n")
            out.extend(b"endobj\n")
        xref_start = len(out)
        out.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(b"trailer\n")
        out.extend(f"<< /Size {len(offsets)} /Root {root_obj_num} 0 R >>\n".encode("ascii"))
        out.extend(b"startxref\n")
        out.extend(f"{xref_start}\n".encode("ascii"))
        out.extend(b"%%EOF\n")
        return bytes(out)


def _pdf_rgb_stroke(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r:.4f} {g:.4f} {b:.4f} RG"


def _prefill_color(idx: int, total: int) -> Tuple[float, float, float]:
    if total <= 1:
        t = 0.0
    else:
        t = idx / (total - 1)
    # dark red (0.55, 0.0, 0.0) -> light red/pink (1.0, 0.65, 0.65)
    r = 0.55 + (1.00 - 0.55) * t
    g = 0.00 + (0.65 - 0.00) * t
    b = 0.00 + (0.65 - 0.00) * t
    return (r, g, b)


def _prefill_name(append: float) -> str:
    return f"prefill_time (append={int(append)})"


_LOAD_PALETTE: List[Tuple[str, Tuple[float, float, float]]] = [
    ("load_time_hbm",  (0.20, 0.40, 0.85)),
    ("load_time_dram", (0.40, 0.60, 0.95)),
    ("load_time_nic",  (0.65, 0.80, 1.00)),
]


def _draw_subplot(content: List[str], *, xs: Sequence[float], series: Sequence[Series],
                  origin_x: float, origin_y: float, subplot_w: float, subplot_h: float,
                  draw_y_label: bool) -> None:
    plot_w = max(1.0, subplot_w - CHART_MARGIN_L_PT - CHART_MARGIN_R_PT)
    plot_h = max(1.0, subplot_h - CHART_MARGIN_T_PT - CHART_MARGIN_B_PT)
    y_values: List[float] = []
    for s in series:
        y_values.extend([y for y in s.ys if math.isfinite(y)])
    if not y_values:
        y_min, y_max = 0.0, 1.0
    else:
        y_data_max = max(y_values)
        y_min = 0.0
        y_max = y_data_max if y_data_max > 0 else 1.0
        y_max += 0.05 * (y_max - y_min)
    x_min = min(xs) if xs else 0.0
    x_max = max(xs) if xs else 1.0
    if x_max == x_min:
        x_max = x_min + 1.0
    x_min = min(x_min, 0.0)

    plot_left = origin_x + CHART_MARGIN_L_PT
    plot_right = plot_left + plot_w
    plot_bottom = origin_y + CHART_MARGIN_B_PT
    plot_top = plot_bottom + plot_h

    def x_to_pt(x: float) -> float:
        return plot_left + (x - x_min) * plot_w / (x_max - x_min)

    def y_to_pt(y: float) -> float:
        return plot_bottom + (y - y_min) * plot_h / (y_max - y_min)

    # gridlines first (inside the plot area), then axis labels, then the black border on top
    y_ticks = _nice_ticks(y_min, y_max, 6)
    x_ticks = _nice_ticks(x_min, x_max, 6)
    for yv in y_ticks:
        y = y_to_pt(yv)
        if y <= plot_bottom + 0.5 or y >= plot_top - 0.5:
            continue
        content.append("0.90 0.90 0.90 RG 2 w")
        content.append(f"{(plot_left + 1):.2f} {y:.2f} m {(plot_right - 1):.2f} {y:.2f} l S")
    for xv in x_ticks:
        x = x_to_pt(xv)
        if x <= plot_left + 0.5 or x >= plot_right - 0.5:
            continue
        content.append("0.95 0.95 0.95 RG 2 w")
        content.append(f"{x:.2f} {(plot_bottom + 1):.2f} m {x:.2f} {(plot_top - 1):.2f} l S")

    # axis tick labels (right-aligned for y so they don't run into the border)
    for yv in y_ticks:
        y = y_to_pt(yv)
        label = _format_si(yv)
        tw = _approx_text_width_pt(label, FONT_PT)
        content.append("0 0 0 rg")
        content.append(f"BT /F1 {FONT_PT:.0f} Tf")
        content.append(f"{(plot_left - 4 - tw):.2f} {(y - 3):.2f} Td ({_pdf_escape_text(label)}) Tj")
        content.append("ET")
    for xv in x_ticks:
        x = x_to_pt(xv)
        label = _format_si(xv)
        tw = _approx_text_width_pt(label, FONT_PT)
        content.append("0 0 0 rg")
        content.append(f"BT /F1 {FONT_PT:.0f} Tf")
        content.append(f"{(x - tw / 2.0):.2f} {(plot_bottom - 16):.2f} Td ({_pdf_escape_text(label)}) Tj")
        content.append("ET")

    # black border on top of gridlines
    content.append("0 0 0 RG 2 w")
    content.append(f"{plot_left:.2f} {plot_bottom:.2f} {plot_w:.2f} {plot_h:.2f} re S")

    load_display = {
        "load_time_hbm":  "HBM",
        "load_time_dram": "DRAM",
        "load_time_nic":  "RNIC",
    }
    prefill_label_specs: List[Tuple[float, float, str, Tuple[float, float, float]]] = []
    load_label_specs: List[Tuple[float, float, str, Tuple[float, float, float]]] = []
    for s in series:
        content.append(_pdf_rgb_stroke(s.color_rgb))
        content.append("2.4 w")
        pts = [(x_to_pt(x), y_to_pt(y)) for x, y in zip(xs, s.ys) if math.isfinite(y)]
        if not pts:
            continue
        x0, y0 = pts[0]
        content.append(f"{x0:.2f} {y0:.2f} m")
        for x, y in pts[1:]:
            content.append(f"{x:.2f} {y:.2f} l")
        content.append("S")
        if s.name.startswith("prefill_time") and len(pts) >= 2:
            label = s.name[len("prefill_time"):].strip()
            if label.startswith("(") and label.endswith(")"):
                label = label[1:-1]
            anchor_idx = max(1, int(len(pts) * 0.55))
            ax, ay = pts[anchor_idx]
            prefill_label_specs.append((ax, ay, label, s.color_rgb))
        elif s.name in load_display and len(pts) >= 2:
            anchor_idx = max(1, int(len(pts) * 0.9))
            ax, ay = pts[anchor_idx]
            load_label_specs.append((ax, ay, load_display[s.name], s.color_rgb))

    placed_boxes: List[Tuple[float, float, float, float]] = []
    line_h = FORMULA_FONT_PT + 2

    def _place_label_above(ax: float, ay: float, label: str, rgb: Tuple[float, float, float]) -> None:
        text_w = _approx_text_width_pt(label, FORMULA_FONT_PT)
        tx = max(plot_left + 2, min(ax - text_w / 2.0, plot_right - text_w - 2))
        ty = ay + 8
        for _ in range(40):
            box = (tx - 1, ty - 2, tx + text_w + 1, ty + FORMULA_FONT_PT)
            collide = any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in placed_boxes
            )
            if not collide and (ty + FORMULA_FONT_PT) <= (plot_top - 2):
                break
            ty += line_h
        ty = min(ty, plot_top - FORMULA_FONT_PT - 1)
        placed_boxes.append((tx - 1, ty - 2, tx + text_w + 1, ty + FORMULA_FONT_PT))
        r, g, b = rgb
        content.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        content.append(f"BT /F1 {FORMULA_FONT_PT:.1f} Tf")
        content.append(f"{tx:.2f} {ty:.2f} Td ({_pdf_escape_text(label)}) Tj")
        content.append("ET")

    # prefill labels above their anchor; stack upward to avoid overlap
    for spec in sorted(prefill_label_specs, key=lambda t: t[0]):
        _place_label_above(*spec)

    # load labels above their anchor; HBM is darkest/highest, place in reverse to keep order tidy
    for spec in sorted(load_label_specs, key=lambda t: t[1]):
        _place_label_above(*spec)

    x_label = "prefix length (tokens)"
    x_label_w = _approx_text_width_pt(x_label, FONT_PT)
    content.append("0 0 0 rg")
    content.append(f"BT /F1 {FONT_PT:.0f} Tf")
    content.append(f"{(plot_left + plot_w / 2.0 - x_label_w / 2.0):.2f} {(origin_y + X_LABEL_Y_PT):.2f} Td ({_pdf_escape_text(x_label)}) Tj")
    content.append("ET")
    if draw_y_label:
        # Rotated text grows upward from its baseline origin, so offset by half
        # the string width to centre it on the plot area (not the subplot box,
        # which includes the bottom margin holding the x-axis label).
        label = "comp/load time (us)"
        label_w = _approx_text_width_pt(label, FONT_PT)
        y_mid = plot_bottom + plot_h / 2.0
        content.append("q")
        content.append(f"0 1 -1 0 {(origin_x + Y_LABEL_X_PT):.2f} {(y_mid - label_w / 2.0):.2f} cm")
        content.append(f"BT /F1 {FONT_PT:.0f} Tf")
        content.append(f"0 0 Td ({_pdf_escape_text(label)}) Tj")
        content.append("ET")
        content.append("Q")


def _render_pdf_multi_model(model_plots: Sequence[Tuple[str, Sequence[float], Sequence[Series], Sequence[str]]]) -> bytes:
    n = len(model_plots)
    page_w = PAGE_MARGIN_L_PT + PAGE_MARGIN_R_PT + n * SUBPLOT_WIDTH_PT + (n - 1) * SUBPLOT_GAP_PT
    page_h = PAGE_MARGIN_B_PT + FORMULA_AREA_H_PT + SUBPLOT_HEIGHT_PT + TOP_AREA_H_PT
    content: List[str] = []
    content.append("1 1 1 rg 0 0 0 RG")
    content.append(f"0 0 {page_w:.2f} {page_h:.2f} re f")

    for i, (model_name, xs, series, formulas) in enumerate(model_plots):
        ox = PAGE_MARGIN_L_PT + i * (SUBPLOT_WIDTH_PT + SUBPLOT_GAP_PT)
        oy = PAGE_MARGIN_B_PT + FORMULA_AREA_H_PT
        model_text_w = _approx_text_width_pt(model_name, FONT_PT)
        model_x = ox + SUBPLOT_WIDTH_PT / 2.0 - model_text_w / 2.0
        model_y = oy + SUBPLOT_HEIGHT_PT + MODEL_TITLE_GAP_PT
        content.append("0 0 0 rg")
        content.append(f"BT /F1 {FONT_PT:.0f} Tf")
        content.append(f"{model_x:.2f} {model_y:.2f} Td ({_pdf_escape_text(model_name)}) Tj")
        content.append("ET")
        _draw_subplot(
            content, xs=xs, series=series,
            origin_x=ox, origin_y=oy,
            subplot_w=SUBPLOT_WIDTH_PT, subplot_h=SUBPLOT_HEIGHT_PT,
            draw_y_label=(i == 0),
        )
        f_top_y = PAGE_MARGIN_B_PT + FORMULA_AREA_H_PT - FORMULA_AREA_TOP_GAP_PT - FORMULA_FONT_PT
        for k, line in enumerate(formulas):
            line_y = f_top_y - k * (FORMULA_FONT_PT + FORMULA_LINE_GAP_PT)
            content.append("0 0 0 rg")
            content.append(f"BT /F1 {FORMULA_FONT_PT:.1f} Tf")
            content.append(f"{(ox + CHART_MARGIN_L_PT):.2f} {line_y:.2f} Td ({_pdf_escape_text(line)}) Tj")
            content.append("ET")

    stream_data = ("\n".join(content) + "\n").encode("latin-1")
    pdf = _Pdf()
    font_obj = pdf.add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    contents_obj = pdf.add_obj(
        b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"endstream"
    )
    page_obj = pdf.add_obj(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.2f} {page_h:.2f}] "
        f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {contents_obj} 0 R >>".encode("ascii")
    )
    pages_obj = pdf.add_obj(f"<< /Type /Pages /Kids [{page_obj} 0 R] /Count 1 >>".encode("ascii"))
    catalog_obj = pdf.add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("ascii"))
    return pdf.build(catalog_obj)


def _print_derived_table(sys: SystemConfig, specs: Sequence[ModelSpec]) -> None:
    print(f"System: {sys.name}")
    print(f"  BF16/FP16 FLOPs: {sys.bf16_flops:.4e}   FP8 FLOPs: {sys.fp8_flops:.4e}")
    print(f"  BW (B/s): HBM={sys.bw_hbm:.2e}  NVLink={sys.bw_nvlink:.2e}  DRAM={sys.bw_dram:.2e}  "
          f"PCIe={sys.bw_pcie:.2e}  NIC={sys.bw_nic:.2e}  SSD={sys.bw_ssd:.2e}")
    print(f"  HBM capacity: {sys.hbm_capacity_gb:.0f} GB")
    print("Derived per-model constants (us):")
    print(f"  {'Model':<24} {'prec':<5} {'a_us':>12} {'b_us':>12} {'c_hbm':>10} {'c_dram':>10} {'c_nic':>10}")
    for m in specs:
        c = derive_constants(m, sys)
        print(f"  {m.name:<24} {m.precision:<5} {c.a_us:>12.6g} {c.b_us:>12.6g} "
              f"{c.c_hbm_us:>10.6g} {c.c_dram_us:>10.6g} {c.c_nic_us:>10.6g}")


def main() -> int:
    if not MODELS:
        raise ValueError("MODELS is empty")

    if os.environ.get("DUMP_CONSTANTS", "0") == "1":
        _print_derived_table(SYSTEM, MODELS)

    if not any(spec.appends for spec in MODELS):
        raise ValueError("no append lengths configured on any model")
    load_palette = dict(_LOAD_PALETTE)

    model_plots: List[Tuple[str, Sequence[float], Sequence[Series], Sequence[str]]] = []
    for spec in MODELS:
        if not spec.appends:
            raise ValueError(f"{spec.name}: appends list is empty")
        c = derive_constants(spec, SYSTEM)
        xs = _frange(c.x_min_b, c.x_max_b, X_STEP_B)
        if not xs:
            raise ValueError(f"empty x range for {spec.name}")
        hbm  = [c.c_hbm_us  * x for x in xs]
        dram = [c.c_dram_us * x for x in xs]
        nic  = [c.c_nic_us  * x for x in xs]
        appends_sorted = sorted(spec.appends)
        series: List[Series] = []
        for i, ap in enumerate(appends_sorted):
            ys = [c.a_us * ap * (x + ap / 2.0) + c.b_us * ap for x in xs]
            series.append(Series(
                name=_prefill_name(ap),
                color_rgb=_prefill_color(i, len(appends_sorted)),
                ys=ys,
            ))
        series.extend([
            Series(name="load_time_hbm",  color_rgb=load_palette["load_time_hbm"],  ys=hbm),
            Series(name="load_time_dram", color_rgb=load_palette["load_time_dram"], ys=dram),
            Series(name="load_time_nic",  color_rgb=load_palette["load_time_nic"],  ys=nic),
        ])
        model_plots.append((spec.name, xs, series, []))

    pdf_bytes = _render_pdf_multi_model(model_plots=model_plots)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_PDF_NAME)
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
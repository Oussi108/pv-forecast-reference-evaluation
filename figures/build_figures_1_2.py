"""Figures 1-2 of the paper (study workflow and model paths), drawn with matplotlib (no AI-generated images).

Every number shown (seeds 11/23/37/51/73; <=1200 rounds, 31 leaves; patience 60 rounds / 6 epochs; width 32, one layer;
four attention heads; 23.4 kW dual-axis; 4.7 MW single-axis) is checked against the manuscript text.
The input insets are real hourly data from NREL PVDAQ system 9068 in 2021 (development year; CC BY 4.0): 24 completed
hours ending 2021-09-24 14:00 local time, every channel min-max scaled for display only.
Outputs (PDF, SVG, 600 dpi PNG): manuscript versions Figure1_study_workflow_v44 and Figure2_model_paths_v44, standalone
versions Figure1_study_workflow_v5style and Figure2_model_paths_v5style, and all_figures.pdf.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1] / 'results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pv_paths import PRIV_PVDAQ  # noqa: E402
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "mathtext.fontset": "stix", "text.color": "#20242b",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
                     "savefig.facecolor": "white"})
INK, MUTED, BORDER = "#20242b", "#565d66", "#929aa5"
BLUE, BLUE_F = "#6080ab", "#e5edf7"
GREEN, GREEN_F = "#638a78", "#e5f0e9"
PINK, PINK_F = "#b78197", "#f4e7ed"
PURPLE, PURPLE_F = "#655395", "#ece7f5"
ORANGE, ORANGE_F = "#bb9062", "#faf0e2"
RUST, RUST_F = "#b0644e", "#f6e6df"
GREY, GREY_F = "#b9bec6", "#f6f7f9"
SRC = PRIV_PVDAQ / "prepared/PVDAQ9068/pvdaq_9068.pkl"
END = pd.Timestamp("2021-09-24 14:00", tz="America/Denver")
CREDIT = "Example series: NREL PVDAQ system 9068 (CC BY 4.0), humidity NOAA ISD Greeley, 2021"

# ------------------------------------------------------------------ real 2021 example window (display only)
_d = pd.read_pickle(SRC)
_w = _d.loc[END - pd.Timedelta(hours=23):END]
assert len(_w) == 24 and _w.index.max().year == 2021
CH = [("power", "power", BLUE), ("irradiance", "ghi", ORANGE), ("temp.", "temperature", RUST), ("RH", "humidity", GREEN),
      ("clear-sky", "clear_ghi", PURPLE)]
SERIES = {}
for name, col, _ in CH:
    v = _w[col].to_numpy(float)
    v = np.nan_to_num(v, nan=np.nanmin(v))
    SERIES[name] = (v - v.min()) / max(v.max() - v.min(), 1e-9)


# ------------------------------------------------------------------ drawing primitives
def canvas(w, h, y0=0.0):
    fig = plt.figure(figsize=(w, h - y0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(y0, h); ax.set_aspect("equal"); ax.axis("off")
    ax._boxes = []
    return fig, ax


def label(ax, x, y, text, size=8, weight="normal", color=INK, ha="center", va="center", italic=False, z=6):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
                   fontstyle="italic" if italic else "normal", linespacing=1.15, zorder=z)


def box(ax, x, y, w, h, fill=GREY_F, edge=BORDER, lw=.7, r=.05, dashed=False, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fill, edgecolor=edge,
                                linewidth=lw, linestyle=(0, (4, 3)) if dashed else "-", zorder=z))


def boxed_label(ax, rect, text, size=8, weight="normal", color=INK):
    x, y, w, h = rect
    t = label(ax, x + w / 2, y + h / 2, text, size, weight, color); ax._boxes.append((t, rect))
    return t


def group(ax, rect, title, edge=BORDER, size=9.2, letter=None):
    x, y, w, h = rect
    box(ax, x, y, w, h, "white", edge, lw=.8, r=.07, dashed=True, z=0)
    t = label(ax, x + w / 2, y + h - .17, f"({letter}) {title}" if letter else title, size, weight="bold")
    ax._boxes.append((t, (x + .03, y + h - .34, w - .06, .34)))


def arrow(ax, p0, p1, color=INK, lw=.85, ms=7, z=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, shrinkA=0, shrinkB=0, color=color,
                                 linewidth=lw, zorder=z))


def trace(ax, x, y, w, h, v, color, lw=.85):
    u = np.linspace(0, 1, len(v))
    ax.plot(x + w * u, y + h * (.08 + .84 * v), color=color, lw=lw, zorder=4, solid_capstyle="round", solid_joinstyle="round")


def real_stack(ax, x, y, w, row_h, labels=True, size=7.2):
    for i, (name, _, color) in enumerate(CH):
        yy = y - i * row_h
        ax.plot([x, x + w], [yy, yy], color=GREY, lw=.4, zorder=3)
        trace(ax, x, yy, w, row_h * .86, SERIES[name], color)
        if labels:
            label(ax, x - .05, yy + row_h * .38, name, size, ha="right", color=MUTED)


def pv_glyph(ax, x, y, w, h, color=BLUE, n=2):
    hh = np.linspace(0, 24, 121)
    for i in range(n):
        s = np.where((hh > 6.2 + .4 * i) & (hh < 17.8 - .4 * i), np.sin(np.pi * (hh - 6.2 - .4 * i) / (11.6 - .8 * i)), 0) ** 1.5
        ax.plot(x + w * hh / 24, y + h * (.05 + .85 * s * (1 - .15 * i)), color=color, lw=.8 - .15 * i, zorder=4)
    ax.plot([x, x + w], [y + .02, y + .02], color=GREY, lw=.4, zorder=3)


def ref_glyph(ax, i, x, y, w, h, c=PURPLE):
    """Small icons for the five fit-free references (schematic)."""
    hh = np.linspace(0, 1, 60)
    bell = np.where((hh > .15) & (hh < .85), np.sin(np.pi * (hh - .15) / .7), 0) ** 1.5
    ax.plot([x, x + w], [y + .01, y + .01], color=GREY, lw=.4, zorder=3)
    if i == 0:                                    # persistence: last value held
        ax.plot([x, x + w * .45, x + w * .45, x + w], [y + h * .2, y + h * .2, y + h * .7, y + h * .7], color=c, lw=.8, zorder=4)
    elif i == 1:                                  # previous day: yesterday's curve
        ax.plot(x + w * hh, y + h * (.05 + .85 * bell), color=c, lw=.8, zorder=4)
    elif i == 2:                                  # hour climatology: hourly bars
        for k, v in enumerate((.25, .6, .9, .6, .25)):
            ax.add_patch(Rectangle((x + w * (.12 + .16 * k), y + .01), w * .1, h * .85 * v, facecolor=c, edgecolor="none", alpha=.7, zorder=4))
    elif i == 3:                                  # smart persistence: clear-sky curve scaled by a ratio
        ax.plot(x + w * hh, y + h * (.05 + .85 * bell), color=c, lw=.6, ls=(0, (2, 1.2)), zorder=4)
        ax.plot(x + w * hh, y + h * (.05 + .55 * bell), color=c, lw=.85, zorder=4)
    else:                                         # convex combination of previous day and climatology
        ax.plot(x + w * hh, y + h * (.05 + .85 * bell), color=c, lw=.5, alpha=.45, zorder=4)
        ax.plot(x + w * hh, y + h * (.05 + .45 * np.ones_like(hh) * (bell > 0)), color=c, lw=.5, alpha=.45, zorder=4)
        ax.plot(x + w * hh, y + h * (.05 + .65 * bell), color=c, lw=.85, zorder=4)


def sun(ax, cx, cy, r=.05, c=ORANGE):
    ax.add_patch(Circle((cx, cy), r, facecolor=c, edgecolor="none", zorder=5))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([cx + 1.6 * r * np.cos(a), cx + 2.3 * r * np.cos(a)], [cy + 1.6 * r * np.sin(a), cy + 2.3 * r * np.sin(a)],
                color=c, lw=.7, zorder=5)


def tree_glyph(ax, cx, cy, s=.16, c=BLUE, fill=BLUE_F):
    pts = {0: (cx, cy + s * .55), 1: (cx - s * .45, cy), 2: (cx + s * .45, cy), 3: (cx - s * .7, cy - s * .55),
           4: (cx - s * .2, cy - s * .55), 5: (cx + s * .2, cy - s * .55), 6: (cx + s * .7, cy - s * .55)}
    for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=c, lw=.6, zorder=4)
    for k, p in pts.items():
        ax.add_patch(Circle(p, s * .1, facecolor=c if k < 3 else fill, edgecolor=c, lw=.55, zorder=5))


def cube(ax, x, y, w=.17, h=.16, d=.06, edge=RUST, fill="#e9c3b6"):
    rgb = np.array(to_rgb(fill))
    face = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)]
    side = [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)]
    for pts, fc in ((top, np.clip(rgb + .07, 0, 1)), (side, np.clip(rgb - .05, 0, 1)), (face, rgb)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=edge, lw=.55, zorder=4))


def plane_stack(ax, x, y, w=.20, h=.30, depth=4, edge=GREEN, fill="#bcd6c8", dx=.10, dy=.07):
    rgb = np.array(to_rgb(fill))
    for i in range(depth - 1, -1, -1):
        o = .03 * i
        pts = [(x + o, y + o), (x + w + o, y + dy + o), (x + w + dx + o, y + h + dy + o), (x + dx + o, y + h + o)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=tuple(np.clip(rgb + (depth - 1 - i) * .02, 0, 1)), edgecolor=edge,
                             lw=.55, zorder=4 + (depth - i) * .01))


def tokens(ax, x, y, w, h, n, color, fill, varied=None):
    gap = .015; ww = (w - (n - 1) * gap) / n
    for i in range(n):
        f = fill if varied is None else tuple(np.clip(np.array(to_rgb(fill)) - varied[i] * .25, 0, 1))
        box(ax, x + i * (ww + gap), y, ww, h, f, color, lw=.45, r=.008, z=4)


def operator(ax, x, y, s, r=.075):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=.8, zorder=6))
    label(ax, x, y + .004, s, 9, z=7)


def dots(ax, x0, y, gap, labels, dark, color, r=.035, size=7.2):
    for i, lab in enumerate(labels):
        xx = x0 + i * gap
        if i:
            ax.plot([xx - gap, xx], [y, y], color=GREY, lw=.6, zorder=3)
        ax.add_patch(Circle((xx, y), r, facecolor=INK if i == dark else color, edgecolor="none", zorder=5))
        label(ax, xx, y - .11, lab, size, color=MUTED)


def chip(ax, x, y, text, color, fill, size=7.2, w=None):
    w = w or .055 * len(text) + .12
    box(ax, x, y, w, .17, fill, color, lw=.55, r=.04, z=3)
    boxed_label(ax, (x, y, w, .17), text, size)
    return x + w


def save(fig, ax, stem, title, records):
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); errs = []
    for t, (x, y, w, h) in ax._boxes:
        b = t.get_window_extent(rr); lo = ax.transData.transform((x + .015, y + .01)); hi = ax.transData.transform((x + w - .015, y + h - .01))
        if b.x0 < lo[0] or b.x1 > hi[0] or b.y0 < lo[1] or b.y1 > hi[1]:
            errs.append(t.get_text())
    for t in ax.texts:
        b = t.get_window_extent(rr)
        if b.x0 < 0 or b.x1 > fig.bbox.width or b.y0 < 0 or b.y1 > fig.bbox.height:
            errs.append("outside figure: " + t.get_text())
    if errs:
        raise ValueError(f"{stem}: text outside its box: {errs}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, metadata={"Title": title} if ext == "pdf" else None)
    records.append({"stem": stem, "width_in": fig.get_figwidth(), "height_in": fig.get_figheight(), "text_containment": "passed"})
    return fig


# ================================================================== Figure 1: study workflow
def figure1(records, ms=False):
    """ms=True: manuscript version (no in-figure title or credit; the caption carries both)."""
    fig, ax = canvas(7.2, 4.32 if ms else 4.62, .17 if ms else 0)
    if not ms:
        label(ax, .18, 4.47, "Study workflow", 10, weight="bold", ha="left")
    G = {"obs": (.12, 1.02, 1.70, 3.25), "info": (1.98, 1.02, 1.82, 3.25), "alt": (3.96, 1.02, 1.98, 3.25), "ver": (6.10, 1.02, 1.02, 3.25)}
    group(ax, G["obs"], "Observations", BLUE, letter="a")
    group(ax, G["info"], "Information at issue time t", ORANGE, 9.0, letter="b")
    group(ax, G["alt"], "Forecast alternatives", PURPLE, letter="c")
    group(ax, G["ver"], "Verification", GREEN, letter="d")
    # observations: three testbeds
    cards = [("DKASC Array 1B", "Alice Springs, AU\n23.4 kW, dual-axis", None),
             ("HKUST campus", "Hong Kong, CN\n31 rooftop systems", None),
             ("PVDAQ 9068", "Colorado, US\n4.7 MW, single-axis", "pre-specified protocol")]
    for i, (name, sub, extra) in enumerate(cards):
        y = 3.27 - i * .70
        box(ax, .22, y, 1.50, .60, BLUE_F, BLUE, lw=.55, r=.04, z=2)
        pv_glyph(ax, .28, y + .14, .34, .32, BLUE, n=2 if i != 1 else 3)
        label(ax, .70, y + .45, name, 8, weight="bold", ha="left")
        label(ax, .70, y + .22 if extra is None else y + .25, sub, 6.9, ha="left", color=MUTED)
        if extra:
            label(ax, .70, y + .055, extra, 6.6, ha="left", color=PINK, italic=True)
    label(ax, .97, 1.16, "scored as separate testbeds", 7, italic=True, color=MUTED)
    # information at issue time: real 24 h x 5 channels
    label(ax, 2.89, 3.86, "24 completed hours × 5 channels", 7.4, color=MUTED)
    real_stack(ax, 2.62, 3.50, 1.08, .26)
    ax.plot([2.62, 3.70], [2.30, 2.30], color=INK, lw=.6, zorder=4); arrow(ax, (3.62, 2.30), (3.72, 2.30), ms=6)
    label(ax, 2.62, 2.18, "t − 24 h", 7, color=MUTED); label(ax, 3.70, 2.18, "t", 7.4, weight="bold")
    box(ax, 2.06, 1.33, 1.66, .68, ORANGE_F, ORANGE, lw=.55, r=.04, z=2)
    sun(ax, 2.20, 1.80, .035)
    label(ax, 2.30, 1.80, "Known at target time t + h", 7.6, weight="bold", ha="left")
    label(ax, 2.89, 1.53, "clear-sky GHI, sin/cos hour,\nsin/cos day of year", 6.9, color=MUTED)
    label(ax, 2.89, 1.16, "normalized power · no future weather", 7, italic=True, color=MUTED)
    # forecast alternatives
    box(ax, 4.05, 2.70, 1.80, 1.20, "white", PURPLE, lw=.55, r=.04, z=1)
    label(ax, 4.13, 3.76, "Fit-free references", 7.8, weight="bold", ha="left")
    refs = ["Persistence", "Previous day", "Hour climatology", "Smart persistence (2022-specified)", "Convex previous day–climatology"]
    for i, r_ in enumerate(refs):
        yy = 3.56 - i * .19
        ref_glyph(ax, i, 4.13, yy - .07, .18, .14)
        label(ax, 4.38, yy, r_, 6.9, ha="left")
    box(ax, 4.05, 1.12, 1.80, 1.48, "white", PURPLE, lw=.55, r=.04, z=1)
    label(ax, 4.13, 2.46, "Learned models (per testbed and lead)", 7.1, weight="bold", ha="left")
    tree_glyph(ax, 4.35, 2.10, .17, BLUE, BLUE_F); label(ax, 4.35, 1.88, "LightGBM", 6.9)
    for k in range(3):
        cube(ax, 4.72 + k * .12, 2.02, .09, .09, .035)
    label(ax, 4.87, 1.88, "GRU", 6.9)
    plane_stack(ax, 5.33, 1.98, .16, .17, 3, GREEN, "#bcd6c8", .06, .04); label(ax, 5.45, 1.88, "Transformer", 6.9)
    label(ax, 4.13, 1.66, "training policy", 6.9, ha="left", color=MUTED)
    x = chip(ax, 4.12, 1.40, "frozen 2021", PURPLE, PURPLE_F, 6.5, .58)
    x = chip(ax, x + .04, 1.40, "annual", PURPLE, PURPLE_F, 6.5, .36)
    chip(ax, x + .04, 1.40, "monthly (365 d)", PURPLE, PURPLE_F, 6.5, .64)
    label(ax, 4.95, 1.25, "five seeds · validation-selected epochs/rounds", 6.5, italic=True, color=MUTED)
    # verification
    box(ax, 6.17, 2.78, .88, 1.10, GREEN_F, GREEN, lw=.55, r=.04, z=1)
    label(ax, 6.24, 3.75, "Point", 7.8, weight="bold", ha="left")
    dots(ax, 6.30, 3.52, .20, ["2 h", "4 h", "6 h", "12 h"], 2, GREEN, .03, 6.6)
    label(ax, 6.61, 3.05, "daylight MAE;\nskill vs a named\nreference", 6.6, color=MUTED)
    box(ax, 6.17, 1.32, .88, 1.36, GREEN_F, GREEN, lw=.55, r=.04, z=1)
    label(ax, 6.24, 2.55, "Interval (6 h)", 7.8, weight="bold", ha="left")
    for i, (nm, w) in enumerate((("raw", .16), ("day/night", .30), ("grouped", .34))):
        yy = 2.33 - i * .17
        label(ax, 6.56, yy, nm, 6.5, ha="right", color=MUTED)
        ax.plot([6.80 - w / 2, 6.80 + w / 2], [yy, yy], color=GREEN if i else GREY, lw=2.2, solid_capstyle="butt", zorder=4)
    ax.plot([6.80, 6.80], [1.92, 2.43], color=INK, lw=.5, ls=(0, (2, 1.5)), zorder=5)
    label(ax, 6.61, 1.60, "coverage, width,\ninterval score;\n80% and 90%", 6.6, color=MUTED)
    label(ax, 6.61, 1.16, "7-day blocks · Holm", 6.7, italic=True, color=MUTED)
    for x0, x1 in ((1.82, 1.98), (3.80, 3.96), (5.94, 6.10)):
        arrow(ax, (x0 + .01, 2.65), (x1 - .01, 2.65), ms=7)
    # chronology
    label(ax, .14, .86, "Chronology", 8.4, weight="bold", ha="left")
    segs = [("2021 · development and model selection", 1.70, BLUE, BLUE_F), ("2022 · smart-reference choice, calibration", 1.78, ORANGE, ORANGE_F),
            ("2023 · evaluation\nretrospective: DKASC, HKUST\npre-specified: PVDAQ", 2.12, PURPLE, PURPLE_F),
            ("2024 · confirmation (PVDAQ),\nsparse check (DKASC)", 1.40, GREEN, GREEN_F)]
    x = .12
    for text, w, c, f in segs:
        box(ax, x, .24, w - .03, .48, f, c, lw=.65, r=.03, z=1)
        boxed_label(ax, (x, .24, w - .03, .48), text, 6.9)
        x += w
    if ms:
        return save(fig, ax, "Figure1_study_workflow_v44", "Figure 1: study workflow", records)
    label(ax, 7.12, .08, CREDIT, 6.2, ha="right", italic=True, color=MUTED)
    return save(fig, ax, "Figure1_study_workflow_v5style", "Study workflow (standalone version)", records)


# ================================================================== Figure 2: model paths
def out_box(ax, x, y, color, p_dark, q_text, q_dots):
    box(ax, x, y, 1.20, .76, "white", BORDER, lw=.55, r=.04, z=2)
    label(ax, x + .09, y + .60, "P", 7.6, weight="bold")
    dots(ax, x + .27, y + .60, .26, ["2 h", "4 h", "6 h", "12 h"], -1, color, .032, 6.4)
    label(ax, x + .09, y + .22, "Q", 7.6, weight="bold")
    if q_dots:
        for i in range(7):
            xx = x + .25 + i * .13
            ax.add_patch(Circle((xx, y + .22), .022 if i != 3 else .034, facecolor=color, edgecolor="none", alpha=.45 if i != 3 else 1, zorder=5))
        ax.plot([x + .25, x + 1.03], [y + .22, y + .22], color=GREY, lw=.5, zorder=4)
        label(ax, x + .64, y + .08, q_text, 6.4, color=MUTED)
    else:
        label(ax, x + .64, y + .22, q_text, 6.6, italic=True, color=MUTED)


def figure2(records, ms=False):
    """ms=True: manuscript version (no in-figure title or credit; the caption carries both)."""
    fig, ax = canvas(7.2, 4.45 if ms else 4.70, .72 if ms else .55)
    if not ms:
        label(ax, .18, 4.55, "Model paths", 10, weight="bold", ha="left")
    group(ax, (.12, .78, 1.52, 3.62), "Shared input", BORDER, letter="a")
    label(ax, .88, 3.98, "history x  ·  24 h × 5 channels", 7.2, color=MUTED)
    real_stack(ax, .72, 3.60, .84, .25, labels=True, size=6.6)
    box(ax, .22, 1.48, 1.32, .88, ORANGE_F, ORANGE, lw=.55, r=.04, z=2)
    sun(ax, .36, 2.18, .035); label(ax, .46, 2.18, "target-time f", 7.6, weight="bold", ha="left")
    label(ax, .88, 1.80, "5 known values: clear-sky\nGHI and sin/cos of hour\nand day of year", 6.7, color=MUTED)
    label(ax, .88, 1.08, "fitted separately for each\ntestbed and lead time;\nno system identifier", 6.6, italic=True, color=MUTED)
    lanes = [(3.30, "(b) LightGBM", BLUE, BLUE_F), (2.17, "(c) GRU", RUST, RUST_F), (1.04, "(d) Transformer", GREEN, GREEN_F)]
    for y, name, c, f in lanes:
        box(ax, 1.84, y, 3.90, 1.02, "white", c, lw=.8, r=.05, dashed=True, z=0)
        ax.add_patch(Rectangle((1.84, y + .05), .045, .92, facecolor=c, edgecolor="none", zorder=1))
        label(ax, 1.96, y + .87, name, 8.6, weight="bold", ha="left")
        arrow(ax, (1.65, y + .45), (1.84, y + .45), ms=7)
        arrow(ax, (4.42, y + .45), (4.50, y + .45), ms=6)
    # LightGBM lane
    y = 3.30
    tokens(ax, 1.98, y + .30, .70, .26, 10, BLUE, BLUE_F, varied=np.abs(np.sin(np.arange(10))) * .5)
    label(ax, 2.33, y + .17, "flatten 120 + 5 features", 6.5, color=MUTED)
    arrow(ax, (2.72, y + .43), (2.88, y + .43), ms=6)
    for k in range(3):
        tree_glyph(ax, 3.04 + k * .30, y + .46, .16, BLUE, BLUE_F)
    label(ax, 4.16, y + .46, "+ … +", 7.4)
    label(ax, 2.86, y + .87, "boosted trees, ≤1200 rounds, 31 leaves", 6.6, ha="left", color=MUTED)
    out_box(ax, 4.50, y + .08, BLUE, -1, "7 quantiles, 6 h only", True)
    # GRU lane
    y = 2.17
    for k in range(5):
        cube(ax, 1.98 + k * .25, y + .36, .16, .15, .05, RUST, "#e9c3b6")
        if k < 4:
            arrow(ax, (2.15 + k * .25, y + .435), (2.22 + k * .25, y + .435), RUST, .6, 5)
    for k, t in enumerate(["t−23", "t−22", "…", "t−1", "t"]):
        label(ax, 2.06 + k * .25, y + .26, t, 6.2, color=MUTED)
    label(ax, 2.52, y + .87, "recurrent encoding, width 32, one layer", 6.6, ha="left", color=MUTED)
    arrow(ax, (3.27, y + .44), (3.37, y + .44), ms=6)
    box(ax, 3.38, y + .30, .34, .28, RUST_F, RUST, lw=.6, r=.03, z=3); boxed_label(ax, (3.38, y + .30, .34, .28), "last\nstate", 6.4)
    operator(ax, 3.86, y + .44, "+"); label(ax, 3.98, y + .44, "f", 7.4, italic=True)
    box(ax, 4.06, y + .30, .34, .28, GREY_F, BORDER, lw=.6, r=.03, z=3); boxed_label(ax, (4.06, y + .30, .34, .28), "head\n32", 6.4)
    out_box(ax, 4.50, y + .08, RUST, -1, "no quantile model", False)
    # Transformer lane
    y = 1.04
    tokens(ax, 1.98, y + .30, .62, .26, 8, GREEN, GREEN_F, varied=np.linspace(0, .5, 8))
    label(ax, 2.36, y + .17, "tokens + position code", 6.5, color=MUTED)
    arrow(ax, (2.64, y + .43), (2.76, y + .43), ms=6)
    plane_stack(ax, 2.80, y + .28, .22, .26, 4, GREEN, "#bcd6c8", .08, .05)
    label(ax, 3.00, y + .87, "self-attention, 4 heads · FF 64", 6.6, ha="left", color=MUTED)
    arrow(ax, (3.24, y + .44), (3.37, y + .44), ms=6)
    box(ax, 3.38, y + .30, .34, .28, GREEN_F, GREEN, lw=.6, r=.03, z=3); boxed_label(ax, (3.38, y + .30, .34, .28), "last\ntoken", 6.4)
    operator(ax, 3.86, y + .44, "+"); label(ax, 3.98, y + .44, "f", 7.4, italic=True)
    box(ax, 4.06, y + .30, .34, .28, GREY_F, BORDER, lw=.6, r=.03, z=3); boxed_label(ax, (4.06, y + .30, .34, .28), "head\n32", 6.4)
    out_box(ax, 4.50, y + .08, GREEN, -1, "7 quantiles, 6 h only", True)
    # training group
    group(ax, (5.92, .78, 1.20, 3.62), "Training", BORDER, letter="e")
    items = [("Loss", "point: absolute error\nquantile: pinball loss"),
             ("Selection", "early stopping on\nOct–Dec 2021; patience\n60 rounds (LightGBM),\n6 epochs (neural)"),
             ("Updating", "frozen 2021, annual,\nmonthly (trailing 365 d);\nepochs and rounds\ninherited, no test tuning"),
             ("Seeds", "11, 23, 37, 51, 73;\nlosses averaged per seed\nbefore equal-system\naggregation")]
    yy = 3.92
    for head, body in items:
        label(ax, 6.00, yy, head, 7.8, weight="bold", ha="left")
        n = body.count("\n") + 1
        label(ax, 6.00, yy - .10 - .065 * n, body, 6.5, ha="left", color=MUTED)
        yy -= .28 + .13 * n
    if ms:
        return save(fig, ax, "Figure2_model_paths_v44", "Figure 2: model paths", records)
    label(ax, 7.12, .64, CREDIT, 6.2, ha="right", italic=True, color=MUTED)
    return save(fig, ax, "Figure2_model_paths_v5style", "Model paths (standalone version)", records)


if __name__ == "__main__":
    REC = []
    figs = [figure1(REC), figure2(REC)]
    with PdfPages(OUT / "all_figures.pdf") as pdf:
        for f in figs:
            pdf.savefig(f)
    figure1(REC, ms=True); figure2(REC, ms=True)       # manuscript versions used in v44
    (OUT / "validation.json").write_text(json.dumps({"figures": REC, "example_series": {
        "source": "NREL PVDAQ system 9068 (CC BY 4.0) power, plane-of-array irradiance and temperature; relative humidity from "
                  "NOAA ISD Greeley 724768-24051; clear-sky from pvlib; file PV_PRIVATE_ROOT/pvdaq9068/prepared/PVDAQ9068/pvdaq_9068.pkl",
        "window_local": [str(_w.index.min()), str(_w.index.max())], "year": 2021, "use": "display only, min-max scaled per channel"},
        "content_source": "v43 Figures 1-2, checked against the v43 manuscript text; no numbers changed"}, indent=1), encoding="utf-8")
    print("built", [r["stem"] for r in REC], "->", OUT)

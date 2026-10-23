"""JRSE versions of Figures 3-6, redrawn from the released aggregate results (no new numbers).

Same data and marks as paper1_code_release/figures/f02_result_figures.py (Figs 3-5) and f03_figure6.py (Fig 6); changed
only for AIP Publishing figure rules: width 170 mm (two-column maximum 6.69 in), all text >= 8 pt at that size, panel
labels "(a)", and the testbed label "Array 1B" (as in the text) instead of "DKASC".
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REL = Path(r'E:/master in china/s2/research/solar energy forecasting/paper1_code_release')
RES = REL / 'results/reference_and_update_inference'
RES9 = REL / 'results/pvdaq9068_evaluation'
OUT = Path(__file__).resolve().parents[1] / 'production'
OUT.mkdir(parents=True, exist_ok=True)
FS = 8.0                                   # minimum text size (pt) at printed size
plt.rcParams.update({'font.family': 'Arial', 'font.size': FS, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.labelsize': FS,
                     'ytick.labelsize': FS, 'legend.fontsize': FS, 'axes.labelsize': FS, 'axes.titlesize': FS + 0.5,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})
INK = '#1f2a33'; MUTED = '#5b6770'; GRID = '#e3e7ea'
BLUE = '#346989'; RUST = '#ab5942'; GREEN = '#527d54'; SAND = '#b88a3b'; GREY = '#9aa3a9'
MODEL_C = {'lightgbm': BLUE, 'gru': RUST, 'transformer': GREEN}
MODEL_M = {'lightgbm': 'o', 'gru': 's', 'transformer': '^'}          # shape per family: readable in grayscale
NAME = {'lightgbm': 'LightGBM', 'gru': 'GRU', 'transformer': 'Transformer'}
MODELS = ['lightgbm', 'gru', 'transformer']
BED = {'DKASC': 'Array 1B', 'HKUST': 'HKUST', 'PVDAQ9068': 'PVDAQ 9068'}
MM = 1 / 25.4
W = 170 * MM
MINUS = '\u2212'


def title(ax, letter, s):
    ax.set_title(f'({letter}) {s}', fontsize=FS + 0.5, color=INK, loc='left', pad=4)


def enforce_min_linewidth(fig, minimum=0.5):
    """AIP: lines at least 0.5 pt. Raise any visible line or edge below the minimum; return the smallest width left."""
    widths = []
    for art in fig.findobj(lambda a: isinstance(a, (Line2D, matplotlib.patches.Patch, matplotlib.collections.LineCollection))):
        if isinstance(art, matplotlib.collections.LineCollection):
            lw = [max(w, minimum) if w > 0 else w for w in art.get_linewidths()]
            art.set_linewidths(lw); widths += [w for w in lw if w > 0]
        else:
            lw = art.get_linewidth()
            if lw and 0 < lw < minimum:
                art.set_linewidth(minimum); lw = minimum
            if lw and (not isinstance(art, matplotlib.patches.Patch) or art.get_edgecolor()[3] > 0):
                widths.append(lw)
    return min(widths)


def save(fig, name, records):
    min_lw = enforce_min_linewidth(fig)
    fig.canvas.draw()
    sizes = {t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()}
    assert min(sizes) >= FS, (name, sorted(sizes))
    assert min_lw >= 0.5, (name, min_lw)
    assert abs(fig.get_figwidth() - W) < 1e-6, name
    for ext in ('pdf', 'svg', 'tiff', 'png'):
        kw = {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {'dpi': 600}
        fig.savefig(OUT / f'{name}.{ext}', **kw)
    records.append({'figure': name, 'width_mm': round(fig.get_figwidth() / MM, 1), 'height_mm': round(fig.get_figheight() / MM, 1),
                    'min_text_pt': min(sizes), 'min_line_pt': round(min_lw, 2)})
    plt.close(fig)


def figure3(records):
    audit = json.loads((RES / 'd01_smart_persistence_audit_h6.json').read_text())
    inf = pd.read_csv(RES / 'd03_inference_holm_by_lead.csv')
    f1 = inf[(inf.family == 'F1_frozen_vs_reference') & (inf.horizon == 6)]
    fig = plt.figure(figsize=(W, 92 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.82, 0.82], left=0.085, right=0.99, bottom=0.31, top=0.90, wspace=0.62)
    ax = fig.add_subplot(gs[0])
    bins = list(audit['DKASC']['v28_error_by_source_clear_ghi'].keys())
    labels = ['20–\n100', '100–\n200', '200–\n400', '400–\n700', '>700']
    x = np.arange(len(bins)); w = 0.40
    for k, (bed, c) in enumerate((('DKASC', SAND), ('HKUST', BLUE))):
        b = audit[bed]['v28_error_by_source_clear_ghi']
        mae = [b[k2]['mae'] for k2 in bins]; share = [b[k2]['share'] for k2 in bins]
        ax.bar(x + (k - 0.5) * w, mae, w, color=c, label=BED[bed], zorder=3)
        for xi, m, s in zip(x + (k - 0.5) * w, mae, share):
            ax.text(xi, m + 0.008, f'{s*100:.0f}%', ha='center', va='bottom', fontsize=FS, color=MUTED, rotation=90)
        ax.axhline(audit[bed]['previous_day'], color=c, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(x, labels)
    ax.set_xlabel('Clear-sky GHI of the hour supplying\nthe persisted ratio (W m$^{-2}$)')
    ax.set_ylabel('Six-hour daylight MAE (normalized power)')
    ax.set_ylim(0, 0.50); ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.legend(loc='upper right', handlelength=1.2, bbox_to_anchor=(1.02, 1.0))
    title(ax, 'a', 'Smart persistence, any-daylight')
    refs = [('smart_v28', 'Smart, any-daylight ratio', GREY, 'o'), ('smart_revised', 'Smart, specified', SAND, 'D'),
            ('previous_day', 'Previous day', BLUE, 's'), ('convex_pd_climatology', 'Convex PD–climatology', INK, '^')]
    lo_lim = -60
    for j, bed in enumerate(('DKASC', 'HKUST')):
        ax = fig.add_subplot(gs[j + 1])
        for i, m in enumerate(MODELS):
            for k, (ref, lab, c, mk) in enumerate(refs):
                r = f1[(f1.testbed == bed) & (f1.model == m) & (f1.reference == ref)].iloc[0]
                y = 2 - i + (1.5 - k) * 0.18
                s, lo, hi = r.skill * 100, r.ci_low * 100, r.ci_high * 100
                if s < lo_lim:
                    ax.annotate(f'{s:.0f}%'.replace('-', MINUS), xy=(lo_lim, y), xytext=(lo_lim + 12, y), fontsize=FS, color=c,
                                va='center', arrowprops=dict(arrowstyle='->', color=c, lw=0.6))
                    continue
                ax.plot([max(lo, lo_lim), hi], [y, y], color=c, lw=1.0, solid_capstyle='butt', zorder=3)
                ax.plot(s, y, marker=mk, color=c, ms=4.0, mec='white', mew=0.4, zorder=4,
                        label=lab if (i == 0 and j == 0) else None)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks([2, 1, 0], ['LightGBM', 'GRU', 'Transformer'])
        ax.set_xlim(lo_lim, 70); ax.set_ylim(-0.55, 2.55)
        ax.set_xlabel('Skill of frozen model\nvs reference (%)')
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        title(ax, 'bc'[j], f'{BED[bed]}, 6 h, 2023')
    h3, l3 = fig.axes[1].get_legend_handles_labels()
    fig.legend(h3, l3, loc='lower center', ncol=4, bbox_to_anchor=(0.54, 0.0), handletextpad=0.4, columnspacing=1.2)
    save(fig, 'Fig3', records)


def figure4(records):
    s = pd.read_csv(RES / 'd05_skill_by_lead.csv')
    fig, axes = plt.subplots(1, 2, figsize=(W, 80 * MM), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.90, wspace=0.08)
    lo_lim = -62
    leads = [2, 4, 6, 12]
    xpos = {2: 0, 4: 1, 6: 2, 12: 3}
    for j, bed in enumerate(('DKASC', 'HKUST')):
        ax = axes[j]
        ax.axhspan(lo_lim, 0, color='#f4f1ec', zorder=0)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        for k, m in enumerate(MODELS):
            for p, ls, fill in (('frozen_2021', (0, (2.5, 1.5)), 'white'), ('monthly_trailing_365_days', '-', None)):
                q = s[(s.testbed == bed) & (s.model == m) & (s.policy == p)].set_index('horizon').loc[leads]
                x = np.array([xpos[h] for h in leads]) + (k - 1) * 0.10
                y = q.skill.values * 100; lo = q.ci_low.values * 100; hi = q.ci_high.values * 100
                ax.plot(x, np.maximum(y, lo_lim), color=MODEL_C[m], lw=1.1, ls=ls, zorder=3)
                for xi, yi, l, h_ in zip(x, y, lo, hi):
                    if yi < lo_lim:
                        ax.plot(xi, lo_lim + 2.5, marker='v', color=MODEL_C[m], ms=4, mfc=fill or MODEL_C[m], zorder=5)
                        ax.text(xi - 0.10, lo_lim + 7.0, f'{yi:.0f}%'.replace('-', MINUS), fontsize=FS, color=MODEL_C[m],
                                ha='right', va='center')
                        continue
                    ax.plot([xi, xi], [max(l, lo_lim), h_], color=MODEL_C[m], lw=0.7, zorder=2)
                    ax.plot(xi, yi, marker=MODEL_M[m], color=MODEL_C[m], ms=4.0, mfc=fill or MODEL_C[m], mew=0.8, zorder=4)
        ax.set_xticks(range(4), [f'{h} h' for h in leads])
        ax.set_xlim(-0.45, 3.4); ax.set_ylim(lo_lim, 45)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        title(ax, 'ab'[j], f'{BED[bed]}, 2023 daylight targets')
        ax.set_xlabel('Lead time')
    axes[0].set_ylabel('Skill vs previous-day persistence (%)')
    h = [Line2D([], [], color=MODEL_C[m], lw=1.1, marker=MODEL_M[m], ms=4.0, label=NAME[m]) for m in MODELS]
    h += [Line2D([], [], color=INK, lw=1.1, ls=(0, (2.5, 1.5)), label='frozen 2021 (open)'),
          Line2D([], [], color=INK, lw=1.1, label='monthly refit (filled)')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.54, 0.0), columnspacing=1.4)
    save(fig, 'Fig4', records)


def figure5(records):
    c = pd.read_csv(RES / 'd05_coverage_by_hour.csv')
    fig, axes = plt.subplots(1, 2, figsize=(W, 80 * MM), sharey=False)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.90, wspace=0.17)
    style = {'raw': (GREY, 'raw quantiles'), 'global_all_hours': (SAND, 'all-hour correction'),
             'global_day_night': (BLUE, 'day/night correction')}
    for j, bed in enumerate(('DKASC', 'HKUST')):
        ax = axes[j]
        q = c[(c.testbed == bed) & (c.model == 'lightgbm') & ~c.target_hour.isin(['daylight', 'night'])].copy()
        q['target_hour'] = q.target_hour.astype(int)
        night = q[q.method == 'raw'].set_index('target_hour').daylight_share.reindex(range(24)).fillna(0)
        for hr in range(24):
            if night[hr] < 0.5:
                ax.axvspan(hr - 0.5, hr + 0.5, color='#eef0f3', zorder=0, lw=0)
        ax.axhline(80, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
        for meth, (col, lab) in style.items():
            g = q[q.method == meth].sort_values('target_hour')
            ax.plot(g.target_hour, g.coverage * 100, color=col, lw=2.2 if meth == 'global_all_hours' else 1.1, marker='o', ms=2.6,
                    ls=(0, (2, 1.5)) if meth == 'raw' else '-', label=lab if j == 0 else None, zorder=4 if meth == 'raw' else 3)
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0 if bed == 'DKASC' else 40, 102)
        ax.set_xticks(range(0, 24, 3)); ax.set_xlabel('Target hour (local time)')
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        dn = {m: c[(c.testbed == bed) & (c.model == 'lightgbm') & (c.method == m) & (c.target_hour == 'daylight')].coverage.iloc[0] * 100
              for m in style}
        title(ax, 'ab'[j], f'{BED[bed]}, LightGBM, nominal 80%')
        txt = (f"daylight coverage:\nraw {dn['raw']:.1f}%, all-hour {dn['global_all_hours']:.1f}%,\n"
               f"day/night {dn['global_day_night']:.1f}%")
        xy, va = ((10.6, 4), 'bottom') if bed == 'DKASC' else ((8.7, 99), 'top')
        ax.text(*xy, txt, fontsize=FS, color=INK, va=va, ha='left', zorder=6)
    axes[0].set_ylabel('Empirical coverage, 2023 (%)')
    hd, lb = axes[0].get_legend_handles_labels()
    hd.append(Patch(color='#eef0f3', label='night target hours')); lb.append('night target hours')
    fig.legend(hd, lb, loc='lower center', ncol=4, bbox_to_anchor=(0.54, 0.0), columnspacing=1.4)
    save(fig, 'Fig5', records)


def neg_zone(ax, lo, orient='x'):
    (ax.axvspan if orient == 'x' else ax.axhspan)(lo, 0, color='#f3eeea', zorder=0, lw=0)
    (ax.axvline if orient == 'x' else ax.axhline)(0, color=MUTED, lw=0.6, zorder=1)


def figure6(records):
    inf = pd.read_csv(RES9 / 'd06_inference_9068.csv')
    c5 = pd.read_csv(RES9 / 'd09_matched_c5_9068.csv')
    tun = pd.read_csv(RES9 / 'd07_tuning_evaluation.csv')
    fig = plt.figure(figsize=(W, 100 * MM))
    B, H = 0.25, 0.64
    POS = [(0.135, B, 0.185, H), (0.415, B, 0.205, H), (0.785, B, 0.135, H)]
    # (a)
    ax = fig.add_axes(POS[0])
    XL = (-20, 25)
    neg_zone(ax, XL[0])
    yt, yl = [], []
    for i, m in enumerate(MODELS):
        for j, (pol, fam, lab) in enumerate((('frozen_2021', 'C1', 'frozen'), ('monthly_trailing_365_days', 'C3_vs_pd', 'monthly'))):
            yb = 5 - (2 * i + j)
            yt.append(yb); yl.append(f'{NAME[m]}\n{lab}')
            for k, (yr, mk) in enumerate(((2023, 'o'), (2024, 's'))):
                r = inf[(inf.family == fam) & (inf.year == yr) & (inf.horizon == 6) & (inf.model == m) & (inf.policy == pol)
                        & (inf.reference == 'previous_day')].iloc[0]
                y = yb + (0.5 - k) * 0.36
                filled = pol != 'frozen_2021'
                ax.plot([r.ci_low * 100, r.ci_high * 100], [y, y], color=MODEL_C[m], lw=0.9, zorder=3)
                ax.plot(r.skill * 100, y, mk, ms=4.0, mfc=MODEL_C[m] if filled else 'white', mec=MODEL_C[m], mew=0.8, zorder=4)
    ax.set_yticks(yt, yl)
    ax.set_ylim(-0.6, 5.6); ax.set_xlim(*XL)
    ax.set_xlabel('6-h skill vs\nprevious day (%)')
    ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    title(ax, 'a', 'PVDAQ 9068, 6 h')
    ax.plot([], [], 'o', color=INK, mfc='white', ms=4, label='2023'); ax.plot([], [], 's', color=INK, mfc='white', ms=4, label='2024')
    ax.legend(loc='upper center', handlelength=1.0, bbox_to_anchor=(0.45, -0.24), ncol=2, columnspacing=0.8)
    # (b)
    ax = fig.add_axes(POS[1])
    YL = (-15, 40)
    neg_zone(ax, YL[0], orient='y')
    leads = [2, 4, 6, 12]
    xpos = {2: 0, 4: 1, 6: 2, 12: 3}
    for i, m in enumerate(MODELS):
        for k, (yr, ls) in enumerate(((2023, (0, (3, 2))), (2024, '-'))):
            q = c5[(c5.model == m) & (c5.year == yr)].set_index('horizon').loc[leads]
            x = np.array([xpos[h] for h in leads]) + (i - 1) * 0.11 + (k - 0.5) * 0.05
            ax.plot(x, q.skill * 100, ls=ls, color=MODEL_C[m], lw=0.8, zorder=3)
            ax.vlines(x, q.ci_low * 100, q.ci_high * 100, color=MODEL_C[m], lw=0.6, alpha=0.8, zorder=2)
            ax.plot(x, q.skill * 100, MODEL_M[m], ms=3.6, mfc='white' if yr == 2023 else MODEL_C[m], mec=MODEL_C[m], mew=0.7,
                    ls='none', zorder=4, label=NAME[m] if yr == 2024 else None)
    ax.set_xticks(range(4), [f'{h} h' for h in leads]); ax.set_xlim(-0.4, 3.4); ax.set_ylim(*YL)
    ax.set_ylabel('Frozen-model skill vs previous day (%)')
    ax.set_xlabel('Lead time\n(matched daylight targets)')
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    title(ax, 'b', 'PVDAQ 9068, frozen fits')
    ax.legend(loc='upper right', handlelength=1.4)
    ax.text(0.98, 0.66, 'open/dashed: 2023\nfilled/solid: 2024', transform=ax.transAxes, ha='right', va='top', fontsize=FS, color=MUTED)
    # (c)
    ax = fig.add_axes(POS[2])
    XL = (-80, 15)
    neg_zone(ax, XL[0])
    yt, yl = [], []
    row = 0
    for bed in ('DKASC', 'HKUST', 'PVDAQ9068'):
        for m in ('gru', 'transformer'):
            yb = 5 - row; row += 1
            yt.append(yb); yl.append(f'{BED[bed]}\n{NAME[m]}')
            for k, (con, filled) in enumerate((('default_vs_pd', False), ('tuned_vs_pd', True))):
                r = tun[(tun.bed == bed) & (tun.model == m) & (tun.year == 2023) & (tun.contrast == con)].iloc[0]
                y = yb + (0.5 - k) * 0.32
                s, lo, hi = r.skill * 100, r.ci_low * 100, r.ci_high * 100
                if s < XL[0]:
                    ax.text(XL[0] + 1, y, f'\u2190 default {s:.0f}%'.replace('-', MINUS), va='center', ha='left', fontsize=FS,
                            color=MODEL_C[m])
                    continue
                ax.plot([max(lo, XL[0]), min(hi, XL[1])], [y, y], color=MODEL_C[m], lw=0.9, zorder=3)
                ax.plot(s, y, 'o' if not filled else 'D', ms=3.8, mfc=MODEL_C[m] if filled else 'white', mec=MODEL_C[m], mew=0.8,
                        zorder=4)
            t = tun[(tun.bed == bed) & (tun.model == m) & (tun.year == 2023) & (tun.contrast == 'tuned_vs_default')].iloc[0]
            ax.text(XL[1] + 3, yb, (f'{t.skill * 100:+.1f}%' if abs(t.skill) < 0.01 else f'{t.skill * 100:+.0f}%').replace('-', MINUS),
                    va='center', ha='left', fontsize=FS, color=MUTED, clip_on=False)
    ax.set_yticks(yt, yl)
    ax.set_ylim(-0.6, 5.6); ax.set_xlim(*XL)
    ax.set_xlabel('6-h skill vs previous\nday, 2023 (%)')
    ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    title(ax, 'c', 'Neural configuration')
    ax.plot([], [], 'o', color=INK, mfc='white', ms=4, label='default'); ax.plot([], [], 'D', color=INK, mfc=INK, ms=3.6, label='tuned')
    ax.legend(loc='upper center', handlelength=1.0, bbox_to_anchor=(0.40, -0.24), ncol=2, columnspacing=0.8)
    save(fig, 'Fig6', records)


if __name__ == '__main__':
    rec = []
    figure3(rec); figure4(rec); figure5(rec); figure6(rec)
    (OUT / 'figures_3_to_6_check.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))

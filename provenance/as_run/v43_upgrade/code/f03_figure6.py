"""F03: Figure 6 (v43) - pre-specified evaluation at PVDAQ 9068 and neural-configuration sensitivity.

(a) 9068 six-hour skill against previous-day persistence, frozen vs monthly, 2023 and 2024 (d06, C1/C3).
(b) 9068 frozen skill against previous-day persistence by lead on matched daylight keys (d09, C5).
(c) Six-hour skill against previous-day persistence of default vs tuned neural configurations at all three testbeds (d07).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'v43/figures'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
INK = '#1f2a33'; MUTED = '#5b6770'; GRID = '#e3e7ea'
BLUE = '#346989'; RUST = '#ab5942'; GREEN = '#527d54'; SAND = '#b88a3b'; GREY = '#9aa3a9'
MODEL_C = {'lightgbm': BLUE, 'gru': RUST, 'transformer': GREEN}
NAME = {'lightgbm': 'LightGBM', 'gru': 'GRU', 'transformer': 'Transformer'}
MODELS = ['lightgbm', 'gru', 'transformer']
MM = 1 / 25.4


def tag(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', color=INK, va='bottom')


def neg_zone(ax, lo, orient='x'):
    (ax.axvspan if orient == 'x' else ax.axhspan)(lo, 0, color='#f3eeea', zorder=0, lw=0)
    (ax.axvline if orient == 'x' else ax.axhline)(0, color=MUTED, lw=0.6, zorder=1)


inf = pd.read_csv(RES / 'd06_inference_9068.csv')
c5 = pd.read_csv(RES / 'd09_matched_c5_9068.csv')
tun = pd.read_csv(RES / 'd07_tuning_evaluation.csv')

fig = plt.figure(figsize=(190 * MM, 74 * MM))
B, H = 0.22, 0.66
POS = [(0.095, B, 0.22, H), (0.415, B, 0.215, H), (0.745, B, 0.19, H)]

# (a) six-hour skill vs previous day, frozen and monthly, two years
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
            y = yb + (0.5 - k) * 0.34
            filled = pol != 'frozen_2021'
            ax.plot([r.ci_low * 100, r.ci_high * 100], [y, y], color=MODEL_C[m], lw=0.9, zorder=3)
            ax.plot(r.skill * 100, y, mk, ms=3.6, mfc=MODEL_C[m] if filled else 'white', mec=MODEL_C[m], mew=0.8, zorder=4)
ax.set_yticks(yt, yl, fontsize=6)
ax.set_ylim(-0.6, 5.6); ax.set_xlim(*XL)
ax.set_xlabel('Six-hour skill vs previous day (%)', fontsize=6.4)
ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
ax.set_title('PVDAQ 9068, six hours', fontsize=6.8, color=INK, loc='left')
ax.plot([], [], 'o', color=INK, mfc='white', ms=3.4, label='2023'); ax.plot([], [], 's', color=INK, mfc='white', ms=3.4, label='2024')
ax.legend(loc='upper left', fontsize=5.8, handlelength=1.0, borderaxespad=0.2)
tag(ax, 'a', x=-0.40)

# (b) frozen skill by lead, matched keys
ax = fig.add_axes(POS[1])
YL = (-15, 40)
neg_zone(ax, YL[0], orient='y')
leads = [2, 4, 6, 12]
xpos = {2: 0, 4: 1, 6: 2, 12: 3}
for i, m in enumerate(MODELS):
    for k, (yr, ls, mk) in enumerate(((2023, (0, (3, 2)), 'o'), (2024, '-', 's'))):
        q = c5[(c5.model == m) & (c5.year == yr)].set_index('horizon').loc[leads]
        x = np.array([xpos[h] for h in leads]) + (i - 1) * 0.1 + (k - 0.5) * 0.045
        ax.plot(x, q.skill * 100, ls=ls, color=MODEL_C[m], lw=0.8, zorder=3)
        ax.vlines(x, q.ci_low * 100, q.ci_high * 100, color=MODEL_C[m], lw=0.6, alpha=0.8, zorder=2)
        ax.plot(x, q.skill * 100, mk, ms=3.0, mfc='white' if yr == 2023 else MODEL_C[m], mec=MODEL_C[m], mew=0.7, zorder=4,
                label=NAME[m] if yr == 2024 else None)
ax.set_xticks(range(4), [f'{h} h' for h in leads]); ax.set_xlim(-0.4, 3.4); ax.set_ylim(*YL)
ax.set_ylabel('Frozen-model skill vs previous day (%)', fontsize=6.4)
ax.set_xlabel('Lead time (matched daylight targets)', fontsize=6.4)
ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
ax.set_title('PVDAQ 9068, frozen 2021 fits', fontsize=6.8, color=INK, loc='left')
ax.legend(loc='upper right', fontsize=5.8, handlelength=1.4)
ax.text(0.98, 0.70, 'open/dashed: 2023\nfilled/solid: 2024', transform=ax.transAxes, ha='right', va='top', fontsize=5.4, color=MUTED)
tag(ax, 'b', x=-0.26)

# (c) default vs tuned neural configurations, 2023
ax = fig.add_axes(POS[2])
XL = (-80, 15)
neg_zone(ax, XL[0])
beds = [('DKASC', 'Array 1B'), ('HKUST', 'HKUST'), ('PVDAQ9068', 'PVDAQ 9068')]
yt, yl = [], []
row = 0
for bed, bl in beds:
    for m in ('gru', 'transformer'):
        yb = 5 - row; row += 1
        yt.append(yb); yl.append(f'{bl}\n{NAME[m]}')
        pts = {}
        for k, (con, filled) in enumerate((('default_vs_pd', False), ('tuned_vs_pd', True))):
            r = tun[(tun.bed == bed) & (tun.model == m) & (tun.year == 2023) & (tun.contrast == con)].iloc[0]
            y = yb + (0.5 - k) * 0.30
            s, lo, hi = r.skill * 100, r.ci_low * 100, r.ci_high * 100
            pts[con] = (s, y)
            if s < XL[0]:
                ax.text(XL[0] + 1, y, f'\u2190 default {s:.0f}%'.replace('-', '\u2212'), va='center', ha='left', fontsize=5.4, color=MODEL_C[m])
                continue
            ax.plot([max(lo, XL[0]), min(hi, XL[1])], [y, y], color=MODEL_C[m], lw=0.9, zorder=3)
            ax.plot(s, y, 'o' if not filled else 'D', ms=3.4, mfc=MODEL_C[m] if filled else 'white', mec=MODEL_C[m], mew=0.8, zorder=4)
        t = tun[(tun.bed == bed) & (tun.model == m) & (tun.year == 2023) & (tun.contrast == 'tuned_vs_default')].iloc[0]
        ax.text(XL[1] + 1.5, yb, (f'{t.skill * 100:+.1f}%' if abs(t.skill) < 0.01 else f'{t.skill * 100:+.0f}%').replace('-', '\u2212'), va='center', ha='left', fontsize=5.4, color=MUTED, clip_on=False)
ax.text(XL[1] + 1.5, 5.75, 'tuned vs\ndefault\nMAE', va='bottom', ha='left', fontsize=5.2, color=MUTED, clip_on=False)
ax.set_yticks(yt, yl, fontsize=6)
ax.set_ylim(-0.6, 5.6); ax.set_xlim(*XL)
ax.set_xlabel('Six-hour skill vs previous day, 2023 (%)', fontsize=6.4)
ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
ax.set_title('Neural configuration', fontsize=6.8, color=INK, loc='left')
ax.plot([], [], 'o', color=INK, mfc='white', ms=3.4, label='default'); ax.plot([], [], 'D', color=INK, mfc=INK, ms=3.2, label='tuned')
ax.legend(loc='upper left', fontsize=5.8, handlelength=1.0, bbox_to_anchor=(0.0, -0.17), ncol=2)
tag(ax, 'c', x=-0.40)

for ext in ('pdf', 'svg'):
    fig.savefig(OUT / f'Figure6_confirmation_and_tuning_v43.{ext}')
fig.savefig(OUT / 'Figure6_confirmation_and_tuning_v43.png', dpi=600)
print('saved')

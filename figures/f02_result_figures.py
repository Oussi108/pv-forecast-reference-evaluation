"""F02: v42 result figures (data-derived; values from D01, D02/D03, D05 outputs).

Figure 3  reference dependence: (a) why the earlier smart specification fails, (b,c) skill of frozen models vs four references.
Figure 4  skill vs lead time, frozen vs monthly, all three model families.
Figure 5  coverage of nominal-80% six-hour intervals by target hour (raw, all-hour, day/night).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results/reference_and_update_inference'
OUT = ROOT / 'results/figures'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
INK = '#1f2a33'; MUTED = '#5b6770'; GRID = '#e3e7ea'
BLUE = '#346989'; RUST = '#ab5942'; GREEN = '#527d54'; SAND = '#b88a3b'; GREY = '#9aa3a9'
MODEL_C = {'lightgbm': BLUE, 'gru': RUST, 'transformer': GREEN}
NAME = {'lightgbm': 'LightGBM', 'gru': 'GRU', 'transformer': 'Transformer'}
MM = 1 / 25.4


def tag(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', color=INK, va='bottom')


def save(fig, name):
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}')
    fig.savefig(OUT / f'{name}.png', dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def figure3():
    audit = json.loads((RES / 'd01_smart_persistence_audit_h6.json').read_text())
    inf = pd.read_csv(RES / 'd03_inference_holm_by_lead.csv')
    f1 = inf[(inf.family == 'F1_frozen_vs_reference') & (inf.horizon == 6)]
    fig = plt.figure(figsize=(190 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], left=0.065, right=0.99, bottom=0.30, top=0.88, wspace=0.42)

    # (a) mechanism
    ax = fig.add_subplot(gs[0])
    bins = list(audit['DKASC']['v28_error_by_source_clear_ghi'].keys())
    labels = ['20–\n100', '100–\n200', '200–\n400', '400–\n700', '>700']
    x = np.arange(len(bins)); w = 0.38
    for k, (bed, c) in enumerate((('DKASC', SAND), ('HKUST', BLUE))):
        b = audit[bed]['v28_error_by_source_clear_ghi']
        mae = [b[k2]['mae'] for k2 in bins]; share = [b[k2]['share'] for k2 in bins]
        bars = ax.bar(x + (k - 0.5) * w, mae, w, color=c, label=bed, zorder=3)
        for xi, m, s in zip(x + (k - 0.5) * w, mae, share):
            ax.text(xi, m + 0.006, f'{s*100:.0f}%', ha='center', va='bottom', fontsize=5.2, color=MUTED, rotation=90)
        ax.axhline(audit[bed]['previous_day'], color=c, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(x, labels, fontsize=5.8)
    ax.set_xlabel('Clear-sky GHI of the hour supplying\nthe persisted ratio (W m$^{-2}$)', fontsize=6.4)
    ax.set_ylabel('Six-hour daylight MAE (normalized power)', fontsize=6.4)
    ax.set_ylim(0, 0.46); ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.legend(loc='upper right', fontsize=6, handlelength=1.2, bbox_to_anchor=(1.0, 1.0))
    ax.text(0.98, 0.78, 'dashed: previous-day MAE' + chr(10) + 'labels: share of targets', transform=ax.transAxes,
            ha='right', va='top', fontsize=5.4, color=MUTED)
    ax.set_title('Smart persistence, any-daylight ratio', fontsize=6.8, color=INK, loc='left')
    tag(ax, 'a', x=-0.2)

    refs = [('smart_v28', 'Smart, any-daylight ratio', GREY, 'o'), ('smart_revised', 'Smart, specified', SAND, 'D'),
            ('previous_day', 'Previous day', BLUE, 's'), ('convex_pd_climatology', 'Convex PD–climatology', INK, '^')]
    lo_lim = -60
    for j, bed in enumerate(('DKASC', 'HKUST')):
        ax = fig.add_subplot(gs[j + 1])
        for i, m in enumerate(('lightgbm', 'gru', 'transformer')):
            for k, (ref, lab, c, mk) in enumerate(refs):
                r = f1[(f1.testbed == bed) & (f1.model == m) & (f1.reference == ref)].iloc[0]
                y = 2 - i + (1.5 - k) * 0.17
                s, lo, hi = r.skill * 100, r.ci_low * 100, r.ci_high * 100
                if s < lo_lim:
                    ax.annotate(f'{s:.0f}%', xy=(lo_lim, y), xytext=(lo_lim + 9, y), fontsize=5.2, color=c, va='center',
                                arrowprops=dict(arrowstyle='->', color=c, lw=0.6))
                    continue
                ax.plot([max(lo, lo_lim), hi], [y, y], color=c, lw=1.0, solid_capstyle='butt', zorder=3)
                ax.plot(s, y, marker=mk, color=c, ms=3.6, mec='white', mew=0.4, zorder=4,
                        label=lab if (i == 0 and j == 0) else None)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks([2, 1, 0], ['LightGBM', 'GRU', 'Transformer'])
        ax.set_xlim(lo_lim, 70); ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Skill of frozen model vs reference (%)', fontsize=6.4)
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.set_title(f'{bed}, six hours, 2023 daylight', fontsize=6.8, color=INK, loc='left')
        tag(ax, 'bc'[j], x=-0.22)
    h3, l3 = fig.axes[1].get_legend_handles_labels()
    fig.legend(h3, l3, loc='lower center', ncol=4, fontsize=6.2, bbox_to_anchor=(0.66, 0.0), handletextpad=0.4, columnspacing=1.4)
    save(fig, 'Figure3_reference_dependence_v42')


# ---------------------------------------------------------------- Figure 4
def figure4():
    s = pd.read_csv(RES / 'd05_skill_by_lead.csv')
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.88, wspace=0.08)
    lo_lim = -62
    leads = [2, 4, 6, 12]
    xpos = {2: 0, 4: 1, 6: 2, 12: 3}
    for j, bed in enumerate(('DKASC', 'HKUST')):
        ax = axes[j]
        ax.axhspan(lo_lim, 0, color='#f4f1ec', zorder=0)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        for k, m in enumerate(('lightgbm', 'gru', 'transformer')):
            for p, ls, mk, fill in (('frozen_2021', (0, (2.5, 1.5)), 'o', 'white'), ('monthly_trailing_365_days', '-', 'o', None)):
                q = s[(s.testbed == bed) & (s.model == m) & (s.policy == p)].set_index('horizon').loc[leads]
                x = np.array([xpos[h] for h in leads]) + (k - 1) * 0.09
                y = q.skill.values * 100; lo = q.ci_low.values * 100; hi = q.ci_high.values * 100
                yc = np.maximum(y, lo_lim)
                ax.plot(x, yc, color=MODEL_C[m], lw=1.1, ls=ls, zorder=3)
                for xi, yi, l, h_ in zip(x, y, lo, hi):
                    if yi < lo_lim:
                        ax.plot(xi, lo_lim + 1.8, marker='v', color=MODEL_C[m], ms=3.5, mfc=fill or MODEL_C[m], zorder=5)
                        ax.text(xi - 0.07, lo_lim + 1.8, f'{yi:.0f}%', fontsize=5, color=MODEL_C[m], ha='right', va='center')
                        continue
                    ax.plot([xi, xi], [max(l, lo_lim), h_], color=MODEL_C[m], lw=0.7, zorder=2)
                    ax.plot(xi, yi, marker=mk, color=MODEL_C[m], ms=3.4, mfc=fill or MODEL_C[m], mew=0.8, zorder=4)
        ax.set_xticks(range(4), [f'{h} h' for h in leads])
        ax.set_xlim(-0.4, 3.4); ax.set_ylim(lo_lim, 45)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        ax.set_title(f'{bed}, 2023 daylight targets', fontsize=6.8, color=INK, loc='left')
        ax.set_xlabel('Lead time', fontsize=6.4)
        tag(ax, 'ab'[j], x=-0.09 if j == 0 else -0.03)
    axes[0].set_ylabel('Skill vs previous-day persistence (%)', fontsize=6.4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MODEL_C[m], lw=1.1, label=NAME[m]) for m in MODEL_C]
    h += [Line2D([], [], color=INK, lw=1.1, ls=(0, (2.5, 1.5)), marker='o', mfc='white', ms=3.4, label='frozen 2021'),
          Line2D([], [], color=INK, lw=1.1, marker='o', ms=3.4, label='monthly refit')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=6.2, bbox_to_anchor=(0.53, 0.0), columnspacing=1.6)
    save(fig, 'Figure4_skill_by_lead_v42')


# ---------------------------------------------------------------- Figure 5
def figure5():
    c = pd.read_csv(RES / 'd05_coverage_by_hour.csv')
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.88, wspace=0.16)
    style = {'raw': (GREY, 'raw quantiles'), 'global_all_hours': (SAND, 'all-hour correction'), 'global_day_night': (BLUE, 'day/night correction')}
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
            ax.plot(g.target_hour, g.coverage * 100, color=col, lw=2.2 if meth == 'global_all_hours' else 1.1, marker='o', ms=2.4,
                    ls=(0, (2, 1.5)) if meth == 'raw' else '-', label=lab if j == 0 else None, zorder=4 if meth == 'raw' else 3)
            dl = c[(c.testbed == bed) & (c.model == 'lightgbm') & (c.method == meth) & (c.target_hour == 'daylight')].coverage.iloc[0]
            ax.text(23.6, dl * 100, '', fontsize=5)
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0 if bed == 'DKASC' else 40, 102)
        ax.set_xticks(range(0, 24, 3)); ax.set_xlabel('Target hour (local time)', fontsize=6.4)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        dn = {m: c[(c.testbed == bed) & (c.model == 'lightgbm') & (c.method == m) & (c.target_hour == 'daylight')].coverage.iloc[0] * 100
              for m in style}
        ax.set_title(f'{bed}, LightGBM, nominal 80%', fontsize=6.8, color=INK, loc='left')
        ax.text(0.02, 0.05, f"daylight coverage: raw {dn['raw']:.1f}% · all-hour {dn['global_all_hours']:.1f}% · day/night {dn['global_day_night']:.1f}%",
                transform=ax.transAxes, fontsize=5.4, color=INK)
        tag(ax, 'ab'[j], x=-0.09 if j == 0 else -0.03)
    axes[0].set_ylabel('Empirical coverage, 2023 (%)', fontsize=6.4)
    from matplotlib.patches import Patch
    hd, lb = axes[0].get_legend_handles_labels()
    hd.append(Patch(color='#eef0f3', label='night target hours')); lb.append('night target hours')
    fig.legend(hd, lb, loc='lower center', ncol=4, fontsize=6.2, bbox_to_anchor=(0.53, 0.0), columnspacing=1.6)
    save(fig, 'Figure5_coverage_by_hour_v42')


if __name__ == '__main__':
    figure3(); figure4()
    if (RES / 'd05_coverage_by_hour.csv').exists():
        figure5()
    print('saved')

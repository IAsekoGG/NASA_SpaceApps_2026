"""Графіки доповіді (ті самі, що в документах) як PNG, англійською й українською.

Дані: data/analysis_rfi_clean/change_rfiaware.json, web/data/app.json, report_figures/chart_rows.json.
    python -I tools/report_charts.py
"""
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT = Path(__file__).resolve().parents[1]
INK, QUIET, GRID, ACC, MUTED = '#1f1f1d', '#6b6a63', '#e4e3de', '#2a78d6', '#b9b8b0'
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12, 'text.color': INK, 'axes.edgecolor': GRID,
                     'xtick.color': QUIET, 'ytick.color': QUIET, 'axes.labelcolor': QUIET,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

T = {
 'en': dict(
   kpi_title='Four numbers that sum up the work',
   kpi=[('98.2 %', '13.7 %', 'of the area flagged as “changed”', 'raw data → new test, track 143 A'),
        ('100 %', '3.4 %', 'false “changes” under the strongest interference', 'interference > 10 × signal, track 143 A'),
        ('+10.4 dB', '+1.3 dB', 'cropland HV on a dirty date vs a clean one', 'track 108 D, ideal is 0 dB'),
        ('', '× 4.0', 'more often than chance, two tracks', 'see the same change on cropland')],
   leak_title='The new test does not map interference: just 3.4 % “changes” where it is strongest',
   leak_sub='Share of pixels flagged “changed”, track 143 A, 11 dates, FDR 5 %; axis: peak interference relative to the HH signal',
   leak_names={'raw': 'raw data', 'naive': 'subtract, then test', 'aware': 'new test'}, leak_x='interference / signal',
   water_title='Water is dark again: in 143 A after cleaning −17…−19 dB on every date instead of −5…−18',
   water_sub='Mean HV power over permanent water, dB; water was not used to estimate the interference (spatial folds)',
   raw='raw', clean='cleaned', ref='true L-band water: below −20 dB',
   field_title='One field: raw data “brighten” by 13 dB, yet nothing happened on the ground',
   field_sub='HH, 480 × 480 m window on cropland, track 143 A, 11 dates', field_raw='raw data', field_clean='after cleaning',
   fc_title='Changes peak in July–September: cereal harvest, sunflower, ploughing',
   fc_sub='Area where the test found the first change, km²; each bar sits on the acquisition that closes the interval',
   track='track', km='km²', months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
   files=['12_chart_four_numbers.png', '13_chart_three_tests.png', '14_chart_water_by_date.png', '15_chart_one_field.png', '16_chart_first_change_by_date.png'],
   dir='Report figures (EN)'),
 'uk': dict(
   kpi_title='Чотири числа, які підсумовують роботу',
   kpi=[('98.2 %', '13.7 %', 'території позначено «змінами»', 'сирі дані → новий тест, трек 143 A'),
        ('100 %', '3.4 %', 'хибних «змін» під найсильнішою завадою', 'завада > 10 × сигналу, трек 143 A'),
        ('+10.4 дБ', '+1.3 дБ', 'рілля HV на брудній даті проти чистої', 'трек 108 D, ідеал — 0 дБ'),
        ('', '× 4.0', 'частіше за випадковість два треки', 'бачать ту саму зміну на ріллі')],
   leak_title='Новий тест не малює перешкоди: під найсильнішою завадою лише 3.4 % «змін»',
   leak_sub='Частка пікселів «змінилось», трек 143 A, 11 дат, FDR 5 %; вісь — максимум завади відносно сигналу HH',
   leak_names={'raw': 'сирі дані', 'naive': 'відняти фон, потім тест', 'aware': 'новий тест'}, leak_x='завада / сигнал',
   water_title='Вода знову темна: у 143 A після чистки −17…−19 дБ на кожну дату замість −5…−18',
   water_sub='Середня потужність HV над постійною водою, дБ; вода в оцінку перешкоди не входила (просторові фолди)',
   raw='сирі', clean='очищені', ref='справжня вода L-band — нижче −20 дБ',
   field_title='Одне поле: у сирих даних «яскравішає» на 13 дБ, насправді жодної події',
   field_sub='HH, вікно 480 × 480 м на ріллі, трек 143 A, 11 дат', field_raw='сирі дані', field_clean='після чистки',
   fc_title='Пік змін — липень–вересень: збирання зернових, соняшник, оранка',
   fc_sub='Площа, де тест знайшов першу зміну, км²; стовпчик стоїть на даті знімка, що закриває проміжок',
   track='трек', km='км²', months=['січ', 'лют', 'бер', 'квіт', 'трав', 'черв', 'лип', 'серп', 'вер', 'жовт', 'лист', 'груд'],
   files=['12_графік_чотири_числа.png', '13_графік_три_тести.png', '14_графік_вода_за_датами.png', '15_графік_одне_поле.png', '16_графік_перша_зміна_за_датами.png'],
   dir='Ілюстрації доповіді'),
}


def base(ax, t):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def heading(fig, title, sub):
    fig.text(0.03, 0.965, title, fontsize=17, fontweight='bold', va='top')
    fig.text(0.03, 0.895, sub, fontsize=12, color=QUIET, va='top')


def months_axis(ax, t):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: t['months'][mdates.num2date(v).month - 1]))


def kpi(t, path):
    fig = plt.figure(figsize=(12, 5.2))
    fig.text(0.03, 0.95, t['kpi_title'], fontsize=17, fontweight='bold', va='top')
    for i, (b, a, n1, n2) in enumerate(t['kpi']):
        x0, y0 = 0.03 + (i % 2) * 0.485, 0.47 - (i // 2) * 0.43
        ax = fig.add_axes([x0, y0, 0.455, 0.38]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        ax.add_patch(matplotlib.patches.FancyBboxPatch((0.005, 0.02), 0.99, 0.96, boxstyle='round,pad=0,rounding_size=0.04',
                                                       fc='white', ec=GRID, lw=1.3, transform=ax.transAxes))
        if b:
            tb = ax.text(0.05, 0.62, f'{b}  →  ', fontsize=19, color=QUIET, va='center')
            fig.canvas.draw(); bb = tb.get_window_extent().transformed(ax.transAxes.inverted())
            ax.text(bb.x1, 0.62, a, fontsize=30, fontweight='bold', color=ACC, va='center')
        else:
            ax.text(0.05, 0.62, a, fontsize=30, fontweight='bold', color=ACC, va='center')
        ax.text(0.05, 0.33, n1, fontsize=12.5, color=INK, va='center')
        ax.text(0.05, 0.15, n2, fontsize=11.5, color=QUIET, va='center')
    fig.savefig(path, dpi=200); plt.close(fig)


def leak(t, app, path):
    L = app['leak_143']; bins = L['bins']; x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(12, 6)); fig.subplots_adjust(left=0.08, right=0.72, top=0.80, bottom=0.14)
    heading(fig, t['leak_title'], t['leak_sub']); base(ax, t)
    style = {'raw': (MUTED, '-', 2), 'naive': (MUTED, '--', 2), 'aware': (ACC, '-', 3)}
    for k in ('raw', 'naive', 'aware'):
        v = L['tests'][k]['by_ratio']; c, ls, lw = style[k]
        ax.plot(x, v, ls, color=c, lw=lw, marker='o', ms=7, alpha=0.75 if k == 'naive' else 1)
        ax.annotate(f"{t['leak_names'][k]}: {v[-1]:g} %", (x[-1], v[-1]), xytext=(14, 0), textcoords='offset points',
                    va='center', fontsize=12.5, fontweight='bold' if k == 'aware' else 'normal', annotation_clip=False)
    ax.set_xticks(x); ax.set_xticklabels([f'{b} ×' for b in bins]); ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f'{v} %' for v in (0, 25, 50, 75, 100)]); ax.set_xlabel(t['leak_x'])
    fig.savefig(path, dpi=200); plt.close(fig)


def water(t, rows, path):
    fig, ax = plt.subplots(figsize=(12, 6.2)); fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.10)
    heading(fig, t['water_title'], t['water_sub']); base(ax, t)
    for tr, ls in (('143 A', '-'), ('108 D', '--')):
        for kind, c in (('raw', MUTED), ('clean', ACC)):
            rs = [r for r in rows if r['track'] == tr and r['kind'] == kind]
            d = [datetime.fromisoformat(r['date']) for r in rs]
            ax.plot(d, [r['db'] for r in rs], ls, color=c, lw=2.2, marker='o', ms=6,
                    label=f"{tr}, {t['clean'] if kind == 'clean' else t['raw']}")
    ax.axhline(-20, color=QUIET, ls=(0, (3, 3)), lw=1)
    ax.text(1.0, -20.6, t['ref'], transform=ax.get_yaxis_transform(), ha='right', va='top', color=QUIET, fontsize=11)
    ax.set_ylim(-22, 1); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g} {"dB" if t is T["en"] else "дБ"}'))
    months_axis(ax, t)
    ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=12, handlelength=2.6)
    fig.savefig(path, dpi=200); plt.close(fig)


def field(t, rows, path):
    fig, ax = plt.subplots(figsize=(12, 5.8)); fig.subplots_adjust(left=0.08, right=0.80, top=0.80, bottom=0.10)
    heading(fig, t['field_title'], t['field_sub']); base(ax, t)
    for kind, c, lw, lab in (('raw', MUTED, 2.2, t['field_raw']), ('clean', ACC, 3, t['field_clean'])):
        rs = [r for r in rows if r['kind'] == kind]; d = [datetime.fromisoformat(r['date']) for r in rs]; v = [r['db'] for r in rs]
        ax.plot(d, v, color=c, lw=lw, marker='o', ms=7)
        ax.annotate(lab, (d[-1], v[-1]), xytext=(14, 0), textcoords='offset points', va='center', fontsize=12.5,
                    fontweight='bold' if kind == 'clean' else 'normal', annotation_clip=False)
    ax.set_ylim(-22, -4); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g} {"dB" if t is T["en"] else "дБ"}'))
    months_axis(ax, t)
    fig.savefig(path, dpi=200); plt.close(fig)


def first_change(t, rf, path):
    fig, ax = plt.subplots(figsize=(12, 6)); fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.10)
    heading(fig, t['fc_title'], t['fc_sub']); base(ax, t)
    for k, (tr, op) in enumerate((('143_A_025', 1.0), ('108_D_065', 0.42))):
        fc = rf[tr]['перша зміна, км²']; d = [datetime.fromisoformat(x) for x in fc]; v = list(fc.values())
        ax.bar(d, v, width=2.6, color=ACC, alpha=op, label=f"{t['track']} {tr[:5].replace('_', ' ')}: {sum(v):,} {t['km']}".replace(',', ' ' if t is T['uk'] else ','))
    months_axis(ax, t); ax.set_ylim(0, 360)
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=12)
    fig.savefig(path, dpi=200); plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    app = json.loads((PROJECT / 'web' / 'data' / 'app.json').read_text(encoding='utf-8'))
    rows = json.loads((PROJECT / 'report_figures' / 'chart_rows.json').read_text(encoding='utf-8'))
    rf = json.loads((PROJECT / 'data' / 'analysis_rfi_clean' / 'change_rfiaware.json').read_text(encoding='utf-8'))
    for lang, t in T.items():
        d = PROJECT / t['dir']; f = t['files']
        kpi(t, d / f[0]); leak(t, app, d / f[1]); water(t, rows['water'], d / f[2])
        field(t, rows['field'], d / f[3]); first_change(t, rf, d / f[4])
        print(lang, 'ok')


if __name__ == '__main__':
    main()

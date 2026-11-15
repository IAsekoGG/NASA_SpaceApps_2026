"""Ілюстрації для доповіді «Каховка крізь перешкоди» → Project/report_figures/*.png.

Спершу tools/eval_rfi_clean.py і tools/run_change_rfi.py. Карти оцінки перешкоди R тут НЕ малюються
(район бойових дій): лише Sentinel-2, сирі й очищені знімки NISAR і карти змін.

    python -I tools/report_figures.py
"""
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import rasterio

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import sarstats as st  # noqa: E402

D = PROJECT / 'data' / 'analysis_rfi_clean'
OUT = PROJECT / 'report_figures'
INK, QUIET, ACCENT, RAWC = '#1f1f1d', '#6b6a63', '#2a78d6', '#d95926'
VMIN, VMAX = -22, -2
SEASON = [(172, '#5b3fd1'), (195, '#8a3fc4'), (218, '#c23f9e'), (241, '#e8606f'), (264, '#f88f45'), (287, '#fbbf2f'), (310, '#f4e44a')]
MONTHS = [(152, 'черв'), (182, 'лип'), (213, 'серп'), (244, 'вер'), (274, 'жовт'), (305, 'лист'), (334, '')]

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'text.color': INK, 'axes.edgecolor': '#c9c8c1',
                     'axes.labelcolor': QUIET, 'xtick.color': QUIET, 'ytick.color': QUIET, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})


def season_cmap():
    lo, hi = MONTHS[0][0], MONTHS[-1][0]
    stops = [((d - lo) / (hi - lo), c) for d, c in SEASON]
    return LinearSegmentedColormap.from_list('season', [(0, SEASON[0][1])] + stops + [(1, SEASON[-1][1])]), lo, hi


def db(lin):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(np.maximum(lin, 1e-6))


def clean_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, extent, km=10, color='white'):
    x0, x1, y0, y1 = extent
    w = x1 - x0
    xs, ys = x0 + 0.04 * w, y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + km], [ys, ys], color=color, lw=3, solid_capstyle='butt')
    ax.text(xs + km / 2, ys + 0.025 * (y1 - y0), f'{km} км', color=color, ha='center', va='bottom', fontsize=10,
            fontweight='bold')


def tag(ax, text, color='white', bg='#1f1f1d'):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.35', fc=bg, ec='none', alpha=0.78))


def footer(fig, text):
    fig.text(0.01, 0.008, text, ha='left', va='bottom', fontsize=8.5, color=QUIET)


def title(fig, main, sub=None, y=0.985):
    fig.text(0.01, y, main, ha='left', va='top', fontsize=17, fontweight='bold', color=INK)
    if sub:
        fig.text(0.01, y - 0.055, sub, ha='left', va='top', fontsize=11.5, color=QUIET)


def load_stack(t, kind):
    z = np.load(D / f'stack_{kind}_{t}.npz')
    x, y = z['x'], z['y']
    ext = [(x[0] - 40) / 1e3, (x[-1] + 40) / 1e3, (y[-1] - 40) / 1e3, (y[0] + 40) / 1e3]
    return z, ext


def crop_box(ext, shape, cx_km, cy_km, w_km, h_km):
    """Індекси рядків/стовпців вікна навколо центру в км і його межі в км."""
    H, W = shape
    x0, x1, y0, y1 = ext
    px = (x1 - x0) / W
    c0 = int((cx_km - w_km / 2 - x0) / px); c1 = int((cx_km + w_km / 2 - x0) / px)
    r0 = int((y1 - (cy_km + h_km / 2)) / px); r1 = int((y1 - (cy_km - h_km / 2)) / px)
    sub_ext = [x0 + c0 * px, x0 + c1 * px, y1 - r1 * px, y1 - r0 * px]
    return (slice(r0, r1), slice(c0, c1)), sub_ext


_CROP = None


def crop160():
    """Маска ріллі (WorldCover 40, суша за JRC) на сітці 160 м."""
    global _CROP
    if _CROP is None:
        sys.path.insert(0, str(PROJECT / 'tools'))
        import eval_rfi_clean as E
        WC, JRC = E.load_aux(); m = E.masks(WC, JRC)['crop'].astype('f4')
        _CROP = st.multilook(m, 2) > 0.5
    return _CROP


def edge_window(raw_db, clean_db, ext, w_km, h_km, lo=2.5, hi=5.0):
    """Центр вікна збільшення: де сирий знімок світліший за очищений на lo…hi дБ (край плями: поля вже
    видно, шум ще не домінує) і найбільше валідних пікселів. Сама оцінка перешкоди не малюється."""
    from scipy.ndimage import uniform_filter
    d = np.nan_to_num(raw_db - clean_db, nan=0.0)
    H, W = d.shape; px = (ext[1] - ext[0]) / W
    kw, kh = int(w_km / px), int(h_km / px)
    crop = crop160()[:H, :W]
    ok = ((d > lo) & (d < hi) & crop).astype('f4'); fin = np.isfinite(raw_db).astype('f4')
    score = uniform_filter(ok, (kh, kw)) * (uniform_filter(fin, (kh, kw)) > 0.99)
    score[:kh // 2 + 1] = 0; score[-kh // 2 - 1:] = 0; score[:, :kw // 2 + 1] = 0; score[:, -kw // 2 - 1:] = 0
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return ext[0] + (c + 0.5) * px, ext[3] - (r + 0.5) * px


def ml_db(lin, k=2):
    return db(st.multilook(lin, k))


def ml_ext(ext, shape_full, k=2):
    return ext


# ---------------------------------------------------------------------------------------------
def fig_s2_story():
    files = [('s2_truecolor_2023-05_full.tif', 'травень 2023 · водосховище'),
             ('s2_truecolor_2023-06_drained.tif', 'червень 2023 · після руйнування греблі'),
             ('s2_truecolor_2024-07_regrowth.tif', 'літо 2024 · дно заростає'),
             ('s2_truecolor_2026-07_forest.tif', 'літо 2026 · молодий ліс на дні')]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.885, bottom=0.04, wspace=0.02, hspace=0.04)
    for ax, (f, lab) in zip(axs.ravel(), files):
        with rasterio.open(PROJECT / 'data' / 's2_story' / f) as s:
            img = np.moveaxis(s.read(), 0, -1); b = s.bounds
        ext = [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3]
        ax.imshow(img, extent=ext, interpolation='lanczos')
        clean_axes(ax); tag(ax, lab); scalebar(ax, ext, 20)
    title(fig, 'Каховське водосховище за три роки: від моря до лісу',
          'Sentinel-2, справжні кольори, 60 м · той самий район 140 × 100 км, що й у радарному аналізі')
    footer(fig, 'Містить модифіковані дані Copernicus Sentinel (2023–2026), обробка команди.')
    fig.savefig(OUT / '01_s2_story.png', dpi=160); plt.close(fig)


def fig_blinded():
    """Радар засліплений: брудна й чиста дата одного треку + вікно з полями крізь пляму."""
    import pandas as pd
    t, dirty, clean = '143_A_025', '2026-07-06', '2026-09-04'
    z, ext = load_stack(t, 'raw'); zc, _ = load_stack(t, 'clean')
    dates = [str(d) for d in z['dates']]
    i_d, i_c = dates.index(dirty), dates.index(clean)
    q = pd.read_csv(next(D.glob('qc_by_date_direct_*.csv'))); q = q[q['трек'] == t].set_index('дата')
    raw_d, raw_c = ml_db(z['hh'][i_d]), ml_db(z['hh'][i_c])
    cx, cy = edge_window(raw_d, ml_db(zc['hh'][i_d]), ext, 20, 20, lo=3, hi=7)
    win, wext = crop_box(ext, raw_d.shape, cx, cy, 20, 20)
    fig = plt.figure(figsize=(15, 5.9))
    gs = fig.add_gridspec(1, 3, left=0.01, right=0.99, top=0.84, bottom=0.06, wspace=0.03, width_ratios=[1.35, 1.35, 1])
    for k, (img, d, c) in enumerate(((raw_d, dirty, RAWC), (raw_c, clean, ACCENT))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, extent=ext, interpolation='antialiased')
        word = 'сильна перешкода' if k == 0 else 'чиста дата'
        lab = f'{d[8:10]}.{d[5:7]} · {word} · рілля HH {q.loc[d, "HH crop сир"]:.1f} дБ'
        clean_axes(ax); tag(ax, lab, bg=c); scalebar(ax, ext, 20)
        if k == 0:
            ax.add_patch(Rectangle((wext[0], wext[2]), wext[1] - wext[0], wext[3] - wext[2], fill=False, ec='white', lw=1.8))
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(raw_d[win], cmap='gray', vmin=VMIN, vmax=VMAX, extent=wext, interpolation='nearest')
    clean_axes(ax); tag(ax, 'збільшено: поля видно крізь пляму'); scalebar(ax, wext, 5)
    title(fig, 'Радар «засліплений»: перешкода лягає поверх знімка, але не стирає його',
          'NISAR L2 GCOV, трек 143 A, HH frequencyB 160 м, шкала −22…−2 дБ')
    footer(fig, 'Дані: NASA/JPL, ISRO — NISAR L2 GCOV (PROVISIONAL). Розмиті світлі плями — радіозавади; їхні джерела не аналізуються.')
    fig.savefig(OUT / '02_radar_blinded.png', dpi=160); plt.close(fig)


def fig_before_after():
    t, date = '143_A_025', '2026-07-06'
    zr, ext = load_stack(t, 'raw'); zc, _ = load_stack(t, 'clean')
    i = [str(d) for d in zr['dates']].index(date)
    raw, cl = ml_db(zr['hh'][i]), ml_db(zc['hh'][i])
    cx, cy = edge_window(raw, cl, ext, 24, 16, lo=2.5, hi=5)
    win, wext = crop_box(ext, raw.shape, cx, cy, 24, 16)
    fig = plt.figure(figsize=(13, 10.2))
    gs = fig.add_gridspec(2, 2, left=0.01, right=0.99, top=0.885, bottom=0.04, wspace=0.03, hspace=0.05, height_ratios=[1, 0.72])
    for k, (img, lab, c) in enumerate(((raw, 'до: сирий знімок', RAWC), (cl, 'після: оцінку перешкоди віднято', ACCENT))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, extent=ext, interpolation='antialiased')
        ax.add_patch(Rectangle((wext[0], wext[2]), wext[1] - wext[0], wext[3] - wext[2], fill=False, ec='white', lw=1.8))
        clean_axes(ax); tag(ax, lab, bg=c); scalebar(ax, ext, 20)
        ax2 = fig.add_subplot(gs[1, k])
        ax2.imshow(img[win], cmap='gray', vmin=VMIN, vmax=VMAX, extent=wext, interpolation='nearest')
        clean_axes(ax2); tag(ax2, 'збільшено 24 × 16 км', bg=c); scalebar(ax2, wext, 5)
    title(fig, 'До і після: поля повертаються з-під радіотуману',
          'NISAR, трек 143 A, 06.07.2026, HH frequencyB 160 м, однакова шкала −22…−2 дБ для всіх панелей')
    footer(fig, 'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Очищення: адитивна модель Y = s + R, фон оцінено по темних пікселях суші й постійній воді.')
    fig.savefig(OUT / '03_before_after.png', dpi=160); plt.close(fig)


def doy_map(ch):
    d = [datetime.fromisoformat(str(x)) for x in ch['dates']]
    mids = np.array([np.nan] + [((d[i - 1] + (d[i] - d[i - 1]) / 2).timetuple().tm_yday) for i in range(1, len(d))], 'f4')
    out = mids[ch['first']]
    out[(ch['first'] == 0) | ~ch['changed']] = np.nan
    return out


def change_ext(ch):
    x, y = ch['x'], ch['y']
    return [(x[0] - 80) / 1e3, (x[-1] + 80) / 1e3, (y[-1] - 80) / 1e3, (y[0] + 80) / 1e3]


def season_colorbar(fig, rect, label):
    cmap, lo, hi = season_cmap()
    cax = fig.add_axes(rect)
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=cmap, extent=[lo, hi, 0, 1])
    cax.set_yticks([]); cax.set_xticks([m for m, _ in MONTHS[:-1]]); cax.set_xticklabels([n for _, n in MONTHS[:-1]], fontsize=10)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.set_title(label, fontsize=10.5, color=QUIET, loc='left', pad=4)


def fig_change_map():
    t = '143_A_025'
    ch = np.load(PROJECT / 'data' / f'change_kakhovka_{t}_rfiaware.npz')
    ext = change_ext(ch); doy = doy_map(ch); bg = ch['median_clean_hh_db']
    res = json.loads((D / 'change_rfiaware.json').read_text(encoding='utf-8'))[t]
    cmap, lo, hi = season_cmap()
    win, wext = crop_box(ext, doy.shape, cx_km=ext[0] + 0.62 * (ext[1] - ext[0]), cy_km=ext[2] + 0.28 * (ext[3] - ext[2]),
                         w_km=24, h_km=18)
    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(1, 2, left=0.01, right=0.99, top=0.84, bottom=0.14, wspace=0.03, width_ratios=[1.25, 1])
    for k, (sl, e, lab) in enumerate(((np.s_[:, :], ext, f'трек 143 A · змінилось {res["змінилось, %"]} % ({res["змінилось, км²"]:,} км²)'.replace(',', ' ')),
                                      (win, wext, 'збільшено 24 × 18 км: цілі поля, одна дата'))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(bg[sl], cmap='gray', vmin=-20, vmax=-4, extent=e, interpolation='antialiased', alpha=0.9)
        ax.imshow(np.ma.masked_invalid(doy[sl]), cmap=cmap, vmin=lo, vmax=hi, extent=e, interpolation='nearest')
        if k == 0:
            ax.add_patch(Rectangle((wext[0], wext[2]), wext[1] - wext[0], wext[3] - wext[2], fill=False, ec='white', lw=1.8))
        clean_axes(ax); tag(ax, lab); scalebar(ax, e, 20 if k == 0 else 5)
    season_colorbar(fig, [0.30, 0.065, 0.40, 0.03], 'коли почався новий стан поверхні (середина проміжку між знімками)')
    title(fig, 'Карта змін: коли на кожному полі сталася перша зміна',
          'Тест змін із відомим фоном перешкоди, усі 11 дат, 160 м, FDR 5 % · тло — медіана очищеного HH')
    footer(fig, 'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Липень — збирання зернових; серпень–вересень — соняшник, оранка; жовтень — кукурудза, сівба.')
    fig.savefig(OUT / '04_change_map.png', dpi=160); plt.close(fig)


def fig_agreement():
    from scipy.ndimage import uniform_filter
    A = np.load(PROJECT / 'data' / 'change_kakhovka_143_A_025_rfiaware.npz')
    B = np.load(PROJECT / 'data' / 'change_kakhovka_108_D_065_rfiaware.npz')
    h, w = min(A['first'].shape[0], B['first'].shape[0]), min(A['first'].shape[1], B['first'].shape[1])
    da, dbm = doy_map(A)[:h, :w], doy_map(B)[:h, :w]
    valid = np.isfinite(A['p'])[:h, :w] & np.isfinite(B['p'])[:h, :w]
    both = np.isfinite(da) & np.isfinite(dbm) & (np.abs(da - dbm) <= 12)
    ext0 = change_ext(A); ext = [ext0[0], ext0[0] + w * 0.16, ext0[3] - h * 0.16, ext0[3]]
    bg = A['median_clean_hh_db'][:h, :w]
    x = json.loads((D / 'change_rfiaware.json').read_text(encoding='utf-8'))['два треки']['рілля']
    cmap, lo, hi = season_cmap()
    kw, kh = int(26 / 0.16), int(18 / 0.16)
    score = uniform_filter(both.astype('f4'), (kh, kw)) * (uniform_filter(valid.astype('f4'), (kh, kw)) > 0.99)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    win, wext = crop_box(ext, both.shape, ext[0] + (c + 0.5) * 0.16, ext[3] - (r + 0.5) * 0.16, 26, 18)
    panels = ((da, 'трек 143 A · 02:39 UTC'), (dbm, 'трек 108 D · 16:46 UTC'),
              (np.where(both, (da + dbm) / 2, np.nan), 'збіг: обидва треки, ±12 діб'))
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, left=0.01, right=0.99, top=0.80, bottom=0.17, wspace=0.03)
    for k, (m, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(bg[win], cmap='gray', vmin=-20, vmax=-4, extent=wext, interpolation='nearest', alpha=0.9)
        ax.imshow(np.ma.masked_invalid(m[win]), cmap=cmap, vmin=lo, vmax=hi, extent=wext, interpolation='nearest')
        clean_axes(ax); tag(ax, lab, bg=ACCENT if k == 2 else '#1f1f1d'); scalebar(ax, wext, 5)
    season_colorbar(fig, [0.30, 0.075, 0.40, 0.035], 'дата першої зміни (середина проміжку між знімками)')
    lift = x['підйом (в обох / очікувано)']; ag = x['дата першої зміни збігається ±12 діб, %']; sh = x['випадково (перемішування), %']
    title(fig, f'Два незалежні треки бачать ті самі поля: у {lift:.1f} раза частіше за випадковість',
          f'Одне вікно 26 × 18 км, три погляди · дати першої зміни на ріллі збігаються в {ag:.0f} % пікселів проти {sh:.0f} % випадково')
    footer(fig, 'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Треки летять у різний час доби з різних боків і обробляються незалежно.')
    fig.savefig(OUT / '05_two_tracks_agree.png', dpi=160); plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    OUT.mkdir(exist_ok=True)
    figs = (fig_blinded, fig_before_after, fig_agreement) if '--quick' in sys.argv else (fig_s2_story, fig_blinded, fig_before_after, fig_change_map, fig_agreement)
    for f in figs:
        f(); print('готово:', f.__name__)


if __name__ == '__main__':
    main()

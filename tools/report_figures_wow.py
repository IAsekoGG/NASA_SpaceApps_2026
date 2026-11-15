"""Виразні ілюстрації для доповіді: сезонний RGB радара, діагональна шторка, таймлапс, концепт-схема,
радар проти оптики. Плюс рядки для графіків документа → report_figures/chart_rows.json.

Карт оцінки перешкоди R тут немає (район бойових дій). Концепт-схема — на синтетичних даних.

    python -I tools/report_figures_wow.py
"""
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import rasterio
from scipy.ndimage import gaussian_filter, uniform_filter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import sarstats as st  # noqa: E402

D = PROJECT / 'data' / 'analysis_rfi_clean'
OUT = PROJECT / 'report_figures'
BG, INK, QUIET = '#0e0f12', '#f2f1ec', '#a9a8a0'
BGRGB = np.array([14, 15, 18]) / 255
FONT = 'C:/Windows/Fonts/segoeui.ttf'; FONTB = 'C:/Windows/Fonts/segoeuib.ttf'
plt.rcParams.update({'font.family': 'Segoe UI', 'text.color': INK, 'figure.facecolor': BG, 'savefig.facecolor': BG})


def db(lin):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(np.maximum(lin, 1e-6))


def load(t, kind):
    z = np.load(D / f'stack_{kind}_{t}.npz')
    return z, [str(d) for d in z['dates']]


def stretch(a, lo=2, hi=98, gamma=1.0):
    v = a[np.isfinite(a)]
    p0, p1 = np.percentile(v, [lo, hi])
    out = np.clip((a - p0) / (p1 - p0), 0, 1) ** gamma
    return np.nan_to_num(out, nan=0.0)


def poster_text(fig, title, sub, foot):
    fig.text(0.02, 0.975, title, ha='left', va='top', fontsize=22, fontweight='bold', color=INK)
    fig.text(0.02, 0.915, sub, ha='left', va='top', fontsize=12.5, color=QUIET)
    fig.text(0.02, 0.015, foot, ha='left', va='bottom', fontsize=9, color=QUIET)


# ---------------------------------------------------------------------------------------------
def seasonal_rgb(t='143_A_025'):
    z, dates = load(t, 'clean')
    hh = st.multilook(z['hh'], 2)
    mon = np.array([int(d[5:7]) for d in dates])
    groups = [(6, 7), (8, 9), (10, 11)]
    ch = [db(np.nanmedian(hh[np.isin(mon, g)], 0)) for g in groups]
    valid = np.all([np.isfinite(c) for c in ch], 0)
    rgb = np.dstack([stretch(c, 2, 99, 0.9) for c in ch])
    rgb = np.where(valid[..., None], rgb, BGRGB)
    return rgb, valid


def fig_radar_rgb():
    rgb, valid = seasonal_rgb()
    fig = plt.figure(figsize=(14, 10.4))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.80]); ax.imshow(rgb, interpolation='antialiased'); ax.set_axis_off()
    # легенда кольорів
    lx = fig.add_axes([0.70, 0.925, 0.28, 0.055]); lx.set_axis_off(); lx.set_xlim(0, 3); lx.set_ylim(0, 1)
    for i, (c, lab) in enumerate((('#ff3b30', 'червень–липень'), ('#34c759', 'серпень–вересень'), ('#2f7bff', 'жовтень–листопад'))):
        lx.add_patch(plt.Rectangle((i, 0.55), 0.9, 0.35, color=c)); lx.text(i + 0.45, 0.25, lab, ha='center', va='center', fontsize=10, color=QUIET)
    poster_text(fig, 'Сезон очима радара: кожне поле має свій колір',
                'Очищений HH NISAR, трек 143 A · канал R — медіана червня–липня, G — серпня–вересня, B — жовтня–листопада\n'
                'сірий і білий — поверхня стабільна, кольори — поле змінилось',
                'Дані: NASA/JPL, ISRO — NISAR L2 GCOV, 11 дат, 160 м. Червоні поля: врожай зібрано після липня; сині: зійшли озимі або зорано восени.')
    fig.savefig(OUT / '07_radar_season_rgb.jpg', dpi=150, pil_kwargs={'quality': 90}); plt.close(fig)
    return rgb


def fig_diagonal_split(t='143_A_025', date='2026-07-06'):
    zr, dates = load(t, 'raw'); zc, _ = load(t, 'clean')
    i = dates.index(date)
    raw, cl = db(st.multilook(zr['hh'][i:i + 1], 2)[0]), db(st.multilook(zc['hh'][i:i + 1], 2)[0])
    g = lambda a: np.nan_to_num(np.clip((a + 22) / 20, 0, 1), nan=0)
    H, W = raw.shape
    yy, xx = np.mgrid[0:H, 0:W]
    C0 = 0.87
    side = (xx / W + (H - yy) / H) < C0               # нижня ліва частина — сирий знімок
    img = np.where(side, g(raw), g(cl))
    rgb = np.dstack([img] * 3)
    tint = np.array([1.0, 0.86, 0.74]); rgb = np.where(side[..., None], rgb * tint, rgb * np.array([0.82, 0.9, 1.0]))
    fin = np.isfinite(raw)
    line = (np.abs((xx / W + (H - yy) / H) - C0) < 0.0035) & fin
    glow = gaussian_filter(line.astype('f4'), 3); glow /= glow.max(); glow *= fin
    rgb = np.clip(rgb + glow[..., None] * 0.9, 0, 1)
    rgb = np.where(fin[..., None], rgb, BGRGB)
    fig = plt.figure(figsize=(14, 10.4))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.80]); ax.imshow(rgb, interpolation='antialiased'); ax.set_axis_off()
    ax.text(0.03, 0.05, 'СИРИЙ ЗНІМОК', transform=ax.transAxes, fontsize=16, fontweight='bold', color='#ffb37a')
    ax.text(0.97, 0.62, 'ПІСЛЯ ОЧИЩЕННЯ', transform=ax.transAxes, fontsize=16, fontweight='bold', color='#8fc1ff', ha='right')
    poster_text(fig, 'Один знімок, дві реальності',
                'NISAR, трек 143 A, 06.07.2026, HH 160 м · ліворуч унизу те, що надіслав супутник, праворуч угорі — після нашого пайплайна; лінія ріже пляму навпіл',
                'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Однакова шкала −22…−2 дБ для обох частин.')
    fig.savefig(OUT / '08_diagonal_split.jpg', dpi=150, pil_kwargs={'quality': 90}); plt.close(fig)


def fig_timelapse(t='143_A_025', width=1000):
    zr, dates = load(t, 'raw'); zc, _ = load(t, 'clean')
    raw, cl = st.multilook(zr['hh'], 2), st.multilook(zc['hh'], 2)
    H, W = raw.shape[1:]
    s = (width // 2 - 12) / W; w, h = int(W * s), int(H * s)
    f, fb, fs = ImageFont.truetype(FONTB, 30), ImageFont.truetype(FONTB, 20), ImageFont.truetype(FONT, 17)
    frames = []
    for i, d in enumerate(dates):
        canvas = Image.new('RGB', (width, h + 110), BG)
        for k, (a, lab, col) in enumerate(((raw[i], 'сирий', (255, 179, 122)), (cl[i], 'очищений', (143, 193, 255)))):
            g = np.nan_to_num(np.clip((db(a) + 22) / 20, 0, 1), nan=0)
            im = Image.fromarray((g * 255).astype('u1')).resize((w, h), Image.LANCZOS)
            x0 = 8 + k * (w + 8); canvas.paste(im, (x0, 70))
            ImageDraw.Draw(canvas).text((x0 + 12, 80), lab.upper(), font=fb, fill=col)
        dr = ImageDraw.Draw(canvas)
        dr.text((12, 14), f'{d[8:10]}.{d[5:7]}.{d[:4]}', font=f, fill=INK)
        dr.text((210, 24), 'NISAR, трек 143 A, HH · кожен кадр — новий проліт супутника', font=fs, fill=QUIET)
        # шкала часу
        x_a, x_b, y = 12, width - 12, h + 92
        dr.line((x_a, y, x_b, y), fill=(70, 70, 76), width=3)
        for j in range(len(dates)):
            xx = x_a + (x_b - x_a) * j / (len(dates) - 1)
            r = 8 if j == i else 4
            dr.ellipse((xx - r, y - r, xx + r, y + r), fill=(143, 193, 255) if j == i else (110, 110, 118))
        frames.append(canvas)
    pal = [f.convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames]
    pal[0].save(OUT / '09_timelapse.gif', save_all=True, append_images=pal[1:], duration=900, loop=0, optimize=True)
    frames[4].save(OUT / '09_timelapse_frame.png')


def fig_concept():
    """Синтетична ілюстрація моделі Y = s + R: поверхня, завада, знімок, очищення."""
    rng = np.random.default_rng(3)
    H, W = 220, 300
    s = np.zeros((H, W))
    y = 0
    while y < H:                                    # прямокутні «поля» різної яскравості
        hgt = rng.integers(18, 40); x = 0
        while x < W:
            wd = rng.integers(20, 60); s[y:y + hgt, x:x + wd] = 10 ** (rng.uniform(-19, -9) / 10); x += wd
        y += hgt
    yy, xx = np.mgrid[0:H, 0:W]
    R = 0.35 * np.exp(-(((xx - 190) / 60) ** 2 + ((yy - 120) / 50) ** 2))
    speck = lambda m: rng.gamma(20, 1 / 20, m.shape) * m
    Y = speck(s + R)
    Rhat = gaussian_filter(R, 4) * 1.0
    C = np.maximum(Y - Rhat, 1e-4)
    panels = [(s, 'поверхня s', 'поля різної яскравості'), (R, 'завада R', 'гладка пляма, додається'),
              (Y, 'знімок Y = s + R', 'що бачить супутник'), (C, 'очищення: Y − оцінка R', 'поля повернулись')]
    fig = plt.figure(figsize=(15, 5.2))
    for k, (a, t1, t2) in enumerate(panels):
        ax = fig.add_axes([0.02 + k * 0.245, 0.10, 0.205, 0.62])
        if k == 1:
            ax.imshow(a, cmap='magma', vmin=0, vmax=0.4)
        else:
            ax.imshow(db(a), cmap='gray', vmin=-22, vmax=-2)
        ax.set_axis_off()
        fig.text(0.02 + k * 0.245, 0.80, t1, fontsize=15, fontweight='bold', color=INK)
        fig.text(0.02 + k * 0.245, 0.745, t2, fontsize=11, color=QUIET)
        if k < 3:
            fig.text(0.02 + k * 0.245 + 0.2225, 0.41, ['+', '=', '→'][k], fontsize=34, fontweight='bold', color='#8fc1ff', ha='center', va='center')
    fig.text(0.02, 0.95, 'Ідея методу в одному рядку', fontsize=20, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.03, 'Синтетична ілюстрація: поля, спекл і пляма згенеровані, а не взяті з реального знімка.', fontsize=9, color=QUIET)
    fig.savefig(OUT / '10_concept.png', dpi=150); plt.close(fig)


def fig_radar_vs_optics(rgb):
    with rasterio.open(PROJECT / 'data' / 's2_story' / 's2_truecolor_2026-07_forest.tif') as s:
        opt = np.moveaxis(s.read(), 0, -1)
    fig = plt.figure(figsize=(15, 6.4))
    for k, (img, lab) in enumerate(((opt, 'оптика: Sentinel-2, літо 2026'), (rgb, 'радар: NISAR, сезонний RGB'))):
        ax = fig.add_axes([0.01 + k * 0.495, 0.07, 0.485, 0.74]); ax.imshow(img, interpolation='antialiased', aspect='auto'); ax.set_axis_off()
        ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=13, fontweight='bold', color=INK, va='top',
                bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec='none', alpha=0.75))
    poster_text(fig, 'Оптика бачить колір, радар бачить зміну',
                'Той самий район 140 × 100 км · радар працює вночі й крізь хмари, а сезонний RGB показує, коли кожне поле змінилось',
                'Дані: Copernicus Sentinel-2 (модифіковані); NASA/JPL, ISRO — NISAR L2 GCOV.')
    fig.savefig(OUT / '11_radar_vs_optics.jpg', dpi=150, pil_kwargs={'quality': 90}); plt.close(fig)


# ---------------------------------------------------------------------------------------------
def chart_rows():
    import pandas as pd
    rows = {}
    q = pd.read_csv(next(D.glob('qc_by_date_direct_*.csv')))
    rows['water'] = [{'track': r['трек'][:5].replace('_', ' '), 'date': r['дата'], 'kind': k,
                      'db': round(float(r[f'HV вода {kk}']), 1)}
                     for _, r in q.iterrows() for k, kk in (('raw', 'сир'), ('clean', 'чист')) if pd.notna(r[f'HV вода {kk}'])]
    ch = json.loads((D / 'change_rfiaware.json').read_text(encoding='utf-8'))
    rows['first_change'] = [{'track': t[:5].replace('_', ' '), 'date': d, 'km2': v} for t in ('143_A_025', '108_D_065')
                            for d, v in ch[t]['перша зміна, км²'].items()]
    # один «типовий» фрагмент поля на краю плями: часовий ряд HH сирий і очищений
    zr, dates = load('143_A_025', 'raw'); zc, _ = load('143_A_025', 'clean')
    raw, cl = st.multilook(zr['hh'], 2), st.multilook(zc['hh'], 2)
    sys.path.insert(0, str(PROJECT / 'tools'))
    import eval_rfi_clean as E
    WC, JRC = E.load_aux(); crop = st.multilook(E.masks(WC, JRC)['crop'].astype('f4'), 2) > 0.9
    def nanmean3(x):
        f = np.isfinite(x)
        return np.where(f, uniform_filter(np.where(f, x, 0), 3) / np.maximum(uniform_filter(f.astype('f4'), 3), 1e-6), np.nan)
    rm, cm = np.stack([nanmean3(x) for x in raw]), np.stack([nanmean3(x) for x in cl])
    rdb, cdb = db(rm), db(cm)
    i7 = dates.index('2026-07-06')
    excess = rdb[i7] - cdb[i7]                       # де 06.07 сирий найсильніше відрізняється від очищеного
    drop = np.nanmax(cdb[:4], 0) - np.nanmin(cdb[4:], 0)   # справжнє падіння після липня (збирання)
    stable = np.nanstd(cdb[4:], 0)
    allfin = np.isfinite(rdb).all(0) & np.isfinite(cdb).all(0)
    rough = np.nanmean(np.abs(np.diff(rdb, axis=0) - np.diff(cdb, axis=0)), 0)   # наскільки сирий ряд «стрибає» відносно очищеного
    smooth = np.nanstd(np.diff(cdb, axis=0), 0)
    score = np.where(crop & allfin & (drop > 3) & (drop < 9) & (np.nanmin(cdb, 0) > -21) & (smooth < 2.0), rough - smooth, -99)
    print('кандидатів:', int((score > -99).sum()))
    r, c = np.unravel_index(np.argmax(score), score.shape)
    rows['field'] = [{'date': d, 'kind': k, 'db': round(float(a[j, r, c]), 2)} for j, d in enumerate(dates)
                     for k, a in (('raw', rdb), ('clean', cdb))]
    (OUT / 'chart_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('поле:', r, c, 'падіння', round(float(drop[r, c]), 1), 'дБ')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    OUT.mkdir(exist_ok=True)
    rgb = fig_radar_rgb(); print('rgb')
    fig_diagonal_split(); print('split')
    fig_timelapse(); print('timelapse')
    fig_concept(); print('concept')
    fig_radar_vs_optics(rgb); print('optics')
    chart_rows(); print('rows')


if __name__ == '__main__':
    main()

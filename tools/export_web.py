"""Експорт результатів Каховки для вебзастосунку Project/web/.

Читає (спершу запустіть tools/eval_rfi_clean.py і tools/run_change_rfi.py):
  data/analysis_rfi_clean/stack_raw_<трек>.npz, stack_clean_<трек>.npz, qc_by_date_*.csv, contrast_pairs_*.csv,
  change_rfiaware.json, change_leak_143_A_025.json; data/change_kakhovka_<трек>_rfiaware.npz.
Пише:
  web/layers/<трек>/<дата>_raw.png, <дата>_clean.png — HH, 160 м, Web Mercator, сірий −22…−2 дБ;
  web/layers/<трек>/change.png — дата першої зміни (палітра в app.json);
  web/layers/agree.png — зміна в обох треках з датами в межах ±12 діб;
  web/data/app.json — кути зображень (lon/lat), дати, палітри, усі числа для графіків.

Перешкода показується лише як якість даних: тонування сирого знімка (частка перешкоди в
спостереженій потужності) і маска пошкоджених пікселів. Джерел перешкод не локалізуємо й не підписуємо.

    python -I tools/export_web.py
"""
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as warp_xy

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import sarstats as st  # noqa: E402

D = PROJECT / 'data' / 'analysis_rfi_clean'
WEB = PROJECT / 'web'
TRACKS = {'143_A_025': 'висхідний, ~02:39 UTC', '108_D_065': 'низхідний, ~16:46 UTC'}
LOOKS = 2
VMIN, VMAX = -22.0, -2.0
# дата першої зміни: послідовна шкала сезону (червень → листопад), однакова для обох треків за днем року
SEASON = [(172, '#5b3fd1'), (195, '#8a3fc4'), (218, '#c23f9e'), (241, '#e8606f'), (264, '#f88f45'), (287, '#fbbf2f'), (310, '#f4e44a')]


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def season_rgb(doy):
    d = [s[0] for s in SEASON]; c = np.array([hex2rgb(s[1]) for s in SEASON], 'f4')
    return np.stack([np.interp(doy, d, c[:, k]) for k in range(3)], -1)


def warp(arr, src_tf, crs, resampling, dst=None):
    """Перепроєктування в EPSG:3857. Повертає масив, трансформ і кути lon/lat (ЛВ, ПВ, ПН, ЛН)."""
    h, w = arr.shape
    if dst is None:
        tf, W, H = calculate_default_transform(crs, 'EPSG:3857', w, h, *array_bounds(h, w, src_tf))
        dst = (tf, W, H)
    tf, W, H = dst
    out = np.full((H, W), np.nan, 'f4')
    reproject(arr.astype('f4'), out, src_transform=src_tf, src_crs=crs, dst_transform=tf, dst_crs='EPSG:3857',
              resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out, dst


def corners(dst):
    tf, W, H = dst
    xs = [tf.c, tf.c + W * tf.a, tf.c + W * tf.a, tf.c]
    ys = [tf.f, tf.f, tf.f + H * tf.e, tf.f + H * tf.e]
    lon, lat = warp_xy('EPSG:3857', 'EPSG:4326', xs, ys)
    return [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]


def save_gray(db, path):
    v = np.clip((db - VMIN) / (VMAX - VMIN), 0, 1)
    g = np.nan_to_num(v * 255).astype('u1'); a = np.where(np.isfinite(db), 255, 0).astype('u1')
    Image.fromarray(np.dstack([g, a]), 'LA').save(path, optimize=True)


RATIO_DEGRADED, RATIO_BAD = 2.0, 4.0     # R / сигнал: похибка оцінки R ~10 % дає ~0.8 дБ при ×2 і ~1.5 дБ при ×4
TINT = [(0.0, (255, 214, 102)), (0.45, (250, 140, 50)), (0.75, (235, 64, 52)), (1.0, (214, 30, 110))]   # сила перешкоди → колір
CLASS_STATS = {}


def save_tinted(db, ratio, path):
    """Сирий HH сірим + тепле тонування осередків перешкоди.
    ratio = R / сигнал (згладжений): колір з'являється від ×1, повний — від ×16 (логарифмічно)."""
    g = np.nan_to_num(np.clip((db - VMIN) / (VMAX - VMIN), 0, 1))
    lr = np.log2(np.maximum(np.nan_to_num(ratio), 1e-3))
    u = np.clip(lr / 4.0, 0, 1)                                     # ×1 → 0, ×16 → 1
    a = np.clip(lr / 1.5, 0, 1) * (0.55 + 0.45 * u)                  # плавний вхід від ×1 до ×2.8
    xs = [t[0] for t in TINT]; cols = np.array([t[1] for t in TINT], 'f4')
    tint = np.stack([np.interp(u, xs, cols[:, k]) for k in range(3)], -1) / 255
    shade = (0.5 + 0.7 * g)[..., None]                              # текстура поверхні проступає крізь тонування
    rgb = (1 - a[..., None]) * g[..., None] + a[..., None] * np.clip(tint * shade, 0, 1)
    save_rgba(rgb * 255, np.where(np.isfinite(db), 255, 0), path)


def save_hatch(lvl, path):
    """Штрихування пошкоджених даних: рідке — «знижена якість», густе — «непридатні»."""
    H, W = lvl.shape
    yy, xx = np.mgrid[0:H, 0:W]
    diag = (xx + yy) % 10
    on1 = (lvl == 1) & (diag == 0)
    on2 = (lvl == 2) & ((diag == 0) | (diag == 5))
    rgb = np.zeros((H, W, 3), 'f4'); rgb[...] = (255, 110, 100)
    a = np.where(on2, 190, np.where(on1, 120, np.where(lvl == 2, 38, 0)))
    save_rgba(rgb, a, path)


CLASS_DEF = [('cropland', '#3987e5'), ('grassland', '#199e70'), ('trees', '#0ca30c'), ('builtup', '#d95926'),
             ('bed', '#c98500'), ('water', '#9085e9')]


def class_stats(raw, cl, Z, t, dates):
    """Середня потужність (дБ) по класах поверхні на кожну дату: сирий і очищений HH, HV,
    частка пікселів класу з пошкодженими даними (R > 2 × сигнал) і частка «змінилось»."""
    import eval_rfi_clean as E
    if 'M' not in CLASS_STATS:
        WC, JRC = E.load_aux(); M = E.masks(WC, JRC)
        E.add_probe(M, JRC, [E.load_track(x) for x in TRACKS])
        CLASS_STATS['M'] = {'cropland': M['crop'], 'grassland': (WC == 30) & M['land'], 'trees': (WC == 10) & M['land'],
                            'builtup': M['urban'], 'bed': M['bed'] & ~M['probe'], 'water': M['probe']}
    masks = CLASS_STATS['M']
    out = {}
    def lmean(a, m):
        v = a[m]; v = v[np.isfinite(v)]
        return round(float(10 * np.log10(np.mean(v))), 2) if v.size > 50 else None
    sig = {p: np.nanmedian(cl[p], 0) for p in ('hh', 'hv')}
    ch = np.load(PROJECT / 'data' / f'change_kakhovka_{t}_rfiaware.npz')
    h2, w2 = ch['changed'].shape
    for c, _ in CLASS_DEF:
        m = masks[c]
        rec = {'area_km2': round(float(m.sum() * 0.0064))}
        for p in ('hh', 'hv'):
            R = Z[f'{t}_{p}_R'].astype('f4')
            rec[f'{p}_raw'] = [lmean(raw[p][i], m) for i in range(len(dates))]
            rec[f'{p}_clean'] = [lmean(np.maximum(cl[p][i], 1e-6), m) for i in range(len(dates))]
            rec[f'{p}_corrupt'] = [round(100 * float((R[i][m] > RATIO_DEGRADED * sig[p][m]).mean()), 1) for i in range(len(dates))]
        mm = m[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).mean((1, 3)) > 0.5
        v = np.isfinite(ch['p']) & mm
        rec['changed_pct'] = round(100 * float(ch['changed'][v].mean()), 1) if v.sum() > 100 else None
        out[c] = rec
    return out


def save_rgba(rgb, alpha, path):
    Image.fromarray(np.dstack([np.nan_to_num(rgb).astype('u1'), alpha.astype('u1')]), 'RGBA').save(path, optimize=True)


def doy_of(dates):
    d = [datetime.fromisoformat(x) for x in dates]
    return [0] + [int((d[i - 1] + (d[i] - d[i - 1]) / 2).timetuple().tm_yday) for i in range(1, len(d))]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore', category=RuntimeWarning)
    app = {'generated': datetime.now().strftime('%Y-%m-%d %H:%M'), 'tracks': {}, 'season_palette': SEASON,
           'gray_range_db': [VMIN, VMAX]}
    dst_common, doy_maps, chg_maps = None, {}, {}
    sys.path.insert(0, str(PROJECT / 'tools'))
    Z = np.load(D / 'clean_stack_freqB.npz')
    for t, label in TRACKS.items():
        out = WEB / 'layers' / t; out.mkdir(parents=True, exist_ok=True)
        raw = np.load(D / f'stack_raw_{t}.npz'); cl = np.load(D / f'stack_clean_{t}.npz')
        dates = [str(x) for x in raw['dates']]
        crs = f"EPSG:{int(raw['epsg'])}"
        x, y = raw['x'], raw['y']
        src_tf = from_origin(x[0] - 40, y[0] + 40, 80 * LOOKS, 80 * LOOKS)
        rawml = st.multilook(raw['hh'], LOOKS); clml = st.multilook(cl['hh'], LOOKS)
        Rml = st.multilook(Z[f'{t}_hh_R'].astype('f4'), LOOKS)
        sig = np.nanmedian(clml, 0)                     # сигнал поверхні: багатодатна медіана очищеної потужності
        from scipy.ndimage import gaussian_filter, uniform_filter
        w_ = np.isfinite(sig).astype('f4')
        sig_s = gaussian_filter(np.nan_to_num(sig), 6) / np.maximum(gaussian_filter(w_, 6), 1e-3)
        files = []
        for i, d in enumerate(dates):
            fin = np.isfinite(rawml[i])
            # 1) сирий знімок із тонуванням: частка перешкоди в спостереженій потужності R / Y
            db = np.where(fin, 10 * np.log10(np.maximum(rawml[i], 1e-6)), np.nan)
            # сила перешкоди відносно згладженого (~1 км) сигналу поверхні: тонування за осередками, не за окремими полями
            rs = np.where(fin, Rml[i] / np.maximum(sig_s, 1e-9), np.nan)
            wdb, dst_common = warp(db, src_tf, crs, Resampling.bilinear, dst_common)
            wrs, _ = warp(rs, src_tf, crs, Resampling.bilinear, dst_common)
            save_tinted(wdb, wrs, out / f'{d}_rawrfi.png')
            save_gray(wdb, out / f'{d}_raw.png')          # оригінальний знімок без підсвітки
            # 2) очищений знімок
            dbc = np.where(fin, 10 * np.log10(np.maximum(clml[i], 1e-6)), np.nan)
            wdc, _ = warp(dbc, src_tf, crs, Resampling.bilinear, dst_common)
            save_gray(wdc, out / f'{d}_clean.png')
            # 3) маска пошкоджених даних: перешкода в k разів сильніша за сигнал
            ratio = np.where(fin, uniform_filter(np.nan_to_num(Rml[i] / np.maximum(sig, 1e-9)), 3), np.nan)
            lvl = np.select([ratio > RATIO_BAD, ratio > RATIO_DEGRADED], [2, 1], 0).astype('f4'); lvl[~fin] = np.nan
            wl, _ = warp(lvl, src_tf, crs, Resampling.nearest, dst_common)
            save_hatch(wl, out / f'{d}_mask.png')
            files.append({'date': d, 'raw': f'layers/{t}/{d}_raw.png', 'rawrfi': f'layers/{t}/{d}_rawrfi.png', 'clean': f'layers/{t}/{d}_clean.png',
                          'mask': f'layers/{t}/{d}_mask.png',
                          'degraded': round(100 * float((lvl[fin] == 1).mean()), 1), 'bad': round(100 * float((lvl[fin] == 2).mean()), 1)})
        CLASS_STATS[t] = class_stats(raw, cl, Z, t, dates)
        ch = np.load(PROJECT / 'data' / f'change_kakhovka_{t}_rfiaware.npz')
        doy = np.array(doy_of(dates), 'f4')[ch['first']]
        doy[(ch['first'] == 0) | ~ch['changed']] = np.nan
        src_tf2 = from_origin(ch['x'][0] - 80, ch['y'][0] + 80, 160, 160)
        wdoy, _ = warp(doy, src_tf2, crs, Resampling.nearest, dst_common)
        save_rgba(season_rgb(np.nan_to_num(wdoy, nan=SEASON[0][0])), np.where(np.isfinite(wdoy), 235, 0), out / 'change.png')
        doy_maps[t] = wdoy
        valid, _ = warp(np.isfinite(ch['p']).astype('f4'), src_tf2, crs, Resampling.nearest, dst_common)
        chg_maps[t] = (np.isfinite(wdoy), valid > 0.5)
        app['tracks'][t] = {'label': label, 'dates': dates, 'images': files, 'change': f'layers/{t}/change.png',
                            'change_doy': doy_of(dates)}
        print(t, len(dates), 'дат експортовано')
    app['bounds'] = corners(dst_common)
    app['classes'] = {'order': [c for c, _ in CLASS_DEF], 'colors': dict(CLASS_DEF), **{t: CLASS_STATS[t] for t in TRACKS}}
    app['mask_thresholds'] = [RATIO_DEGRADED, RATIO_BAD]

    a, b = doy_maps['143_A_025'], doy_maps['108_D_065']
    both = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= 12)
    save_rgba(season_rgb(np.nan_to_num((a + b) / 2, nan=SEASON[0][0])), np.where(both, 240, 0), WEB / 'layers' / 'agree.png')
    app['agree'] = 'layers/agree.png'

    # ---- числа для графіків -------------------------------------------------------------
    qc = pd.read_csv(next(D.glob('qc_by_date_direct_*.csv')))
    app['qc'] = {t: {'dates': q['дата'].tolist(),
                     **{f'{p}_{z}_{k}': [None if pd.isna(v) else round(float(v), 2) for v in q[f'{p.upper()} {zz} {kk}']]
                        for p in ('hh', 'hv') for z, zz in (('water', 'вода'), ('urban', 'urban'), ('crop', 'crop'))
                        for k, kk in (('raw', 'сир'), ('clean', 'чист'))}}
                 for t, q in qc.groupby('трек')}
    cp = pd.read_csv(next(D.glob('contrast_pairs_direct_*.csv')))
    g = cp.groupby(['зона', 'трек', 'поляр.'])[['зсув середніх сир, дБ', 'зсув середніх чист, дБ']].median().round(2)
    app['contrast'] = [{'zone': z, 'track': t, 'pol': p, 'raw': float(r.iloc[0]), 'clean': float(r.iloc[1])} for (z, t, p), r in g.iterrows()]
    rf = json.loads((D / 'change_rfiaware.json').read_text(encoding='utf-8'))
    app['change'] = {t: {k: rf[t][k] for k in ('ENL', 'змінилось, %', 'змінилось, км²', 'валідно, км²', 'перша зміна, км²',
                                               'частка «змінилось» за R/сигнал HH, %', 'дно проти DSWx-S1')} for t in TRACKS}
    app['cross_track'] = rf['два треки']
    leak = json.loads((D / 'change_leak_143_A_025.json').read_text(encoding='utf-8'))
    app['naive_leak_143'] = {r['варіант']: {'changed': r['змінилось, %'], 'by_ratio': r['частка «змінилось» за R/сигнал']} for r in leak}
    # той самий розподіл за силою перешкоди для трьох тестів на 143_A: сирі дані, наївний omnibus на очищених, з урахуванням R
    bins = [(0, .5, '< 0.5'), (.5, 1, '0.5–1'), (1, 2, '1–2'), (2, 4, '2–4'), (4, 10, '4–10'), (10, np.inf, '> 10')]
    ratio = np.load(PROJECT / 'data' / 'change_kakhovka_143_A_025_rfiaware.npz')['ratio']
    tests = {}
    for key, f in (('raw', 'change_kakhovka_143_A_025_raw.npz'), ('naive', 'change_kakhovka_143_A_025_clean.npz'),
                   ('aware', 'change_kakhovka_143_A_025_rfiaware.npz')):
        z = np.load(PROJECT / 'data' / f); v = np.isfinite(z['p']); c = z['changed']
        h, w = c.shape; r = ratio[:h, :w]
        tests[key] = {'total': round(100 * float(c[v].mean()), 1),
                      'by_ratio': [round(100 * float(c[v & (r >= lo) & (r < hi)].mean()), 1) for lo, hi, _ in bins],
                      'n': [int((v & (r >= lo) & (r < hi)).sum()) for lo, hi, _ in bins]}
    app['leak_143'] = {'bins': [b[2] for b in bins], 'tests': tests}
    wi = pd.read_csv(next(D.glob('water_iou_direct_*.csv')))
    app['water_iou'] = {t: {k: round(float(v), 3) for k, v in q.groupby('дані')['IoU'].median().items()} for t, q in wi.groupby('трек')}
    (WEB / 'data').mkdir(exist_ok=True)
    (WEB / 'data' / 'app.json').write_text(json.dumps(app, ensure_ascii=False, indent=1), encoding='utf-8')
    tot = sum(f.stat().st_size for f in (WEB / 'layers').rglob('*.png'))
    print(f'Шари: {tot / 2**20:.1f} МБ; кути: {app["bounds"]}')


if __name__ == '__main__':
    main()

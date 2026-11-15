"""Перевірка очищення RFI на Каховці: два треки, 80 м, frequencyB.

Читає кеш data/cache_ts (його будує ноутбук nisar_kakhovka_timeseries.ipynb), чистить кожен трек
pipeline/rfi_clean.py і рахує метрики, які в оцінку перешкод НЕ входили:
  1) рівень над водою: справжній сигнал < −20 дБ. Якщо вода використовується як опора
     (--probe water), рівень рахується по черзі на ділянках русла, які в оцінку не входили
     (просторові фолди ~16 км);
  2) стабільність медіани забудови між датами;
  3) контраст полів: ширина розподілу (IQR) ріллі на «брудній» даті проти найближчої чистої
     дати того самого треку. Адитивна перешкода стискає контраст у дБ (IQR-відношення < 1);
  4) вода проти DSWx-S1 лише в коридорі русла й дна (JRC ≥ 1 %): на решті суші DSWx-S1
     сам помиляється на зораних полях. Поріг HH навчається на інших датах.

Узгодженість двох треків як доказ НЕ використовуємо: плями перешкод прив'язані до місцевості
й корелюють між треками (r ≈ 0.6–0.9 на парних датах).

Запуск:  python -I tools/eval_rfi_clean.py [--probe water] [--q 10] [--blk 20]
"""
from __future__ import annotations

import argparse, json, re, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.ndimage import uniform_filter, median_filter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import rfi_clean as rc  # noqa: E402

DATA = PROJECT / 'data'
CACHE = DATA / 'cache_ts'
TRACKS = ('143_A_025', '108_D_065')
BBOX, EPSG, RES = (33.4, 46.9, 35.2, 47.8), 32636, 80.0

# ---- сітка, як у ноутбуці -------------------------------------------------------------
_t = Transformer.from_crs(4326, EPSG, always_xy=True)
_X, _Y = _t.transform([BBOX[0], BBOX[0], BBOX[2], BBOX[2]], [BBOX[1], BBOX[3], BBOX[1], BBOX[3]])
X = np.arange(np.floor((min(_X) - 40) / 80) * 80 + 40, max(_X), RES)
Y = np.arange(np.ceil((max(_Y) - 40) / 80) * 80 + 40, min(_Y), -RES)
H, W = len(Y), len(X)
TRANSFORM = from_origin(X[0] - RES / 2, Y[0] + RES / 2, RES, RES)
EXTENT = [(X[0] - 40) / 1e3, (X[-1] + 40) / 1e3, (Y[-1] - 40) / 1e3, (Y[0] + 40) / 1e3]
PIX_KM2 = RES ** 2 / 1e6


def to_grid(src_path, resampling=Resampling.nearest, nodata=255):
    dst = np.full((H, W), nodata, 'u1')
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=TRANSFORM, dst_crs=f'EPSG:{EPSG}',
                  src_nodata=src.nodata if src.nodata is not None else nodata, dst_nodata=nodata,
                  resampling=resampling)
    return dst


def load_aux():
    wc = to_grid(DATA / 'aux' / 'worldcover_2021_aoi.tif', Resampling.mode, nodata=0)
    jrc = to_grid(DATA / 'aux' / 'jrc_occurrence_aoi.tif', Resampling.average, nodata=255).astype('f4')
    jrc[jrc > 100] = np.nan
    return wc, jrc


def masks(WC, JRC):
    j = np.nan_to_num(JRC, nan=0)
    M = {'land': (j < 5) & (WC != 80), 'bed': j >= 50, 'corridor': j >= 1}
    M['urban'] = (WC == 50) & M['land']
    M['crop'] = (WC == 40) & M['land']
    return M


def add_probe(M, JRC, tracks):
    """Постійна відкрита вода ЗАРАЗ: вода за DSWx-S1 на ≥ 80 % дат обох треків, історично вода
    (JRC ≥ 80) і ≥ 2 пікселі від берега. JRC сам по собі не годиться: це все колишнє водосховище."""
    d = np.concatenate([S['dswx'] for S in tracks])
    seen = (d < 250).sum(0); wet = (d == 1).sum(0)
    w = (seen >= 6) & (wet >= 0.8 * seen) & (np.nan_to_num(JRC, nan=0) >= 80)
    M['probe'] = w & (uniform_filter(w.astype('f4'), 5) > 0.99)
    return M


def load_track(t, freq='B'):
    files = sorted(CACHE.glob(f'{t}_*.npz'))
    S = {'time': [], 'hh': [], 'hv': [], 'dswx': []}
    for f in files:
        z = np.load(f)
        tt = datetime.strptime(re.search(r'_(\d{8}T\d{4})\.npz', f.name).group(1), '%Y%m%dT%H%M')
        S['time'].append(tt)
        S['hh'].append(z[f'hh{freq}'].astype('f4')); S['hv'].append(z[f'hv{freq}'].astype('f4'))
        d = CACHE / f'dswx_s1_{tt:%Y%m%dT%H%M}.npz'
        S['dswx'].append(np.load(d)['mos'] if d.exists() else np.full((H, W), 255, 'u1'))
    for k in ('hh', 'hv', 'dswx'):
        S[k] = np.stack(S[k])
    return S


def folds(blk, k=4, tile=8):
    """Просторові фолди: квадрати tile×tile блоків (~16 км) по колу."""
    by, bx = np.meshgrid(np.arange(H) // (blk * tile), np.arange(W) // (blk * tile), indexing='ij')
    return (by + bx) % k


def pairs(ta, tb, max_days=4):
    out = []
    for i, a in enumerate(ta):
        j = int(np.argmin([abs((a - b).total_seconds()) for b in tb]))
        if abs((a - tb[j]).total_seconds()) <= max_days * 86400:
            out.append((i, j))
    return out


def ml_db(db, k=5):
    """Багатовигляд k×k у лінійній потужності → дБ; пропуски лишаються пропусками."""
    fin = np.isfinite(db)
    l = np.where(fin, rc.to_lin(np.nan_to_num(db, nan=-40)), 0)
    n = uniform_filter(fin.astype('f4'), k)
    out = rc.to_db(uniform_filter(l, k) / np.maximum(n, 1e-6))
    out[n < 0.6] = np.nan
    return out


def lmean_db(v):
    return float(rc.to_db(np.nanmean(rc.to_lin(v))))


def iqr(v):
    p = np.nanpercentile(v, [25, 75]); return p[1] - p[0]


def best_threshold(xs, ys):
    grid = np.arange(-30, -3, 0.25)
    tp = np.zeros_like(grid); fp = np.zeros_like(grid); fn = np.zeros_like(grid)
    for x, y in zip(xs, ys):
        for k, g in enumerate(grid):
            p = x < g
            tp[k] += (p & y).sum(); fp[k] += (p & ~y).sum(); fn[k] += (~p & y).sum()
    return grid[int(np.argmax(tp / np.maximum(tp + fp + fn, 1)))]


def clean_track(S, M, a):
    """Чистить HH і HV треку. При --probe water додатково рахує R без кожного фолду води.
    --hh-mode from-hv: R_hh = κ · R_hv (rfi_clean.estimate_rfi_pair); direct: кожен канал окремо."""
    F = folds(a.blk)
    probe = M['probe'] if a.probe == 'water' else None
    kw = dict(blk=a.blk, q=a.q, sigma_blk=a.sigma, floor=a.floor, method=a.method, seasonal=a.seasonal, dark_frac=a.dark,
              probe_min_frac=a.probe_min_frac, probe_floor_sigma=a.probe_floor_sigma)
    lin = {pol: rc.to_lin(S[pol]) for pol in ('hh', 'hv')}
    gain = {'hh': a.gain * a.gain_hh, 'hv': a.gain * a.gain_hv}

    def estimate(pr):
        if a.hh_mode == 'from-hv':
            Rh, Rv = rc.estimate_rfi_pair(lin['hh'], lin['hv'], M['land'], probe=pr, kappa_sigma=a.kappa_sigma, **kw)
            return {'hh': Rh, 'hv': Rv}
        return {pol: rc.estimate_rfi(lin[pol], M['land'], probe=pr, **kw) for pol in ('hh', 'hv')}

    R = estimate(probe)
    for pol in ('hh', 'hv'):
        Rp = R[pol] * gain[pol]
        c, neg = rc.clean(lin[pol], Rp)
        S[f'{pol}_R'], S[f'{pol}_c'], S[f'{pol}_neg'] = Rp, rc.to_db(c), neg
        S[f'{pol}_hold'] = S[f'{pol}_c'] if probe is None else np.full_like(S[pol], np.nan)
    if probe is not None:          # чесна перевірка водою: фолд k очищуємо оцінкою без води фолду k
        for k in range(4):
            Rk = estimate(probe & (F != k)); sel = F == k
            for pol in ('hh', 'hv'):
                S[f'{pol}_hold'][:, sel] = rc.to_db(rc.clean(lin[pol], Rk[pol] * gain[pol])[0])[:, sel]
    return S


def evaluate(T, M, a):
    WC_core = M['probe']
    rows = []
    for t, S in T.items():
        for i, tt in enumerate(S['time']):
            core = WC_core & np.isfinite(S['hh'][i])
            r = {'трек': t, 'дата': f'{tt:%Y-%m-%d}'}
            for pol in ('hh', 'hv'):
                P = pol.upper()
                # середнє в лінійній потужності: після віднімання окремі пікселі шумлять навколо нуля,
                # і медіана в дБ зміщується донизу через обрізання; середнє — незміщене
                lmean = lambda v: float(rc.to_db(np.nanmean(rc.to_lin(v))))
                r[f'{P} вода сир'] = lmean(S[pol][i][core])
                r[f'{P} вода чист'] = lmean(S[f'{pol}_hold'][i][core])
                r[f'{P} вода медіана чист'] = np.nanmedian(S[f'{pol}_hold'][i][core])
                for z in ('urban', 'crop'):
                    r[f'{P} {z} сир'] = lmean(S[pol][i][M[z]])
                    r[f'{P} {z} чист'] = lmean(S[f'{pol}_c'][i][M[z]])
                r[f'R {P} p95, дБ'] = float(rc.to_db(np.nanpercentile(S[f'{pol}_R'][i][M['land']], 95)))
                r[f'{P} <0, %'] = 100 * S[f'{pol}_neg'][i][M['land']].mean()
            rows.append(r)
    QC = pd.DataFrame(rows)

    # контраст полів на брудних датах проти чистої дати того самого треку
    cr = []
    for t, S in T.items():
        n = len(S['time'])
        for pol in ('hh', 'hv'):
            Rdb = rc.to_db(S[f'{pol}_R'])
            raw = [ml_db(S[pol][i]) for i in range(n)]; cl = [ml_db(S[f'{pol}_c'][i]) for i in range(n)]
            for i in range(n):
                for j in range(n):
                  if i == j or abs((S['time'][i] - S['time'][j]).days) > 36:
                      continue
                  for zone in ('рілля', 'дно'):
                    zm = M['crop'] if zone == 'рілля' else (M['bed'] & (S['dswx'][i] != 1) & (S['dswx'][j] != 1) & ~M['probe'])
                    m = zm & (Rdb[i] > -12) & (Rdb[j] < -18) & np.isfinite(raw[i]) & np.isfinite(raw[j])
                    if m.sum() < (3000 if zone == 'рілля' else 800):
                        continue
                    cr.append({'трек': t, 'зона': zone, 'поляр.': pol.upper(), 'брудна': f"{S['time'][i]:%d.%m}", 'чиста': f"{S['time'][j]:%d.%m}",
                               'пікс.': int(m.sum()),
                               'IQR-відн. сир': iqr(raw[i][m]) / iqr(raw[j][m]), 'IQR-відн. чист': iqr(cl[i][m]) / iqr(cl[j][m]),
                               'зсув медіан сир, дБ': np.median(raw[i][m]) - np.median(raw[j][m]),
                               'зсув медіан чист, дБ': np.median(cl[i][m]) - np.median(cl[j][m]),
                               'зсув середніх сир, дБ': lmean_db(S[pol][i][m]) - lmean_db(S[pol][j][m]),
                               'зсув середніх чист, дБ': lmean_db(S[f'{pol}_c'][i][m]) - lmean_db(S[f'{pol}_c'][j][m])})
    CR = pd.DataFrame(cr)

    # вода проти DSWx-S1 у коридорі
    wr = []
    for t, S in T.items():
        n = len(S['time'])
        valid = [np.isfinite(S['hh'][i]) & (S['dswx'][i] < 250) & M['corridor'] for i in range(n)]
        truth = [np.isin(S['dswx'][i], (1, 3)) for i in range(n)]
        for tag, key in (('сир', 'hh'), ('чист', 'hh_c')):
            xs = [median_filter(np.nan_to_num(S[key][i], nan=0.0), 3) for i in range(n)]
            ok = [i for i in range(n) if valid[i].sum() > 0.9 * M['corridor'].sum() * 0.5 and truth[i][valid[i]].sum() >= 1000]
            for i in ok:
                tr = [j for j in ok if j != i]
                thr = best_threshold([xs[j][valid[j]] for j in tr], [truth[j][valid[j]] for j in tr])
                p, w = xs[i][valid[i]] < thr, truth[i][valid[i]]
                wr.append({'трек': t, 'дата': f"{S['time'][i]:%Y-%m-%d}", 'дані': tag, 'поріг': thr,
                           'IoU': (p & w).sum() / max((p | w).sum(), 1)})
    WT = pd.DataFrame(wr)
    return QC, CR, WT


def report(QC, CR, WT, a):
    L = [f"Параметри: water(min {a.probe_min_frac}, floorσ {a.probe_floor_sigma}), hh={a.hh_mode} (κσ={a.kappa_sigma}), method={a.method}, seasonal={a.seasonal}, probe={a.probe}, q={a.q}, blk={a.blk} ({a.blk * 80 / 1000:.1f} км), sigma={a.sigma}, dark={a.dark}, gain={a.gain}×(HH {a.gain_hh}, HV {a.gain_hv}), floor={a.floor}", '']
    rows = []
    for t in TRACKS:
        q = QC[QC['трек'] == t]
        for P in ('HH', 'HV'):
            rows.append({'трек': t, 'пол': P,
                         'вода сир': q[f'{P} вода сир'].median(), 'вода чист*': q[f'{P} вода чист'].median(),
                         'вода σ сир': q[f'{P} вода сир'].std(), 'вода σ чист*': q[f'{P} вода чист'].std(),
                         'забуд σ сир': q[f'{P} urban сир'].std(), 'забуд σ чист': q[f'{P} urban чист'].std()})
    fr = []
    for t in TRACKS:
        q = QC[QC['трек'] == t]
        for P in ('HH', 'HV'):
            raw, cl = 10 ** (q[f'{P} вода сир'].values / 10), 10 ** (q[f'{P} вода чист'].values / 10)
            fl = raw.min(); ex, res = raw - fl, cl - fl; m = ex > 2 * fl
            fr.append(f"{t} {P}: {np.sum(res[m] * ex[m]) / np.sum(ex[m] ** 2):+.3f} ({m.sum()} дат)")
    L += ['Частка перешкоди, що лишилась над водою (фолди; 0 — ідеал, >0 — недочищено): ' + '; '.join(fr), '']
    L += ['Середня потужність над постійною водою й забудовою (дБ; медіана по датах і розкид між датами); * — фолди, яких не було в оцінці:',
          pd.DataFrame(rows).round(2).to_string(index=False), '']
    if len(CR):
        g = CR.groupby(['зона', 'трек', 'поляр.']).agg(пар=('пікс.', 'size'), **{c: (c, 'median') for c in
              ('IQR-відн. чист', 'зсув медіан сир, дБ', 'зсув медіан чист, дБ', 'зсув середніх сир, дБ', 'зсув середніх чист, дБ')})
        L += ['Рілля і дно (не вода) на брудній даті (R > −12 дБ) проти чистої (R < −18 дБ) ±36 діб, медіана по парах дат (ідеал: IQR 1, зсув 0):',
              g.round(2).to_string(), '']
    if len(WT):
        wt = WT.pivot_table(index='трек', columns='дані', values='IoU', aggfunc='median')
        L += ['Вода проти DSWx-S1 у коридорі русла й дна, медіана IoU:', wt.round(3).to_string()]
    return '\n'.join(L)


def figures(T, QC, OUT, a):
    for t, S in T.items():
        n = len(S['time'])
        fig, axs = plt.subplots(3, n, figsize=(2.3 * n, 6.6), constrained_layout=True)
        for i in range(n):
            axs[0, i].imshow(S['hh'][i][::2, ::2], cmap='gray', vmin=-20, vmax=-2, extent=EXTENT)
            axs[1, i].imshow(np.where(np.isfinite(S['hh'][i]), rc.to_db(S['hh_R'][i]), np.nan)[::2, ::2], cmap='magma', vmin=-25, vmax=-5, extent=EXTENT)
            axs[2, i].imshow(S['hh_c'][i][::2, ::2], cmap='gray', vmin=-20, vmax=-2, extent=EXTENT)
            axs[0, i].set_title(f"{S['time'][i]:%d.%m}", fontsize=8)
            for r in range(3):
                axs[r, i].set_axis_off()
        for r, lab in enumerate(('HH сирий', 'оцінка RFI', 'HH чистий')):
            axs[r, 0].text(-0.1, .5, lab, transform=axs[r, 0].transAxes, rotation=90, va='center', ha='right')
        fig.suptitle(f'{t}, freqB: сирий HH / адитивна оцінка перешкод / після віднімання (−20…−2 дБ; RFI −25…−5 дБ)')
        fig.savefig(OUT / f'before_after_{t}.png', dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True, sharey=True)
    for ax, P in zip(axs, ('HH', 'HV')):
        for t, mk in zip(TRACKS, 'os'):
            q = QC[QC['трек'] == t]; d = pd.to_datetime(q['дата'])
            ax.plot(d, q[f'{P} вода сир'], mk + ':', c='tab:blue', label=f'{t[:5]} вода, сирий')
            ax.plot(d, q[f'{P} вода чист'], mk + '-', c='tab:blue', label=f'{t[:5]} вода, чистий (фолди)')
            ax.plot(d, q[f'{P} urban сир'], mk + ':', c='tab:red', label=f'{t[:5]} забудова, сирий')
            ax.plot(d, q[f'{P} urban чист'], mk + '-', c='tab:red', label=f'{t[:5]} забудова, чистий')
        ax.axhspan(-30, -20, color='tab:blue', alpha=.07)
        ax.set_title(f'{P}: вода й забудова як незалежна перевірка'); ax.set_ylabel('дБ'); ax.set_ylim(-28, 2)
    axs[0].legend(fontsize=7, ncol=2)
    fig.savefig(OUT / 'probe_water_urban.png', dpi=120); plt.close(fig)


def write_change_stacks(T, OUT, max_ratio=2.0):
    """Стеки у форматі pipeline/03_change.py (лінійна потужність) + маска надійності.

    Піксель-дата надійний, якщо оцінка перешкоди не більш ніж у max_ratio разів перевищує
    сигнал (багатодатна медіана очищеної потужності): похибка оцінки R ~10 % тоді дає < ~1 дБ."""
    names = {p.name for p in (DATA / 'h5').glob('*.h5')}
    for t, S in T.items():
        crid = []
        for tt in S['time']:
            m = [re.search(r'_(P\d{5})_', n).group(1) for n in names if f'_{t}_' in n and f'_{tt:%Y%m%dT%H%M}' in n]
            crid.append(m[0] if m else 'P?????')
        ok = np.ones(S['hh'].shape, bool)
        for pol in ('hh', 'hv'):
            sig = np.nanmedian(rc.to_lin(S[f'{pol}_c']), 0)
            ok &= S[f'{pol}_R'] <= max_ratio * np.maximum(sig, 1e-6)
        common = dict(dates=np.array([f'{x:%Y-%m-%d}' for x in S['time']]), crid=np.array(crid), epsg=EPSG,
                      spacing=RES, x=X, y=Y)
        np.savez_compressed(OUT / f'stack_raw_{t}.npz', hh=rc.to_lin(S['hh']), hv=rc.to_lin(S['hv']),
                            source=f'NISAR L2 GCOV freqB, {t}, raw', **common)
        hh, hv = rc.to_lin(S['hh_c']), rc.to_lin(S['hv_c'])
        np.savez_compressed(OUT / f'stack_clean_{t}.npz', hh=hh, hv=hv, reliable=ok,
                            source=f'NISAR L2 GCOV freqB, {t}, RFI-cleaned (rfi_clean.py)', **common)
        rel_all = ok.all(0)
        np.savez_compressed(OUT / f'stack_cleanmask_{t}.npz', hh=np.where(rel_all, hh, np.nan), hv=np.where(rel_all, hv, np.nan),
                            source=f'NISAR L2 GCOV freqB, {t}, RFI-cleaned, лише надійні на всіх датах', **common)
        print(f'{t}: надійних піксель-дат {ok.mean():.0%}; надійних на всіх датах {rel_all.mean():.0%} пікселів')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', type=float, default=10.0, help='нижній квантиль суші в блоці, %%')
    ap.add_argument('--blk', type=int, default=12, help='розмір блоку, пікселів по 80 м')
    ap.add_argument('--sigma', type=float, default=1.0, help='згладжування фону, у блоках')
    ap.add_argument('--floor', default='min', help="'min' або 'p<k>' — підлога по датах")
    ap.add_argument('--probe', default='water', choices=('none', 'water'))
    ap.add_argument('--method', default='dark-mean', choices=('dark-mean', 'quantile'))
    ap.add_argument('--gain', type=float, default=1.0, help='множник оцінки перешкоди')
    ap.add_argument('--gain-hh', type=float, default=1.05, help='калібрування за водою (фолди), HH')
    ap.add_argument('--gain-hv', type=float, default=1.10, help='калібрування за водою (фолди), HV')
    ap.add_argument('--probe-min-frac', type=float, default=0.1, help='мін. частка пікселів води в блоці')
    ap.add_argument('--probe-floor-sigma', type=float, default=0.0, help='згладжування підлоги води, блоків')
    ap.add_argument('--hh-mode', default='direct', choices=('direct', 'from-hv'),
                    help="як оцінювати перешкоду HH; 'from-hv' (κ·R_hv) дає витік у тесті змін, лише для діагностики")
    ap.add_argument('--kappa-sigma', type=float, default=8.0, help='вікно локальної регресії κ, блоків')
    ap.add_argument('--dark', type=float, default=0.3, help='частка темних пікселів суші в блоці')
    ap.add_argument('--season', dest='seasonal', action='store_true', help='сезонний множник (експериментально)')
    ap.add_argument('--out', default=str(DATA / 'analysis_rfi_clean'))
    ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(exist_ok=True)

    WC, JRC = load_aux(); M = masks(WC, JRC)
    T = {t: load_track(t) for t in TRACKS}
    add_probe(M, JRC, T.values())
    print(f"Опорна вода: {M['probe'].sum() * PIX_KM2:.0f} км²")
    T = {t: clean_track(S, M, a) for t, S in T.items()}
    QC, CR, WT = evaluate(T, M, a)
    txt = report(QC, CR, WT, a); print(txt)
    if a.no_save:
        return
    tag = f'{a.hh_mode}_{a.method}{"_s" if a.seasonal else ""}_{a.probe}_b{a.blk}_s{a.sigma:g}_d{a.dark:g}_g{a.gain:g}'
    (OUT / f'summary_{tag}.txt').write_text(txt, encoding='utf-8')
    QC.to_csv(OUT / f'qc_by_date_{tag}.csv', index=False, encoding='utf-8-sig')
    CR.to_csv(OUT / f'contrast_pairs_{tag}.csv', index=False, encoding='utf-8-sig')
    WT.to_csv(OUT / f'water_iou_{tag}.csv', index=False, encoding='utf-8-sig')
    figures(T, QC, OUT, a)
    np.savez_compressed(OUT / 'clean_stack_freqB.npz',
                        **{f'{t}_{k}': T[t][k].astype('f2') for t in TRACKS for k in ('hh_c', 'hv_c', 'hh_R', 'hv_R')},
                        **{f'{t}_time': np.array([f'{x:%Y-%m-%dT%H:%M}' for x in T[t]['time']]) for t in TRACKS},
                        params=json.dumps(vars(a)))
    write_change_stacks(T, OUT)
    print('Збережено в', OUT)


if __name__ == '__main__':
    main()

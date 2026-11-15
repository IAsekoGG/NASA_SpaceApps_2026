"""Обкладинка доповіді: один знімок NISAR, розрізаний діагоналлю — сирий і очищений у двополяризаційних
кольорах (R = HH, G = HV, B = HH − HV), яскрава розтяжка. → report_figures/00_cover_dualpol.jpg

    python -I tools/report_cover.py [--date 2026-07-18] [--c 0.95]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tools'))
import report_figures_dualpol as R  # noqa: E402

OUT = PROJECT / 'report_figures'


def bright_dualpol(hh, hv, rng, gamma=0.62, sat=1.25):
    rgb = R.dualpol(hh, hv, rng, gamma=gamma)
    fin = np.isfinite(hh) & np.isfinite(hv)
    g = rgb.mean(-1, keepdims=True)
    rgb = np.clip(g + (rgb - g) * sat, 0, 1)                 # трохи насиченості
    return np.where(fin[..., None], rgb, R.BGRGB), fin


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--date', default='2026-07-18'); ap.add_argument('--c', type=float, default=0.95); ap.add_argument('--looks', type=int, default=3); ap.add_argument('--out', default='00_cover_dualpol.jpg')
    a = ap.parse_args()
    t = '143_A_025'
    dates, S = R.load(t, a.looks)
    i, j = dates.index(a.date), dates.index(R.REF[t])
    hh_r, hv_r = (x[j] for x in S['raw'])
    h, v = R.db(hh_r), R.db(hv_r); m = np.isfinite(h) & np.isfinite(v)
    p = lambda arr, lo, hi: tuple(np.percentile(arr[m], [lo, hi]))
    rng = (p(h, 1, 96), p(v, 1, 96), p(h - v, 3, 97))         # яскравіша розтяжка, ніж у порівняльних рисунках
    raw, fin = bright_dualpol(*[x[i] for x in S['raw']], rng)
    cl, _ = bright_dualpol(*[x[i] for x in S['clean']], rng)
    H, W = fin.shape
    yy, xx = np.mgrid[0:H, 0:W]
    u = xx / W + (H - yy) / H
    side = u < a.c                                             # нижня ліва частина — сирий знімок
    img = np.where(side[..., None], raw, cl)
    line = (np.abs(u - a.c) < 0.0028) & fin
    glow = gaussian_filter(line.astype('f4'), 2.2); glow /= glow.max(); glow *= fin
    halo = gaussian_filter(line.astype('f4'), 9); halo /= halo.max(); halo *= fin
    img = np.clip(img + glow[..., None] * 1.0 + halo[..., None] * 0.18, 0, 1)
    img = np.where(fin[..., None], img, R.BGRGB)
    # обрізати порожні краї кадру
    rows, cols = np.where(fin.any(1))[0], np.where(fin.any(0))[0]
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    fig = plt.figure(figsize=(16, 16 * img.shape[0] / img.shape[1] + 1.1))
    hfrac = (16 * img.shape[0] / img.shape[1]) / fig.get_figheight()
    ax = fig.add_axes([0, 0, 1, hfrac]); ax.imshow(img, interpolation='lanczos'); ax.set_axis_off()
    ax.text(0.03, 0.06, 'СИРИЙ ЗНІМОК', transform=ax.transAxes, fontsize=20, fontweight='bold', color='#ffd28a',
            bbox=dict(boxstyle='round,pad=0.4', fc=R.BG, ec='none', alpha=0.55))
    ax.text(0.97, 0.90, 'ПІСЛЯ ОЧИЩЕННЯ', transform=ax.transAxes, fontsize=20, fontweight='bold', color='#c9b6ff', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc=R.BG, ec='none', alpha=0.55))
    fig.text(0.015, 1 - 0.35 / fig.get_figheight(), 'Один знімок, дві реальності', fontsize=24, fontweight='bold', color=R.INK, va='top')
    fig.text(0.985, 1 - 0.47 / fig.get_figheight(), f'NISAR · трек 143 A · {a.date[8:10]}.{a.date[5:7]}.2026 · R = HH, G = HV, B = HH − HV',
             fontsize=12, color=R.QUIET, va='top', ha='right')
    fig.savefig(OUT / a.out, dpi=150, pil_kwargs={'quality': 92}); plt.close(fig)
    print('ok', img.shape)


if __name__ == '__main__':
    main()

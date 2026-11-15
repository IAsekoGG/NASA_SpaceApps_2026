"""Двополяризаційні кольорові композити NISAR: R = HH, G = HV, B = HH − HV.

Перешкода додає однакову потужність у HH і HV, але сигнал HV на 6–10 дБ слабший, тож у дБ HV росте
набагато сильніше, а HH − HV падає. На композиті поле під плямою стає зеленим — змінюється колір,
тобто поляризаційний «підпис» поверхні, а не лише яскравість. Після очищення кольори повертаються.

    python -I tools/report_figures_dualpol.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import sarstats as st  # noqa: E402

D = PROJECT / 'data' / 'analysis_rfi_clean'
OUT = PROJECT / 'report_figures'
BG, INK, QUIET = '#0e0f12', '#f2f1ec', '#a9a8a0'
BGRGB = np.array([14, 15, 18]) / 255
FONT = 'C:/Windows/Fonts/segoeui.ttf'; FONTB = 'C:/Windows/Fonts/segoeuib.ttf'
plt.rcParams.update({'font.family': 'Segoe UI', 'text.color': INK, 'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.facecolor': BG, 'axes.edgecolor': '#3a3a40', 'axes.labelcolor': QUIET, 'xtick.color': QUIET, 'ytick.color': QUIET})

# діапазони каналів (дБ), однакові для всіх панелей
R_RNG, G_RNG, B_RNG = (-22, -4), (-28, -10), (2, 12)


def db(lin):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(np.maximum(lin, 1e-6))


LOOKS = 3   # 240 м: однаково для всіх панелей, щоб спекл очищеного HV не забивав кольори
REF = {'108_D_065': '2026-09-25', '143_A_025': '2026-09-04'}   # найчистіші за HV дати


def load(t, looks=LOOKS):
    zr = np.load(D / f'stack_raw_{t}.npz'); zc = np.load(D / f'stack_clean_{t}.npz')
    dates = [str(d) for d in zr['dates']]
    ml = lambda a: st.multilook(a, looks)
    return dates, {'raw': (ml(zr['hh']), ml(zr['hv'])), 'clean': (ml(zc['hh']), ml(zc['hv']))}


def ranges_from(hh, hv):
    """Діапазони каналів за чистою датою (2–98 перцентилі); ті самі для всіх панелей рисунка."""
    h, v = db(hh), db(hv); m = np.isfinite(h) & np.isfinite(v)
    p = lambda a: tuple(np.percentile(a[m], [2, 98]))
    return p(h), p(v), p(h - v)


def dualpol(hh, hv, rng=None, gamma=0.9):
    """hh, hv — лінійна потужність. Повертає RGB 0..1, пропуски — колір тла."""
    h, v = db(hh), db(hv)
    r_r, r_g, r_b = rng or (R_RNG, G_RNG, B_RNG)
    n = lambda a, r: np.clip((a - r[0]) / (r[1] - r[0]), 0, 1) ** gamma
    rgb = np.dstack([n(h, r_r), n(v, r_g), n(h - v, r_b)])
    fin = np.isfinite(hh) & np.isfinite(hv)
    return np.where(fin[..., None], np.nan_to_num(rgb), BGRGB)


def key_axes(fig, rect):
    """Легенда кольорів: що означає кожен канал і як виглядає перешкода."""
    ax = fig.add_axes(rect); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    items = [('#ff3b30', 'R — HH'), ('#34c759', 'G — HV'), ('#2f7bff', 'B — HH − HV'),
             ('#7a3df0', 'фіолетовий — відкритий ґрунт'), ('#5fd35f', 'зелений — рослинність або перешкода'), ('#151515', 'темний — вода')]
    for i, (c, t) in enumerate(items):
        x = (i % 3) * 3.4; y = 0.62 if i < 3 else 0.12
        ax.add_patch(plt.Rectangle((x, y), 0.35, 0.28, color=c, ec='#555' if c == '#151515' else 'none'))
        ax.text(x + 0.5, y + 0.14, t, va='center', fontsize=10.5, color=QUIET)


def tag(ax, text, color):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec='none', alpha=0.8))


# ---------------------------------------------------------------------------------------------
def fig_compare(t, dirty, ref, name, label):
    dates, S = load(t)
    i, j = dates.index(dirty), dates.index(ref)
    rng = ranges_from(*[a[j] for a in S['raw']])
    panels = [(dualpol(*[a[i] for a in S['raw']], rng), f'{dirty[8:10]}.{dirty[5:7]} · сирий знімок', '#ffb37a'),
              (dualpol(*[a[i] for a in S['clean']], rng), f'{dirty[8:10]}.{dirty[5:7]} · після очищення', '#8fc1ff'),
              (dualpol(*[a[j] for a in S['raw']], rng), f'{ref[8:10]}.{ref[5:7]} · найменш забруднена дата, сирий', INK)]
    fig = plt.figure(figsize=(16, 6.6))
    for k, (img, lab, c) in enumerate(panels):
        ax = fig.add_axes([0.01 + k * 0.33, 0.14, 0.32, 0.62]); ax.imshow(img, interpolation='antialiased'); ax.set_axis_off()
        tag(ax, lab, c)
    fig.text(0.01, 0.965, 'Перешкода міняє колір, а не лише яскравість', fontsize=22, fontweight='bold', color=INK, va='top')
    fig.text(0.01, 0.895, f'NISAR, трек {label}, двополяризаційний композит R = HH, G = HV, B = HH − HV, 240 м · '
             'під плямою поля «зеленіють»: HV у дБ росте сильніше за HH, і співвідношення каналів руйнується',
             fontsize=11.5, color=QUIET, va='top')
    key_axes(fig, [0.01, 0.02, 0.98, 0.09])
    fig.savefig(OUT / f'12_dualpol_{name}.jpg', dpi=150, pil_kwargs={'quality': 90}); plt.close(fig)


def fig_season_grid(t, label):
    dates, S = load(t)
    n = len(dates); nc = 6
    fig = plt.figure(figsize=(18, 7.4))
    w, h = 0.98 / nc, 0.36
    rng = ranges_from(*[a[dates.index(REF[t])] for a in S['raw']])
    # два ряди по 11 дат не влазять читабельно: беремо 6 дат через сезон
    pick = [0, 2, 4, 6, 8, 10] if n >= 11 else list(range(min(n, 6)))
    for row, kind, col in ((0, 'raw', '#ffb37a'), (1, 'clean', '#8fc1ff')):
        for k, i in enumerate(pick):
            ax = fig.add_axes([0.01 + k * w, 0.50 - row * 0.41, w - 0.006, h])
            ax.imshow(dualpol(*[a[i] for a in S[kind]], rng), interpolation='antialiased'); ax.set_axis_off()
            ax.text(0.03, 0.95, f'{dates[i][8:10]}.{dates[i][5:7]}', transform=ax.transAxes, va='top', fontsize=11,
                    fontweight='bold', color=col, bbox=dict(boxstyle='round,pad=0.25', fc=BG, ec='none', alpha=0.8))
        fig.text(0.012, 0.50 - row * 0.41 + h + 0.008, 'СИРІ ЗНІМКИ' if kind == 'raw' else 'ПІСЛЯ ОЧИЩЕННЯ',
                 fontsize=12, fontweight='bold', color=col)
    fig.text(0.01, 0.975, 'Зелені плями мандрують від дати до дати — поля залишаються на місці', fontsize=20, fontweight='bold', color=INK, va='top')
    fig.text(0.01, 0.925, f'NISAR, трек {label}, R = HH, G = HV, B = HH − HV · шість дат сезону, однакові діапазони для всіх панелей',
             fontsize=11.5, color=QUIET, va='top')
    fig.savefig(OUT / f'13_dualpol_season_{t[:3]}.jpg', dpi=140, pil_kwargs={'quality': 90}); plt.close(fig)


def fig_density(t, dirty, ref, label):
    """HH проти HV для ріллі: під перешкодою хмара пікселів сповзає до діагоналі HV ≈ HH."""
    dates, S = load(t, 2)
    sys.path.insert(0, str(PROJECT / 'tools'))
    import eval_rfi_clean as E
    WC, JRC = E.load_aux(); crop = st.multilook(E.masks(WC, JRC)['crop'].astype('f4'), 2) > 0.9
    i, j = dates.index(dirty), dates.index(ref)
    sets = [('сирий', i, 'raw', '#ffb37a', f'{dirty[8:10]}.{dirty[5:7]}, сирий'),
            ('очищений', i, 'clean', '#8fc1ff', f'{dirty[8:10]}.{dirty[5:7]}, після очищення'),
            ('чиста дата', j, 'raw', INK, f'{ref[8:10]}.{ref[5:7]}, найменш забруднена дата')]
    fig = plt.figure(figsize=(16, 6.2))
    ext = [-26, -2, -30, -4]
    meds = []
    for k, (_, idx, kind, col, lab) in enumerate(sets):
        hh, hv = db(S[kind][0][idx]), db(S[kind][1][idx])
        m = crop & np.isfinite(hh) & np.isfinite(hv)
        ax = fig.add_axes([0.05 + k * 0.32, 0.13, 0.27, 0.66])
        H, xe, ye = np.histogram2d(hh[m], hv[m], bins=140, range=[[ext[0], ext[1]], [ext[2], ext[3]]])
        ax.imshow(np.log1p(H.T), origin='lower', extent=ext, aspect='auto', cmap='magma', interpolation='bilinear')
        ax.plot([ext[0], ext[1]], [ext[0], ext[1]], color='#8fc1ff', lw=1, ls='--')
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.text(-5.5, -4.8, 'HV = HH', color='#8fc1ff', fontsize=9.5, ha='right', va='top')
        med = np.nanmedian(hh[m] - hv[m]); meds.append(med)
        ax.set_title(f'{lab}\nмедіана HH − HV = {med:.1f} дБ', fontsize=12, color=col, loc='left')
        ax.set_xlabel('HH, дБ'); ax.set_ylabel('HV, дБ' if k == 0 else '')
    fig.text(0.01, 0.975, 'Доказ: під перешкодою пікселі сповзають до діагоналі HV = HH', fontsize=20, fontweight='bold', color=INK, va='top')
    fig.text(0.01, 0.915, f'Рілля, трек {label}, щільність пікселів у площині HH–HV · на найменш забрудненій даті HV нижче за HH на {meds[2]:.1f} дБ; '
             f'однакова перешкода в обох каналах стискає різницю до {meds[0]:.1f} дБ, після очищення — {meds[1]:.1f} дБ', fontsize=11.5, color=QUIET, va='top')
    fig.savefig(OUT / f'14_hh_hv_density_{t[:3]}.png', dpi=150); plt.close(fig)


def gif_dualpol(t, label, width=1100):
    dates, S = load(t)
    H, W = S['raw'][0].shape[1:]
    s = (width // 2 - 12) / W; w, h = int(W * s), int(H * s)
    f, fb, fs = ImageFont.truetype(FONTB, 30), ImageFont.truetype(FONTB, 19), ImageFont.truetype(FONT, 16)
    frames = []
    rng = ranges_from(*[a[dates.index(REF[t])] for a in S['raw']])
    for i, d in enumerate(dates):
        canvas = Image.new('RGB', (width, h + 104), BG)
        for k, (kind, lab, col) in enumerate((('raw', 'СИРИЙ', (255, 179, 122)), ('clean', 'ОЧИЩЕНИЙ', (143, 193, 255)))):
            rgb = dualpol(*[a[i] for a in S[kind]], rng)
            im = Image.fromarray((rgb * 255).astype('u1')).resize((w, h), Image.LANCZOS)
            x0 = 8 + k * (w + 8); canvas.paste(im, (x0, 66))
            ImageDraw.Draw(canvas).text((x0 + 12, 76), lab, font=fb, fill=col)
        dr = ImageDraw.Draw(canvas)
        dr.text((12, 12), f'{d[8:10]}.{d[5:7]}.{d[:4]}', font=f, fill=INK)
        dr.text((215, 22), f'NISAR {label} · R = HH, G = HV, B = HH − HV', font=fs, fill=QUIET)
        y = h + 88; xa, xb = 12, width - 12
        dr.line((xa, y, xb, y), fill=(70, 70, 76), width=3)
        for jj in range(len(dates)):
            xx = xa + (xb - xa) * jj / (len(dates) - 1); r = 8 if jj == i else 4
            dr.ellipse((xx - r, y - r, xx + r, y + r), fill=(143, 193, 255) if jj == i else (110, 110, 118))
        frames.append(canvas.convert('P', palette=Image.ADAPTIVE, colors=128))
    frames[0].save(OUT / f'15_dualpol_timelapse_{t[:3]}.gif', save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    OUT.mkdir(exist_ok=True)
    fig_compare('108_D_065', '2026-07-03', REF['108_D_065'], '108D', '108 D (вечірній)'); print('compare 108')
    fig_compare('143_A_025', '2026-07-18', REF['143_A_025'], '143A', '143 A (нічний)'); print('compare 143')
    fig_season_grid('143_A_025', '143 A'); print('grid')
    fig_density('143_A_025', '2026-07-18', REF['143_A_025'], '143 A'); print('density')
    gif_dualpol('143_A_025', '143 A'); print('gif')


if __name__ == '__main__':
    main()

"""Очищення NISAR GCOV від радіоперешкод (RFI) за адитивною моделлю.

Модель: у кожну дату t спостережена потужність (лінійна, не дБ)
    s_obs(x, t) = s_true(x, t) + R(x, t),
де R — фон перешкод: гладкі плями на десятки кілометрів, що переїжджають від дати до дати
(див. data/analysis_timeseries/rfi_background_*.png). Межі полів крізь плями видно, тож
перешкода саме додається до сигналу, а не замінює його.

Оцінка R (method='dark-mean', за замовчуванням). У кожному блоці ~1 км беремо «темні» пікселі
суші — нижні 30 % за БАГАТОДАТНОЮ медіаною — і їхню середню потужність на кожну дату:
    R(blk, t) ≈ mean_dark(blk, t) − min_t' mean_dark(blk, t').
Вибір пікселів не залежить від шуму перешкоди на конкретній даті, тому середнє незміщене.
(Нижній квантиль тієї самої дати, method='quantile', систематично занижує R.) Над дном
водосховища, де суші мало, опорою є постійна відкрита вода (probe): там справжній сигнал
L-band малий, і зсув середньої потужності води = R.
Що присутнє на всіх датах (тепловий шум і постійна частина перешкод), лишається в «підлозі».

Перевірки (tools/eval_rfi_clean.py): вода в утриманих просторових фолдах, стабільність забудови,
контраст полів на брудних датах проти чистих, вода проти DSWx-S1. Підсумок — data/README.md.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

EPS = 10 ** (-3.5)          # −35 дБ: нижня межа після віднімання


def to_lin(db):
    return np.power(10.0, np.asarray(db, 'f4') / 10.0)


def to_db(lin):
    return (10.0 * np.log10(np.maximum(lin, 1e-12))).astype('f4')


def block_quantile(lin, mask, blk=20, q=10.0, min_frac=0.2):
    """Квантиль q лінійної потужності в блоках blk×blk лише по пікселях mask. Повертає (hb, wb)."""
    H, W = lin.shape
    hb, wb = H // blk, W // blk
    v = np.where(mask & np.isfinite(lin), lin, np.nan)[:hb * blk, :wb * blk]
    v = v.reshape(hb, blk, wb, blk).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    out = np.nanpercentile(v, q, axis=-1)
    out[np.isfinite(v).sum(-1) < blk * blk * min_frac] = np.nan
    return out


def nan_smooth(a, sigma):
    """Гаусове згладжування з пропусками; заповнює дірки з сусідів."""
    w = np.isfinite(a).astype('f4')
    num = gaussian_filter(np.nan_to_num(a, nan=0.0), sigma, mode='nearest')
    den = gaussian_filter(w, sigma, mode='nearest')
    out = num / np.maximum(den, 1e-6)
    out[den < 1e-3] = np.nan
    return out


def fill_nan(a):
    """Заповнює решту пропусків найближчим значенням (ітеративне згладжування)."""
    a = a.copy(); s = 1.0
    while np.isnan(a).any() and s < 64:
        f = nan_smooth(a, s); a[np.isnan(a)] = f[np.isnan(a)]; s *= 2
    return np.nan_to_num(a, nan=np.nanmedian(a))


def upsample(blocks, shape, blk):
    """Блоки → повна сітка (білінійно, центри блоків на місці)."""
    up = zoom(blocks, blk, order=1, mode='nearest', grid_mode=True)
    H, W = shape
    out = np.empty((H, W), 'f4')
    h, w = min(H, up.shape[0]), min(W, up.shape[1])
    out[:h, :w] = up[:h, :w]
    out[h:, :w] = out[h - 1:h, :w]
    out[:, w:] = out[:, w - 1:w]
    return out


def block_mean(lin, mask, blk=20, min_frac=0.05, trim=(0.0, 1.0)):
    """Середня лінійна потужність у блоках по пікселях mask.

    Без обрізання хвоста за замовчуванням: шум перешкоди має довгий хвіст угору, і відкидання
    найяскравіших 5 % занижувало R на ~10–15 % (перевірено водою й контрастом полів)."""
    H, W = lin.shape
    hb, wb = H // blk, W // blk
    v = np.where(mask & np.isfinite(lin), lin, np.nan)[:hb * blk, :wb * blk]
    v = v.reshape(hb, blk, wb, blk).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    if trim[1] < 1.0:
        hi = np.nanpercentile(v, 100 * trim[1], axis=-1, keepdims=True)
        v = np.where(v <= hi, v, np.nan)
    out = np.nanmean(v, axis=-1)
    out[np.isfinite(v).sum(-1) < blk * blk * min_frac] = np.nan
    return out


def dark_set(lin_stack, land, blk=20, frac=0.3):
    """Темні пікселі суші: нижні frac за БАГАТОДАТНОЮ медіаною в кожному блоці.

    Вибір не залежить від шуму перешкоди на конкретній даті. Якщо ж брати нижній квантиль
    тієї самої дати, він вибирає пікселі, де флуктуація перешкоди випадково мала, і R
    недооцінюється, тим сильніше, чим R більший за сигнал (HV)."""
    H, W = land.shape
    med = np.nanmedian(lin_stack, 0)
    thr_b = block_quantile(med, land, blk, 100 * frac, min_frac=0.05)
    thr = np.full((H, W), -np.inf, 'f4')
    hb, wb = thr_b.shape
    thr[:hb * blk, :wb * blk] = np.repeat(np.repeat(np.nan_to_num(thr_b, nan=-np.inf), blk, 0), blk, 1)
    return land & np.isfinite(med) & (med <= thr)


def _floor(Qs, floor):
    if floor == 'min':
        return np.nanmin(Qs, 0)
    return np.nanpercentile(Qs, float(floor[1:]), axis=0)      # 'p<k>': стійкіше до шуму


def estimate_rfi(lin_stack, land, blk=20, q=10.0, sigma_blk=1.5, floor='min', probe=None,
                 probe_min_frac=0.03, return_blocks=False, method='dark-mean', dark_frac=0.3, seasonal=False, probe_floor_sigma=0.0):
    """Оцінює адитивний фон перешкод для кожної дати.

    lin_stack: (n, H, W) лінійна потужність одного треку й однієї поляризації.
    land: (H, W) маска суші, по якій рахуються нижні квантилі.
    probe: (H, W) маска постійної відкритої води (необов'язково). Справжній сигнал води L-band
        набагато нижчий за перешкоди, тож медіана над водою = тепловий шум + R. Там, де суші
        в блоці мало (дно водосховища, русло), фон береться з води, а не екстраполюється з берегів.
    Повертає R (n, H, W) [і блокові проміжні величини, якщо return_blocks].
    """
    n, H, W = lin_stack.shape
    # Згладжуємо ДО мінімуму по датах: мінімум шумних значень зміщений донизу, і R вийшов би завищеним.
    sm = lambda Q: np.stack([np.where(np.isfinite(Q[i]), nan_smooth(Q[i], sigma_blk), np.nan) for i in range(n)])
    if method == 'dark-mean':
        D = dark_set(lin_stack, land, blk, dark_frac)
        Ql = sm(np.stack([block_mean(lin_stack[i], D, blk) for i in range(n)]))
    else:                                                   # 'quantile': стара версія, для порівняння
        Ql = sm(np.stack([block_quantile(lin_stack[i], land, blk, q) for i in range(n)]))
    if seasonal and method == 'dark-mean':
        Rl, m = _separate_season(Ql, iters=6)
    else:
        Rl, m = np.maximum(Ql - _floor(Ql, floor), 0.0), np.ones(n)   # зсув рівня темної суші
    Rb = Rl
    if probe is not None:
        Qw = sm(np.stack([block_mean(lin_stack[i], probe, blk, min_frac=probe_min_frac) for i in range(n)]))
        flw = _floor(Qw, floor)
        if probe_floor_sigma > 0:
            # «підлога» води — тепловий шум, він плавний по території; мінімум по датах у блоці з
            # кількох пікселів зміщений донизу шумом, тож згладжуємо його просторово
            flw = np.where(np.isfinite(flw), nan_smooth(flw, probe_floor_sigma), np.nan)
        Rw = np.maximum(Qw - flw, 0.0)                      # зсув рівня води
        nl, nw = np.isfinite(Rl), np.isfinite(Rw)
        Rb = np.where(nl & nw, 0.5 * (Rl + Rw), np.where(nw, Rw, Rl))
    Rb = np.stack([nan_smooth(Rb[i], 0.7) for i in range(n)])   # легке зшивання суші й води
    R = np.stack([upsample(fill_nan(Rb[i]), (H, W), blk) for i in range(n)])
    if return_blocks:
        return R.astype('f4'), {'Rb': Rb, 'season': m}
    return R.astype('f4')


def _separate_season(Q, iters=6, rel_clean=0.25, min_blocks=40):
    """Розділяє рівень темної суші на сезон і перешкоду:
        Q(t, b) = s(b) · m(t) + R(t, b),  R ≥ 0.
    m(t) — спільний для всієї ділянки множник (зволоження, сходи, оранка): він зачіпає всі блоки
    разом, а перешкода — лише плями. Тому m(t) оцінюємо як медіану Q/s по блоках, де на цю
    дату перешкода мала (R < rel_clean · сигналу), і уточнюємо ітераціями."""
    n = Q.shape[0]
    m = np.ones(n)
    for _ in range(iters):
        s = np.nanmin(Q / m[:, None, None], 0)
        R = Q - s[None] * m[:, None, None]
        ratio = Q / s[None]
        new = m.copy()
        for t in range(n):
            sel = np.isfinite(ratio[t]) & (R[t] < rel_clean * s * m[t])
            if sel.sum() >= min_blocks:
                new[t] = np.nanmedian(ratio[t][sel])
        m = new / np.nanmin(new)          # нормуємо: найтемніша дата = 1
    s = np.nanmin(Q / m[:, None, None], 0)
    return np.maximum(Q - s[None] * m[:, None, None], 0.0), m


def clean(lin_stack, R):
    """Віднімає фон перешкод. Повертає очищену потужність і частку пікселів, що впали нижче нуля."""
    c = lin_stack - R
    neg = (c <= 0) & np.isfinite(lin_stack)
    c = np.where(np.isfinite(lin_stack), np.maximum(c, EPS), np.nan)
    return c.astype('f4'), neg


def snr(lin_clean, R, floor_lin):
    """Відношення сигналу до (перешкоди + підлоги), дБ: де < 0, значення пікселя ненадійне."""
    return to_db(lin_clean / np.maximum(R + floor_lin, 1e-9))


def kappa_fit(Rl_hh, Rhv_b, sigma_blk=8.0, shrink=0.1):
    """Перешкода HH як κ · перешкода HV (по блоках), κ — локальна зважена регресія по суші.

    Rl_hh(t, b) = κ(t, b) · R_hv(t, b) + c(t) + шум. Вільний член c(t) — справжня регіональна
    зміна HH (зволоження ґрунту, сходи), його НЕ віднімаємо. κ(t, b) рахується в гаусовому вікні
    sigma_blk блоків і стягується до глобального κ(t) з вагою shrink (де перешкоди мало, κ
    невизначений, але й не потрібен). Повертає κ (n, hb, wb), c (n), глобальний κ (n), r (n)."""
    n = Rl_hh.shape[0]
    K = np.full(Rl_hh.shape, np.nan); c = np.zeros(n); kg = np.zeros(n); rr = np.zeros(n)
    for t in range(n):
        x, y = Rhv_b[t], Rl_hh[t]
        m = np.isfinite(x) & np.isfinite(y)
        A = np.c_[x[m], np.ones(m.sum())]
        kg[t], c[t] = np.linalg.lstsq(A, y[m], rcond=None)[0]
        rr[t] = np.corrcoef(x[m], y[m])[0, 1]
        xm, ym = np.where(m, x, 0.0), np.where(m, y - c[t], 0.0)
        sxy = gaussian_filter(xm * ym, sigma_blk, mode='constant')
        sxx = gaussian_filter(xm * xm, sigma_blk, mode='constant')
        lam = shrink * np.nanmedian(sxx[sxx > 0]) if (sxx > 0).any() else 1.0
        K[t] = (sxy + lam * kg[t]) / (sxx + lam)
    return np.maximum(K, 0.0), c, kg, rr


def estimate_rfi_pair(lin_hh, lin_hv, land, probe=None, blk=12, sigma_blk=1.0, dark_frac=0.3,
                      kappa_sigma=8.0, return_info=False, **kw):
    """Оцінка перешкоди для HH і HV разом: R_hv — як estimate_rfi (суша + вода),
    R_hh = κ · R_hv. HV слабко реагує на поверхню (вітер над водою, зволоження ріллі), тож
    є чистішим «трасером» перешкоди; HH-оцінка з власного каналу плутала ці ефекти з RFI."""
    n, H, W = lin_hh.shape
    R_hv, info = estimate_rfi(lin_hv, land, blk=blk, sigma_blk=sigma_blk, probe=probe, dark_frac=dark_frac,
                              return_blocks=True, **kw)
    Rhv_b = info['Rb']
    sm = lambda Q: np.stack([np.where(np.isfinite(Q[i]), nan_smooth(Q[i], sigma_blk), np.nan) for i in range(n)])
    D = dark_set(lin_hh, land, blk, dark_frac)
    Q = sm(np.stack([block_mean(lin_hh[i], D, blk) for i in range(n)]))
    Rl_hh = Q - np.nanmin(Q, 0)
    K, c, kg, rr = kappa_fit(Rl_hh, Rhv_b, kappa_sigma)
    Kf = np.stack([upsample(fill_nan(K[i]), (H, W), blk) for i in range(n)])
    R_hh = (Kf * R_hv).astype('f4')
    if return_info:
        return R_hh, R_hv, {'kappa': K, 'intercept': c, 'kappa_global': kg, 'r': rr}
    return R_hh, R_hv

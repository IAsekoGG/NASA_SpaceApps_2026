"""Статистика змін для стеку NISAR GCOV dual-pol (HHHH, HVHV), лінійна потужність.

Формули: Conradsen, Nielsen, Skriver 2016 (omnibus-тест, факторизація R_j), діагональний варіант
для незалежних каналів; Benjamini-Hochberg для контролю частки хибних відкриттів.
Нотатка з виведенням: Research/SAR science/papers/conradsen2016-omnibus.md

Стек усюди має форму (k, c, H, W): k дат, c каналів (HH, HV). NaN = недійсний піксель.
"""
import numpy as np
from scipy.special import polygamma
from scipy.stats import chi2


def multilook(a, m):
    """Блочне усереднення m x m у лінійній потужності. NaN, якщо валідних менше половини блоку."""
    if m == 1:
        return a.astype(np.float32)
    h, w = a.shape[-2] // m * m, a.shape[-1] // m * m
    b = a[..., :h, :w].reshape(*a.shape[:-2], h // m, m, w // m, m)
    valid = np.isfinite(b).sum(axis=(-3, -1))
    out = np.nansum(b, axis=(-3, -1)) / np.maximum(valid, 1)
    out[valid < (m * m) / 2] = np.nan
    return out.astype(np.float32)


def _trigamma_inverse(v):
    """Розв'язує psi'(n) = v відносно n (Var[ln I] = psi'(n) для гамма-спекла з n looks)."""
    n = 1.0 / v + 0.5
    for _ in range(30):
        n = n - (polygamma(1, n) - v) / polygamma(2, n)
    return n


def estimate_enl(stack, block=8):
    """Ефективний ENL за просторовою дисперсією ln I у блоках block x block.

    Більшість блоків лежить усередині однорідних полів, блоки на межах дають важкий хвіст
    дисперсії, тому беремо моду розподілу блочних оцінок, а не середнє.
    Повертає (enl, enl_по_датах_і_каналах).
    """
    k, c = stack.shape[:2]
    h, w = stack.shape[2] // block * block, stack.shape[3] // block * block
    per = np.full((k, c), np.nan)
    for i in range(k):
        for ch in range(c):
            x = np.log(stack[i, ch, :h, :w]).reshape(h // block, block, w // block, block)
            x = x.transpose(0, 2, 1, 3).reshape(-1, block * block)
            x = x[np.isfinite(x).all(axis=1)]
            if x.shape[0] < 200:
                continue
            log_var = np.log(x.var(axis=1, ddof=1))
            lo, hi = np.percentile(log_var, [1, 80])
            hist, edges = np.histogram(log_var, bins=60, range=(lo, hi))
            smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
            mode = np.exp(0.5 * (edges[smooth.argmax()] + edges[smooth.argmax() + 1]))
            per[i, ch] = _trigamma_inverse(mode)
    return float(np.nanmedian(per)), per


def estimate_enl_temporal(stack, max_pixels=2_000_000, seed=0):
    """Ефективний ENL за часовою мінливістю: ln(I_t / I_t+1) незмінного пікселя ~ ln F(2n, 2n).

    Не залежить від просторової текстури (малі поля, строкате дно), бо порівнює піксель сам із собою.
    Міжквартильний розмах стійкий до частки змінених пікселів і до спільного зсуву рівня між датами.
    Упередження: якщо змінилась значна частина сцени, n занижується (тест обережніший);
    якщо спекл між датами корельований (когерентні поверхні), n завищується (більше хибних тривог).
    Повертає (найбільший n серед сусідніх пар дат і каналів, масив (k-1, c)).
    """
    from scipy.optimize import brentq
    from scipy.stats import f as f_dist
    k, c = stack.shape[:2]
    flat = stack.reshape(k, c, -1)
    ok = np.nonzero(np.isfinite(flat).all(axis=(0, 1)) & (flat > 0).all(axis=(0, 1)))[0]
    if ok.size > max_pixels:
        ok = np.random.default_rng(seed).choice(ok, max_pixels, replace=False)
    logs = np.log(flat[:, :, ok].astype(np.float64))
    per = np.full((k - 1, c), np.nan)
    for i in range(k - 1):
        for ch in range(c):
            q25, q75 = np.percentile(logs[i, ch] - logs[i + 1, ch], [25, 75])
            iqr = q75 - q25
            model = lambda n: 2 * np.log(f_dist.ppf(0.75, 2 * n, 2 * n)) - iqr
            if model(0.3) > 0 > model(5000):
                per[i, ch] = brentq(model, 0.3, 5000)
    return float(np.nanmax(per)), per


def omnibus_p(stack, n):
    """p-значення гіпотези "усі k дат однакові" для кожного пікселя. Мале p = зміна.

    Діагональний варіант: канали вважаються незалежними, z ~ chi2(c * (k - 1)).
    Якщо HH і HV насправді корельовані, хибних тривог буде більше за номінал.
    """
    k, c = stack.shape[:2]
    rho = 1 - (k + 1) / (6 * k * n)
    x = stack.astype(np.float64)
    ln_q = n * (k * np.log(k) + np.log(x).sum(axis=0) - k * np.log(x.sum(axis=0)))  # (c, H, W)
    z = -2 * rho * ln_q.sum(axis=0)
    return chi2.sf(z, c * (k - 1))


def rj_p(stack, n):
    """p-значення множників R_j, j = 2..k: "дата j дорівнює попереднім j-1". Форма (k-1, ...)."""
    k, c = stack.shape[:2]
    x = stack.astype(np.float64)
    cum = np.cumsum(x, axis=0)
    out = np.empty((k - 1,) + stack.shape[2:])
    for j in range(2, k + 1):
        ln_r = n * (j * np.log(j) - (j - 1) * np.log(j - 1)
                    + (j - 1) * np.log(cum[j - 2]) + np.log(x[j - 1]) - j * np.log(cum[j - 1]))
        rho = 1 - (1 + 1 / (j * (j - 1))) / (6 * n)
        out[j - 2] = chi2.sf(-2 * rho * ln_r.sum(axis=0), c)
    return out


def bh_threshold(p, q=0.05):
    """Поріг Benjamini-Hochberg: пікселі з p <= порогу позначаються зміною при FDR ~ q."""
    p = np.sort(p[np.isfinite(p)])
    if p.size == 0:
        return 0.0
    ok = p <= q * np.arange(1, p.size + 1) / p.size
    return float(p[ok.nonzero()[0].max()]) if ok.any() else 0.0


def change_points(stack, n, alpha_q, alpha_r=0.01):
    """Коли змінилось: послідовний пошук із перезапуском (Conradsen 2016, рис. 4).

    stack: (k, c, N) лише валідні пікселі. alpha_q: поріг для omnibus (зазвичай поріг BH),
    alpha_r: рівень для окремих R_j.
    Повертає:
      first  (N,) uint8: індекс дати, з якої почався новий режим (1..k-1), 0 = без змін;
      count  (N,) uint8: кількість знайдених змін;
      soft   (N,) bool: omnibus значущий, але жоден R_j не пройшов alpha_r (поступова зміна),
                        тоді first = дата з найменшим p серед R_j.
    """
    k, _, npx = stack.shape
    first = np.zeros(npx, np.uint8)
    count = np.zeros(npx, np.uint8)
    soft = np.zeros(npx, bool)
    start = np.zeros(npx, np.int64)
    active = np.ones(npx, bool)
    for _ in range(k - 1):
        if not active.any():
            break
        for s in np.unique(start[active]):
            idx = np.nonzero(active & (start == s))[0]
            if k - s < 2:
                active[idx] = False
                continue
            sub = stack[s:, :, idx]
            significant = omnibus_p(sub, n) < alpha_q
            pr = rj_p(sub, n)
            hit = pr < alpha_r
            found = significant & hit.any(axis=0)
            new_start = s + hit.argmax(axis=0) + 1
            if s == 0:  # поступова зміна: позначаємо найімовірнішу дату
                gradual = significant & ~hit.any(axis=0)
                first[idx[gradual]] = pr[:, gradual].argmin(axis=0) + 1
                count[idx[gradual]] = 1
                soft[idx[gradual]] = True
            fresh = idx[found]
            first[fresh] = np.where(first[fresh] == 0, new_start[found], first[fresh])
            count[fresh] += 1
            start[fresh] = new_start[found]
            active[idx[~found]] = False
    return first, count, soft


def calibration_report(p, changed):
    """Діагностика калібровки без опорної ділянки.

    Пікселі без змін дають рівномірні p, тому верхня половина гістограми (p > 0.5) має бути
    плоскою. upper_ratio = N(0.5..0.75) / N(0.75..1): ~1 добре; помітно > 1 означає завищений
    ENL (забагато хибних тривог); помітно < 1 означає занижений ENL (тест надто обережний).
    pi0 (оцінка Storey) = частка пікселів без змін.
    """
    p = p[np.isfinite(p)]
    hist, _ = np.histogram(p, bins=20, range=(0, 1))
    a = int(((p > 0.5) & (p <= 0.75)).sum())
    b = int((p > 0.75).sum())
    return {
        "n_pixels": int(p.size),
        "p_histogram_20bins": hist.tolist(),
        "upper_ratio": round(a / max(b, 1), 3),
        "pi0_storey": round(min(1.0, float((p > 0.5).mean() / 0.5)), 3),
        "share_p_below_0.05": round(float((p < 0.05).mean()), 4),
        "share_flagged_fdr": round(float(changed.mean()), 4),
    }

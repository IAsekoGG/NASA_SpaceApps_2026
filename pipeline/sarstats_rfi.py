"""Тест змін для стеку з відомим адитивним фоном перешкод (RFI).

Замість того, щоб віднімати оцінку перешкоди й тестувати різницю (тоді шум на брудних датах
більший, ніж думає звичайний omnibus, і тест картує перешкоди), тестуємо СИРІ інтенсивності з
відомим зсувом. Модель для пікселя, каналу й дати t:

    Y_t ~ Gamma(форма n_t, середнє s_t + R_t),

де s_t — сигнал поверхні, R_t — оцінка перешкоди (rfi_clean.py), n_t — ефективна кількість
виглядів. Похибку самої оцінки R (відносна eps, ~0.1 за калібруванням водою) додаємо до
дисперсії: Var Y = (s+R)²/L + (eps·R)², звідки n_t = 1 / (1/L + eps²·R²/(s+R)²).
Брудні дати (R ≫ s) автоматично отримують малу вагу, викидати їх не треба.

H0 для набору дат T: s_t = s для всіх t ∈ T. Девіанс (−2 ln Q) рахується як
    D(T) = 2 Σ_t n_t [ ln(μ0_t / μ̂_t) + Y_t/μ0_t − Y_t/μ̂_t ],
μ0_t = ŝ + R_t (ŝ — оцінка МП під H0), μ̂_t = max(Y_t, R_t) (кожна дата вільна, s_t ≥ 0).
Канали (HH, HV) незалежні, девіанси додаються. Асимптотично D ~ χ²(c·(|T|−1)).
Девіанси вкладених наборів віднімаються, тож множники R_j (Conradsen 2016) — це
D(1..j) − D(1..j−1) ~ χ²(c). При R ≡ 0 і eps = 0 це рівно звичайний omnibus із sarstats.py.
"""
import numpy as np
from scipy.stats import chi2

TINY = 1e-9


def _fit_s(Y, R, L, eps, iters=25):
    """МП-оцінка спільного сигналу s ≥ 0 по осі 0 (дати). Y, R: (k, ...). Повертає s, n_t."""
    s = np.maximum(np.nanmean(Y - R, axis=0), TINY)
    for _ in range(iters):
        mu = s + R
        n = 1.0 / (1.0 / L + (eps * R / mu) ** 2)
        w = n / mu ** 2                      # рівняння правдоподібності: Σ n_t (Y_t − μ_t)/μ_t² = 0
        s = np.maximum((w * (Y - R)).sum(0) / w.sum(0), TINY)
    mu = s + R
    n = 1.0 / (1.0 / L + (eps * R / mu) ** 2)
    return s, n


def deviance(Y, R, L, eps=0.1):
    """−2 ln Q гіпотези «s однаковий на всіх датах» для стеку (k, c, ...). Повертає (...)."""
    Y = np.maximum(Y.astype(np.float64), TINY); R = np.maximum(R.astype(np.float64), 0.0)
    D = 0.0
    for ch in range(Y.shape[1]):
        y, r = Y[:, ch], R[:, ch]
        s, n = _fit_s(y, r, L, eps)
        mu0 = s + r
        muh = np.maximum(y, r + TINY)
        D = D + 2 * (n * (np.log(mu0 / muh) + y / mu0 - y / muh)).sum(0)
    return D


def _bartlett(k, L):
    return 1 - (k + 1) / (6 * k * L)


def omnibus_p(Y, R, L, eps=0.1):
    k, c = Y.shape[:2]
    return chi2.sf(_bartlett(k, L) * deviance(Y, R, L, eps), c * (k - 1))


def rj_p(Y, R, L, eps=0.1):
    """p-значення R_j, j = 2..k: «дата j така сама, як попередні j−1». Форма (k−1, ...)."""
    k, c = Y.shape[:2]
    Dprev = np.zeros(Y.shape[2:])
    out = np.empty((k - 1,) + Y.shape[2:])
    for j in range(2, k + 1):
        Dj = deviance(Y[:j], R[:j], L, eps)
        rho = 1 - (1 + 1 / (j * (j - 1))) / (6 * L)
        out[j - 2] = chi2.sf(np.maximum(rho * (Dj - Dprev), 0), c)
        Dprev = Dj
    return out


def change_points(Y, R, L, alpha_q, alpha_r=0.01, eps=0.1):
    """Те саме, що sarstats.change_points, але з відомим фоном перешкод. Y, R: (k, c, N)."""
    k, _, npx = Y.shape
    first = np.zeros(npx, np.uint8); count = np.zeros(npx, np.uint8)
    soft = np.zeros(npx, bool); start = np.zeros(npx, np.int64); active = np.ones(npx, bool)
    for _ in range(k - 1):
        if not active.any():
            break
        for s in np.unique(start[active]):
            idx = np.nonzero(active & (start == s))[0]
            if k - s < 2:
                active[idx] = False
                continue
            y, r = Y[s:, :, idx], R[s:, :, idx]
            significant = omnibus_p(y, r, L, eps) < alpha_q
            pr = rj_p(y, r, L, eps)
            hit = pr < alpha_r
            found = significant & hit.any(axis=0)
            new_start = s + hit.argmax(axis=0) + 1
            if s == 0:
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

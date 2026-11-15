"""Самоперевірка sarstats_rfi.py на симуляції з адитивними перешкодами.

    python -I -X utf8 Project/pipeline/selftest_change_rfi.py

Перевіряє: (1) без перешкод тест збігається зі звичайним omnibus; (2) при H0 і сильних перешкодах
частка p < 0.05 ≈ 5 % у кожній групі за силою перешкоди (немає «витоку»), а наївний тест на
очищених даних витікає; (3) потужність і дата зміни.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sarstats as st  # noqa: E402
import sarstats_rfi as sr  # noqa: E402

rng = np.random.default_rng(11)
K, C, L, EPS = 11, 2, 30.0, 0.1
N = 200_000
ok_all = []


def check(name, ok, detail):
    ok_all.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")


def gamma(mean, n=L):
    return rng.gamma(n, 1.0, size=mean.shape) / n * mean


# сигнал: HH ~ −15 дБ, HV ~ −21 дБ з розкидом полів
s = np.stack([10 ** (rng.uniform(-19, -9, N) / 10), 10 ** (rng.uniform(-25, -15, N) / 10)])     # (C, N)
S = np.repeat(s[None], K, 0)                                                                     # (K, C, N)

# 1. Без перешкод = звичайний omnibus
Y0 = gamma(S)
p_new = sr.omnibus_p(Y0, np.zeros_like(Y0), L, eps=0.0)
p_old = st.omnibus_p(Y0, L)
check("R = 0: збіг зі sarstats.omnibus_p", np.allclose(p_new, p_old, rtol=1e-6, atol=1e-10),
      f"макс. різниця {np.abs(p_new - p_old).max():.2e}")

# 2. H0 з перешкодами: на кожну дату перешкода в частині пікселів, R/s до +20 дБ, оцінка R з похибкою 10 %
ratio_db = rng.uniform(-15, 20, (K, 1, N)) * (rng.random((K, 1, N)) < 0.5) * (rng.random(N) > 0.2)   # 20 % пікселів чисті, решта: половина дат брудні
R = np.where(ratio_db != 0, S[:, :1] * 10 ** (ratio_db / 10), 0.0) * np.array([1.0, 1.0])[None, :, None]  # R спільна для каналів у абсолюті
R = np.broadcast_to(R, S.shape).copy()
Y = gamma(S + R)
Rhat = R * (1 + EPS * rng.standard_normal(R.shape))
p = sr.omnibus_p(Y, np.maximum(Rhat, 0), L, EPS)
naive = st.omnibus_p(np.maximum(Y - Rhat, 1e-9), L)
worst = (R / S).max(axis=(0, 1))
bins = [(0, 0.01), (0.01, 1), (1, 10), (10, 1e9)]
lab = ['без перешкод', 'R ≤ s', 'R до 10 s', 'R > 10 s']
row = lambda q: ' | '.join(f'{lb} (n={((worst >= lo) & (worst < hi)).sum()}): {(q[(worst >= lo) & (worst < hi)] < 0.05).mean():.3f}' for (lo, hi), lb in zip(bins, lab))
shares = [(p[(worst >= lo) & (worst < hi)] < 0.05).mean() for lo, hi in bins]
# брудні пікселі допускаємо трохи обережнішими (похибка R робить χ² наближення консервативним)
check("H0 з перешкодами: p<0.05 ≈ 0.05, без витоку", 0.04 < shares[0] < 0.06 and all(0.015 < x < 0.065 for x in shares), row(p))
nshares = [(naive[(worst >= lo) & (worst < hi)] < 0.05).mean() for lo, hi in bins]
check("наївний omnibus на (Y − R) витікає", nshares[-1] > 0.3, row(naive))

# 3. Потужність: −3 дБ в обох каналах з дати 6 на 20 % пікселів
truth = rng.random(N) < 0.2
S1 = S.copy(); S1[6:, :, truth] *= 10 ** (-3 / 10)
Y1 = gamma(S1 + R)
p1 = sr.omnibus_p(Y1, np.maximum(Rhat, 0), L, EPS)
thr = st.bh_threshold(p1, 0.05); flag = p1 <= thr
fdp = (flag & ~truth).sum() / max(flag.sum(), 1)
pw = lambda m: (flag & truth & m).sum() / max((truth & m).sum(), 1)
check("FDR 5 % при перешкодах", fdp < 0.07,
      f"хибних серед позначених {fdp:.3f}; потужність: без перешкод {pw(worst < 0.01):.2f}, R ≤ s {pw((worst >= 0.01) & (worst < 1)):.2f}, "
      f"R до 10 s {pw((worst >= 1) & (worst < 10)):.2f}, R > 10 s {pw(worst >= 10):.2f}")
sel = flag & truth
first, _, _ = sr.change_points(Y1[:, :, sel], np.maximum(Rhat[:, :, sel], 0), L, thr, eps=EPS)
check("дата зміни (±1 дата) серед знайдених", (np.abs(first.astype(int) - 6) <= 1).mean() > 0.8,
      f"точно {(first == 6).mean():.2f}, ±1 {(np.abs(first.astype(int) - 6) <= 1).mean():.2f}")

print('\nУсе пройшло' if all(ok_all) else '\nЄ провали')
sys.exit(0 if all(ok_all) else 1)

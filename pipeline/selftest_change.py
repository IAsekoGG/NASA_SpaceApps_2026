"""Самоперевірка sarstats.py на симуляції. Без даних NISAR і без логіну.

    python -I -X utf8 Project/pipeline/selftest_change.py [--write-stack]

--write-stack зберігає синтетичний стек у Project/data/stack_synthetic.npz (той самий формат,
що пише 01_fetch_subset.py), щоб прогнати 03_change.py від початку до кінця.
"""
import pathlib
import sys

import numpy as np
from scipy.stats import kstest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sarstats as st  # noqa: E402

rng = np.random.default_rng(7)
K, ENL = 10, 20.0
results = []


def speckle(mean, enl=ENL):
    return (rng.gamma(enl, 1.0, size=mean.shape) / enl * mean).astype(np.float32)


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")


# 1. H0: p-значення рівномірні
flat = speckle(np.ones((K, 2, 200, 200), np.float32) * np.array([0.1, 0.02], np.float32)[None, :, None, None])
p = st.omnibus_p(flat, ENL).ravel()
share, ks = (p < 0.05).mean(), kstest(p, "uniform").statistic
check("H0, частка p<0.05", 0.045 < share < 0.055, f"{share:.4f} (очікуємо 0.05), KS={ks:.4f}")
pr = st.rj_p(flat, ENL)
check("H0, множники R_j", all(0.04 < (pr[j] < 0.05).mean() < 0.06 for j in range(K - 1)),
      "частки p<0.05 по j: " + " ".join(f"{(pr[j] < 0.05).mean():.3f}" for j in range(K - 1)))

# 2. Сцена з полями різної яскравості: оцінка ENL за просторовою модою
H = W = 480
fields = np.repeat(np.repeat(rng.uniform(0.02, 0.4, (H // 24, W // 24)), 24, 0), 24, 1).astype(np.float32)
mean = np.stack([np.stack([fields, fields * 0.2])] * K)            # (K, 2, H, W)
truth = np.zeros((H, W), bool)
truth[:, : W // 10] = True                                          # 10% пікселів зі зміною
change_at = 6                                                       # новий режим з дати з індексом 6
mean[change_at:, 1][:, truth] *= 10 ** (-3 / 10)                    # HV падає на 3 дБ
scene = speckle(mean)
enl_hat, _ = st.estimate_enl(scene)
check("оцінка ENL на сцені з полями", abs(enl_hat / ENL - 1) < 0.15, f"{enl_hat:.1f} при істинному {ENL:.0f}")

enl_t, _ = st.estimate_enl_temporal(scene)
check("оцінка ENL за часом (10% змінених)", abs(enl_t / ENL - 1) < 0.1, f"{enl_t:.1f} при істинному {ENL:.0f}")
small = np.repeat(np.repeat(rng.uniform(0.02, 0.4, (H // 5, W // 5)), 5, 0), 5, 1).astype(np.float32)
tiny = speckle(np.stack([np.stack([small, small * 0.2])] * K))      # поля 5 пікс., менші за блок 8
check("малі поля: просторова оцінка занижує, часова тримає",
      st.estimate_enl(tiny)[0] < 0.6 * ENL and abs(st.estimate_enl_temporal(tiny)[0] / ENL - 1) < 0.1,
      f"просторова {st.estimate_enl(tiny)[0]:.1f}, часова {st.estimate_enl_temporal(tiny)[0]:.1f} при істинному {ENL:.0f}")

# 3. FDR і потужність, з оціненим ENL
p = st.omnibus_p(scene, enl_hat)
thr = st.bh_threshold(p, 0.05)
flag = p <= thr
fdp, power = (flag & ~truth).sum() / max(flag.sum(), 1), (flag & truth).mean() / truth.mean()
check("BH-FDR 5%", fdp < 0.07, f"частка хибних серед позначених {fdp:.3f}, потужність {power:.2f} (HV -3 дБ)")

# 4. Дата зміни: слабка зміна (3 дБ, інформаційно) і сильна (6 дБ, критерій)
def timing(stack, flagged):
    v = stack.reshape(K, 2, -1)[:, :, flagged.ravel()]
    first, _, soft = st.change_points(v, enl_hat, thr)
    tf = truth.ravel()[flagged.ravel()]
    return (first[tf] == change_at).mean(), (np.abs(first[tf].astype(int) - change_at) <= 1).mean(), soft.mean()


hit3, near3, soft3 = timing(scene, flag)
mean6 = mean.copy()
mean6[change_at:, 1][:, truth] *= 10 ** (-3 / 10)                   # разом 6 дБ
scene6 = speckle(mean6)
flag6 = st.omnibus_p(scene6, enl_hat) <= thr
hit6, near6, soft6 = timing(scene6, flag6)
print(f"       слабка зміна 3 дБ: дата точно {hit3:.2f}, у межах ±1 проходу {near3:.2f}, поступових {soft3:.2f}")
check("дата зміни, 6 дБ", hit6 > 0.85, f"точно {hit6:.2f}, у межах ±1 проходу {near6:.2f}, "
      f"потужність {(flag6 & truth).mean() / truth.mean():.2f}")

# 5. Діагностика калібровки реагує на хибний ENL
rep = {tag: st.calibration_report(st.omnibus_p(scene, e), flag)["upper_ratio"] for tag, e in
       [("правильний", enl_hat), ("завищений x2", 2 * enl_hat), ("занижений /2", enl_hat / 2)]}
check("діагностика upper_ratio", abs(rep["правильний"] - 1) < 0.08 < rep["завищений x2"] - 1 and rep["занижений /2"] < 0.92,
      ", ".join(f"{k}: {v}" for k, v in rep.items()))

# 6. Multilook підвищує ENL у m*m разів
ml = st.multilook(speckle(np.ones((2, 2, 480, 480), np.float32), enl=4.0), 4)
enl_ml, _ = st.estimate_enl(np.concatenate([ml, ml]), block=8)
check("multilook 4x4", abs(enl_ml / 64 - 1) < 0.2, f"ENL після усереднення {enl_ml:.0f} при очікуваних 64")

if "--write-stack" in sys.argv:
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "stack_synthetic.npz"
    out.parent.mkdir(exist_ok=True)
    dates = [f"2026-{m:02d}-{d:02d}" for m, d in [(6, 21), (7, 3), (7, 15), (7, 27), (8, 20), (9, 1), (9, 13), (9, 25), (10, 7), (10, 19)]]
    np.savez_compressed(out, hh=scene[:, 0], hv=scene[:, 1], dates=np.array(dates), epsg=32636,
                        x=500000 + 80.0 * np.arange(W), y=5290000 - 80.0 * np.arange(H), spacing=80.0,
                        crid=np.array(["SYNTH"] * K), source="synthetic selftest, not NISAR")
    print("синтетичний стек:", out)

print("\nУСЕ ГАРАЗД" if all(results) else "\nЄ ПРОВАЛИ")
sys.exit(0 if all(results) else 1)

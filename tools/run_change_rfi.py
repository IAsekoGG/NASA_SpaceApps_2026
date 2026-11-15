"""Тест змін із відомим фоном перешкод (pipeline/sarstats_rfi.py) на всіх датах обох треків Каховки.

Спершу tools/eval_rfi_clean.py (пише stack_raw_*.npz і clean_stack_freqB.npz з оцінками R).
На вхід ідуть СИРІ інтенсивності й оцінка R, нічого не віднімається і не маскується.

    python -I tools/run_change_rfi.py [--looks 2] [--eps 0.1] [--fdr 0.05]

Вихід: data/change_kakhovka_<трек>_rfiaware.npz, data/analysis_rfi_clean/change_rfiaware.json і рисунок.
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'pipeline'))
import sarstats as st  # noqa: E402
import sarstats_rfi as sr  # noqa: E402

D = PROJECT / 'data' / 'analysis_rfi_clean'
sys.path.insert(0, str(PROJECT / 'tools'))
TRACKS = ('143_A_025', '108_D_065')


def run_track(t, a, Z):
    raw = np.load(D / f'stack_raw_{t}.npz')
    dates = [str(d) for d in raw['dates']]
    Y = np.stack([st.multilook(raw['hh'], a.looks), st.multilook(raw['hv'], a.looks)], 1)            # (k, 2, h, w)
    R = np.stack([st.multilook(Z[f'{t}_hh_R'].astype('f4'), a.looks), st.multilook(Z[f'{t}_hv_R'].astype('f4'), a.looks)], 1)
    k, c, h, w = Y.shape
    valid = (np.isfinite(Y) & (Y > 0)).all((0, 1)) & np.isfinite(R).all((0, 1))
    L = a.enl or a._common_enl
    yv, rv = Y[:, :, valid], np.maximum(R[:, :, valid], 0)
    p = np.full((h, w), np.nan); p[valid] = sr.omnibus_p(yv, rv, L, a.eps)
    thr = st.bh_threshold(p[valid], a.fdr)
    changed = valid & (p <= thr)
    first = np.zeros((h, w), np.uint8); count = np.zeros((h, w), np.uint8); soft = np.zeros((h, w), bool)
    if changed.any():
        f, cnt, s = sr.change_points(Y[:, :, changed], np.maximum(R[:, :, changed], 0), L, thr, a.alpha_r, a.eps)
        first[changed], count[changed], soft[changed] = f, cnt, s

    # витік: частка «змінилось» за силою перешкоди відносно сигналу HH
    with np.errstate(invalid='ignore', divide='ignore'):
        sig = np.nanmedian(np.maximum(Y[:, 0] - R[:, 0], 1e-6), 0)
        ratio = np.nanmax(R[:, 0], 0) / sig
    groups = {'R < 0.5 s': (0, .5), '0.5–1': (.5, 1), '1–2': (1, 2), '2–4': (2, 4), '4–10': (4, 10), '> 10 s': (10, np.inf)}
    leak = {g: round(100 * float(changed[valid & (ratio >= lo) & (ratio < hi)].mean()), 1)
            for g, (lo, hi) in groups.items() if (valid & (ratio >= lo) & (ratio < hi)).sum() > 500}
    px = (80 * a.looks / 1000) ** 2
    cal = st.calibration_report(p[valid], changed[valid])
    res = {'трек': t, 'дат': k, 'ENL': round(float(L), 1), 'eps': a.eps, 'поріг BH': thr,
           'валідно, км²': round(float(valid.sum() * px)), 'змінилось, км²': round(float(changed.sum() * px)),
           'змінилось, %': round(100 * float(changed[valid].mean()), 1), 'поступових, %': round(100 * float(soft[changed].mean()), 1) if changed.any() else 0,
           'частка «змінилось» за R/сигнал HH, %': leak,
           'r(змінилось, log R/сигнал)': round(float(np.corrcoef(changed[valid], np.log10(np.maximum(ratio[valid], 1e-3)))[0, 1]), 3),
           'калібровка upper_ratio': cal['upper_ratio'], 'pi0': cal['pi0_storey'],
           'перша зміна, км²': {dates[i]: round(float((first == i).sum() * px)) for i in range(1, k)}}
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_hh = 10 * np.log10(np.nanmedian(np.maximum(Y[:, 0] - R[:, 0], 1e-4), 0))
    dst = D / f'change_kakhovka_{t}_rfiaware.npz' if a.dir else PROJECT / 'data' / f'change_kakhovka_{t}_rfiaware.npz'
    np.savez_compressed(dst, p=p.astype('f4'), changed=changed, first=first,
                        count=count, soft=soft, ratio=ratio.astype('f4'), median_clean_hh_db=mean_hh.astype('f4'),
                        x=raw['x'][::a.looks][:w], y=raw['y'][::a.looks][:h], epsg=int(raw['epsg']), dates=np.array(dates))
    return res, dict(first=first, changed=changed, valid=valid, ratio=ratio, bg=mean_hh, dates=dates)


def cross_track(M, classes=None):
    """Дати першої зміни у двох треках (календарний день середини проміжку) на пікселях, змінених в обох.
    classes: {назва: маска (h, w)} — те саме окремо по класах поверхні."""
    def doy(m):
        d = [datetime.fromisoformat(x) for x in m['dates']]
        mids = np.array([0] + [((d[i - 1] + (d[i] - d[i - 1]) / 2).timetuple().tm_yday) for i in range(1, len(d))], 'f4')
        return np.where(m['first'] > 0, mids[m['first']], np.nan)
    A, B = M[TRACKS[0]], M[TRACKS[1]]
    h, w = min(A['first'].shape[0], B['first'].shape[0]), min(A['first'].shape[1], B['first'].shape[1])
    da, db = doy(A)[:h, :w], doy(B)[:h, :w]
    both_valid = A['valid'][:h, :w] & B['valid'][:h, :w]
    ca, cb = A['changed'][:h, :w], B['changed'][:h, :w]
    out = {}
    for name, cm in {'усе': np.ones((h, w), bool), **(classes or {})}.items():
        v = both_valid & cm[:h, :w]
        rng = np.random.default_rng(0)
        pa, pb, joint = ca[v].mean(), cb[v].mean(), (ca & cb)[v].mean()
        m = v & ca & cb
        agree = float(np.mean(np.abs(da[m] - db[m]) <= 12)) if m.sum() else np.nan
        shuf = float(np.mean(np.abs(da[m] - rng.permutation(db[m])) <= 12)) if m.sum() else np.nan
        out[name] = {'пікс.': int(v.sum()), 'змінилось у A, %': round(100 * pa, 1), 'у D, %': round(100 * pb, 1),
                     'в обох, %': round(100 * joint, 1), 'очікувано при незалежності, %': round(100 * pa * pb, 1),
                     'підйом (в обох / очікувано)': round(joint / max(pa * pb, 1e-9), 2),
                     'дата першої зміни збігається ±12 діб, %': round(100 * agree, 1), 'випадково (перемішування), %': round(100 * shuf, 1)}
    return out


def bed_vs_dswx(t, m, looks):
    """Дно водосховища проти DSWx-S1 (незалежний еталон, ±3 доби від дати NISAR): частка «змінилось»
    на дні, сухому весь сезон, на дні під водою весь сезон і там, де DSWx бачить перехід суша↔вода;
    плюс частка переходів, де дата першої зміни збігається з першим переходом DSWx (±1 дата)."""
    import eval_rfi_clean as E
    WC, JRC = E.load_aux(); bed80 = E.masks(WC, JRC)['bed']
    S = E.load_track(t); d = S['dswx']; n = len(d)
    ds = lambda x: x[:x.shape[0] // looks * looks, :x.shape[1] // looks * looks].reshape(
        x.shape[0] // looks, looks, x.shape[1] // looks, looks).mean((1, 3))
    h, w = m['changed'].shape
    wf = np.stack([ds((d[i] == 1).astype('f4')) for i in range(n)])[:, :h, :w]
    sf = np.stack([ds((d[i] < 250).astype('f4')) for i in range(n)])[:, :h, :w]
    ok = sf > 0.99; wet = ok & (wf > 0.75); dry = ok & (wf < 0.05); nok = ok.sum(0)
    bed = (ds(bed80.astype('f4')) > 0.5)[:h, :w] & m['valid'] & (nok >= 7)
    groups = {'сухе весь сезон': bed & (dry.sum(0) == nok), 'вода весь сезон': bed & (wet.sum(0) == nok),
              'перехід суша↔вода': bed & (wet.sum(0) >= 2) & (dry.sum(0) >= 2)}
    res = {g: {'пікс.': int(v.sum()), 'змінилось, %': round(100 * float(m['changed'][v].mean()), 1)} for g, v in groups.items()}
    state = np.where(wet, 1, np.where(dry, 0, -1)); first = m['first'].astype(int)
    tr = groups['перехід суша↔вода'] & m['changed']; hits = []
    for y, x in zip(*np.nonzero(tr)):
        s_ = [(i, state[i, y, x]) for i in range(n) if state[i, y, x] >= 0]
        sw = next((i1 for (i0, s0), (i1, s1) in zip(s_[:-1], s_[1:]) if s1 != s0), 0)
        if sw and first[y, x]:
            hits.append(abs(first[y, x] - sw) <= 1)
    res['дата зміни = перший перехід DSWx ±1 дата, %'] = round(100 * float(np.mean(hits)), 1) if hits else None
    return res


def class_masks(looks):
    import eval_rfi_clean as E
    WC, JRC = E.load_aux(); Mk = E.masks(WC, JRC)
    ds = lambda m: m[:m.shape[0] // looks * looks, :m.shape[1] // looks * looks].reshape(
        m.shape[0] // looks, looks, m.shape[1] // looks, looks).mean((1, 3)) > 0.5
    return {'рілля': ds(Mk['crop']), 'дно водосховища': ds(Mk['bed']), 'забудова': ds(Mk['urban'])}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument('--looks', type=int, default=2)
    ap.add_argument('--eps', type=float, default=0.1, help='відносна похибка оцінки R (калібрування водою)')
    ap.add_argument('--fdr', type=float, default=0.05)
    ap.add_argument('--alpha-r', type=float, default=0.01)
    ap.add_argument('--enl', type=float)
    ap.add_argument('--dir', help='тека з результатами eval_rfi_clean.py (за замовчуванням data/analysis_rfi_clean)')
    a = ap.parse_args()
    global D
    if a.dir:
        D = Path(a.dir)
    Z = np.load(D / 'clean_stack_freqB.npz')
    # Спільний ENL: обидва треки — той самий режим і обробка. Просторова оцінка занижена текстурою полів,
    # часова (сусідні чисті дати) завищена корельованим спеклом (60–84); беремо меншу з просторових — обережно.
    est = {}
    for t in TRACKS:
        raw = np.load(D / f'stack_raw_{t}.npz')
        est[t] = st.estimate_enl(np.stack([st.multilook(raw['hh'], a.looks), st.multilook(raw['hv'], a.looks)], 1))[0]
    a._common_enl = min(est.values())
    print('ENL просторовий:', {k: round(v, 1) for k, v in est.items()}, '→ спільний', round(a._common_enl, 1))
    out, M = {}, {}
    for t in TRACKS:
        res, M[t] = run_track(t, a, Z)
        res['дно проти DSWx-S1'] = bed_vs_dswx(t, M[t], a.looks); out[t] = res
        print(json.dumps(res, ensure_ascii=False))
    out['два треки'] = cross_track(M, class_masks(a.looks))
    for k_, v_ in out['два треки'].items():
        print(k_, json.dumps(v_, ensure_ascii=False))
    (D / 'change_rfiaware.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    fig, axs = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    for j, t in enumerate(TRACKS):
        m = M[t]; k = len(m['dates'])
        ax = axs[0, j]; ax.imshow(m['bg'], cmap='gray', vmin=-20, vmax=-4)
        q = ax.imshow(np.ma.masked_equal(m['first'], 0), cmap='turbo', vmin=1, vmax=k - 1, interpolation='nearest')
        cb = fig.colorbar(q, ax=ax, shrink=.7); cb.set_ticks(range(1, k)); cb.set_ticklabels(m['dates'][1:])
        ax.set_title(f"{t}: дата першої зміни, усі {k} дат, без маски ({out[t]['змінилось, %']}% змінилось)"); ax.set_axis_off()
        ax = axs[1, j]
        g = out[t]['частка «змінилось» за R/сигнал HH, %']
        ax.bar(list(g), list(g.values()), color='tab:blue'); ax.set_ylabel('% пікселів «змінилось»')
        ax.set_title(f'{t}: частка змін за силою перешкоди (рівна висота = немає витоку)'); ax.set_xlabel('макс. R / сигнал HH')
    fig.suptitle('Тест змін із відомим фоном перешкод (sarstats_rfi.py), 160 м, FDR 5 %')
    fig.savefig(D / 'change_rfiaware.png', dpi=100)
    print('Збережено:', D / 'change_rfiaware.json', D / 'change_rfiaware.png')


if __name__ == '__main__':
    main()

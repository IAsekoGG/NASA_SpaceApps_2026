"""Тест змін (pipeline/03_change.py) на сирих і очищених від RFI стеках Каховки + перевірка «витоку» перешкод.

Спершу запустіть tools/eval_rfi_clean.py — він пише стеки в data/analysis_rfi_clean/.
Варіанти для треку (за замовчуванням 143_A_025):
  raw        — сирі дані, усі дати;
  clean      — очищені, усі дати;
  clean_sub  — очищені, дати від --from (за замовчуванням 2026-08-01), лише пікселі, надійні на всіх цих датах.
Витік перешкод: якщо карта «змінилось» корелює з силою перешкоди (максимум R по датах),
тест картує перешкоди, а не поверхню.

    python -I tools/run_change_clean.py [--track 143_A_025] [--looks 2]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
D = PROJECT / 'data' / 'analysis_rfi_clean'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default='143_A_025')
    ap.add_argument('--from', dest='date_from', default='2026-08-01')
    ap.add_argument('--looks', type=int, default=2, help='додаткове усереднення перед тестом (2 → 160 м)')
    ap.add_argument('--enl', type=float, help='ENL для очищених варіантів (за замовчуванням оцінка 03_change.py)')
    ap.add_argument('--only', help='кома-список варіантів')
    a = ap.parse_args()
    t = a.track

    raw = np.load(D / f'stack_raw_{t}.npz'); cl = np.load(D / f'stack_clean_{t}.npz')
    meta = {k: raw[k] for k in ('dates', 'crid', 'epsg', 'spacing', 'x', 'y')}
    sub = np.array([d >= a.date_from for d in raw['dates']])
    rel = cl['reliable'][sub].all(0)
    variants = {
        'raw': dict(hh=raw['hh'], hv=raw['hv'], **meta),
        'clean': dict(hh=cl['hh'], hv=cl['hv'], **meta),
        'clean_sub': dict(hh=np.where(rel, cl['hh'][sub], np.nan), hv=np.where(rel, cl['hv'][sub], np.nan),
                          **{**meta, 'dates': meta['dates'][sub], 'crid': meta['crid'][sub]}),
    }
    z = np.load(D / 'clean_stack_freqB.npz')
    Rmax = np.maximum(z[f'{t}_hh_R'].astype('f4').max(0), 0)
    sigmed = np.nanmedian(cl['hh'], 0)
    rfi_ratio = Rmax / np.maximum(sigmed, 1e-6)           # наскільки сильно перешкода хоч раз перевищувала сигнал HH

    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only.split(',')}
    rows = []
    for name, arrs in variants.items():
        p = D / f'_change_in_{t}_{name}.npz'
        np.savez(p, source=f'{t} {name}', **arrs)
        tag = f'kakhovka_{t}_{name}' + (f'_enl{a.enl:g}' if a.enl and name != 'raw' else '')
        r = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(PROJECT / 'pipeline' / '03_change.py'), '--stack', str(p),
                            '--name', tag, '--looks', str(a.looks)] + (['--enl', str(a.enl)] if a.enl and name != 'raw' else []),
                           capture_output=True, text=True, encoding='utf-8')
        print(f'=== {name}\n{r.stdout}{r.stderr[-2000:]}')
        out = np.load(PROJECT / 'data' / f'change_{tag}.npz')
        ch = out['changed']; valid = np.isfinite(out['p'])
        L = a.looks; h, w = ch.shape
        rr = rfi_ratio[:h * L, :w * L].reshape(h, L, w, L).mean((1, 3))
        bins = [0, 0.5, 1, 2, 4, np.inf]; lab = ['<0.5', '0.5–1', '1–2', '2–4', '>4']
        share = []
        for lo, hi, lb in zip(bins[:-1], bins[1:], lab):
            m = valid & (rr >= lo) & (rr < hi)
            share.append((lb, int(m.sum()), float(ch[m].mean()) if m.sum() > 500 else np.nan))
        corr = float(np.corrcoef(ch[valid].astype('f4'), np.log10(np.maximum(rr[valid], 1e-3)))[0, 1])
        js = json.loads((PROJECT / 'web' / 'data' / f'change_{tag}.json').read_text(encoding='utf-8'))
        rows.append({'варіант': name, 'дат': len(arrs['dates']), 'валідно, км²': js['area_valid_km2'],
                     'змінилось, %': round(100 * ch[valid].mean(), 1), 'ENL': js['enl_used'],
                     'r(змінилось, log R/сигнал)': round(corr, 3),
                     'частка «змінилось» за R/сигнал': {lb: (None if np.isnan(s) else round(100 * s, 1)) for lb, n, s in share}})
    print('\nВитік перешкод у карту змін (R/сигнал — максимум по датах, HH):')
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    (D / f'change_leak_{t}{"_enl%g" % a.enl if a.enl else ""}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for name in variants:
        (D / f'_change_in_{t}_{name}.npz').unlink(missing_ok=True)


if __name__ == '__main__':
    main()

"""Перевірка радіозавад (RFI) у NISAR GCOV без завантаження знімка: читає лише метадані (кілька МБ).

    python -I -X utf8 Project/tools/check_rfi.py --urls Project/data_plan/kakhovka_urls_143_A_025.txt --bbox 33.4,46.9,35.2,47.8 --dates 2
    python -I -X utf8 Project/tools/check_rfi.py --files Project/data/h5 --bbox 33.4,46.9,35.2,47.8

Що дивитись:
  NESZ (noiseEquivalentBackscatter): рівень шуму. Норма для NISAR приблизно -25 дБ і нижче.
    Суходіл на L-band дає HH близько -8..-12 дБ і HV близько -15..-20 дБ, тож NESZ вище -15 дБ
    означає, що HV тоне в шумі, а вище -8 дБ тоне й HH.
  rfiLikelihood: за описом у файлі має бути в межах [0, 1].
Потрібен Earthdata Login (для --urls).
"""
import argparse
import pathlib

import h5py
import numpy as np

CAL = "/science/LSAR/GCOV/metadata/calibrationInformation"


def db(v):
    return 10 * np.log10(v) if v > 0 else float("nan")


def report(h5, bbox):
    out = []
    for freq in ("frequencyA", "frequencyB"):
        if f"{CAL}/{freq}" not in h5:
            continue
        g = h5[f"{CAL}/{freq}"]
        nes = g["noiseEquivalentBackscatter"]
        x, y, epsg = nes["xCoordinates"][:], nes["yCoordinates"][:], int(nes["projection"][()])
        sel = (slice(None), slice(None))
        if bbox:
            from rasterio.warp import transform_bounds
            w, s, e, n = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
            rows, cols = np.nonzero((y >= s) & (y <= n))[0], np.nonzero((x >= w) & (x <= e))[0]
            if rows.size and cols.size:
                sel = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        parts = [freq[-1]]
        for pol in ("HH", "HV"):
            if pol not in nes:
                continue
            a = nes[pol][sel]
            a = a[np.isfinite(a) & (a > 0)]
            rfi = float(np.asarray(g[pol]["rfiLikelihood"][()]).ravel()[0]) if f"{pol}/rfiLikelihood" in g else float("nan")
            if a.size:
                p10, p50, p90 = np.percentile(a, [10, 50, 90])
                parts.append(f"{pol} NESZ {db(p10):6.1f} / {db(p50):6.1f} / {db(p90):6.1f} дБ, rfi {rfi:.3g}")
        out.append("  ".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", help="файл зі списком URL (data_plan/*_urls_*.txt)")
    ap.add_argument("--files", help="папка з локальними .h5")
    ap.add_argument("--bbox", help="W,S,E,N: NESZ лише в цьому вікні")
    ap.add_argument("--dates", type=int, default=2, help="скільки дат зі списку перевірити (рівномірно)")
    args = ap.parse_args()
    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else None
    print("NESZ у вікні: 10-й / 50-й / 90-й перцентиль")
    if args.files:
        for path in sorted(pathlib.Path(args.files).glob("*.h5")):
            with h5py.File(path, "r") as h5:
                print(path.name[17:33], path.name.split("_")[11][:8], " | ".join(report(h5, bbox)))
        return
    import earthaccess
    earthaccess.login(strategy="netrc")
    fs = earthaccess.get_fsspec_https_session()
    rows = [line.split("\t") for line in pathlib.Path(args.urls).read_text(encoding="utf-8").splitlines() if line.count("\t") == 4]
    pick = sorted(set(np.linspace(0, len(rows) - 1, min(args.dates, len(rows))).round().astype(int)))
    for i in pick:
        date, level, _, _, url = rows[i]
        with fs.open(url, mode="rb", block_size=4 * 2 ** 20, cache_type="blockcache") as f, h5py.File(f, "r") as h5:
            print(date, level[:4], " | ".join(report(h5, bbox)))


if __name__ == "__main__":
    main()

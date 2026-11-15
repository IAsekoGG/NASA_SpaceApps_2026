"""Читає вікно AOI з NISAR L2 GCOV (HHHH, HVHV, mask) для кожної дати і збирає стек.

Спершу перевірка на одному файлі (друкує структуру, нічого не зберігає):
    python -I -X utf8 Project/pipeline/01_fetch_subset.py --probe

Увесь стек, потокове читання лише потрібних чанків:
    python -I -X utf8 Project/pipeline/01_fetch_subset.py

Якщо потік повільний або рветься: --download (повні файли в Project/data/h5, ~1.8 ГБ кожен).
Потрібен Earthdata Login: python -c "import earthaccess; earthaccess.login(persist=True)"

Вихід: Project/data/stack_<name>.npz з hh, hv (k, H, W) у лінійній потужності, NaN = недійсний.
Кожна дата кешується в Project/data/cache_<name>/, повторний запуск продовжує з місця зупинки.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sarstats as st  # noqa: E402

PROJECT = pathlib.Path(__file__).resolve().parents[1]
GRID = "/science/LSAR/GCOV/grids/frequencyA"
IDENT = "/science/LSAR/identification"


def read_url_list(path, levels):
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 5 and parts[1] in levels:
            rows.append({"date": parts[0], "level": parts[1], "crid": parts[2], "url": parts[4]})
    return rows


def login():
    import earthaccess
    for strategy in ("netrc", "environment"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth.authenticated:
                return earthaccess
        except Exception:
            pass
    raise SystemExit("Немає Earthdata Login. Зареєструйтесь на https://urs.earthdata.nasa.gov/users/new і виконайте:\n"
                     '  python -c "import earthaccess; earthaccess.login(persist=True)"')


def grid_info(g):
    proj = g["projection"]
    epsg = int(proj[()]) if proj.shape == () and int(proj[()]) > 0 else int(proj.attrs["epsg_code"])
    x, y = g["xCoordinates"][:], g["yCoordinates"][:]
    return epsg, x, y, float(x[1] - x[0]), float(y[1] - y[0])


def read_window(h5, ref, bbox, looks):
    """Повертає (hh, hv, ref, meta). ref задає сітку першого файлу, решта дат мусять лягти на неї."""
    g = h5[GRID]
    epsg, x, y, dx, dy = grid_info(g)
    if ref is None:
        from rasterio.warp import transform_bounds
        w, s, e, n = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
        cols = np.nonzero((x >= w) & (x <= e))[0]
        rows = np.nonzero((y >= s) & (y <= n))[0]
        if cols.size < looks or rows.size < looks:
            raise SystemExit(f"AOI {bbox} не перетинає кадр (EPSG:{epsg}, x {x[0]:.0f}..{x[-1]:.0f}, y {y[-1]:.0f}..{y[0]:.0f})")
        nrow, ncol = rows.size // looks * looks, cols.size // looks * looks
        ref = {"epsg": epsg, "dx": dx, "dy": dy, "x0": float(x[cols[0]]), "y0": float(y[rows[0]]), "nrow": nrow, "ncol": ncol}
    if epsg != ref["epsg"] or abs(dx - ref["dx"]) > 1e-6 or abs(dy - ref["dy"]) > 1e-6:
        raise SystemExit(f"Інша сітка: EPSG {epsg}, крок {dx}/{dy} проти {ref['epsg']}, {ref['dx']}/{ref['dy']}. Це інший режим або кадр.")
    fc, fr = (ref["x0"] - x[0]) / dx, (ref["y0"] - y[0]) / dy
    if abs(fc - round(fc)) > 0.01 or abs(fr - round(fr)) > 0.01:
        raise SystemExit(f"Сітка зсунута на нецілий піксель ({fr:.3f}, {fc:.3f}): потрібне перепроєктування, стек не збираю.")
    c0, r0 = int(round(fc)), int(round(fr))
    rs, cs = slice(max(r0, 0), min(r0 + ref["nrow"], y.size)), slice(max(c0, 0), min(c0 + ref["ncol"], x.size))

    def layer(name, fill, dtype):
        out = np.full((ref["nrow"], ref["ncol"]), fill, dtype)
        if rs.stop > rs.start and cs.stop > cs.start:
            out[rs.start - r0: rs.stop - r0, cs.start - c0: cs.stop - c0] = g[name][rs, cs]
        return out

    mask = layer("mask", 255, np.uint8)
    good = (mask >= 1) & (mask <= 5)
    bands = []
    for name in ("HHHH", "HVHV"):
        a = layer(name, np.nan, np.float32)
        a[~good | ~(a > 0)] = np.nan
        bands.append(st.multilook(a, looks))
    meta = {"valid_share": float(good.mean())}
    try:
        meta["crid"] = h5[f"{IDENT}/compositeReleaseId"][()].decode()
    except Exception:
        meta["crid"] = ""
    return bands[0], bands[1], ref, meta


def probe(h5, bbox):
    g = h5[GRID]
    epsg, x, y, dx, dy = grid_info(g)
    for key in ("listOfPolarizations", "listOfCovarianceTerms"):
        if key in g:
            print(f"{key}: {[v.decode() for v in np.atleast_1d(g[key][()])]}")
    print(f"EPSG:{epsg}, крок {dx} / {dy} м, сітка {y.size} x {x.size}")
    for name in ("HHHH", "HVHV", "HHHV", "mask", "numberOfLooks", "rtcGammaToSigmaFactor"):
        if name in g:
            d = g[name]
            print(f"  {name:22s} {d.dtype} {d.shape} chunks={d.chunks} compression={d.compression}")
        else:
            print(f"  {name:22s} немає")
    for key in ("compositeReleaseId", "trackNumber", "frameNumber", "orbitPassDirection", "zeroDopplerStartTime", "isFullFrame"):
        if f"{IDENT}/{key}" in h5:
            v = h5[f"{IDENT}/{key}"][()]
            print(f"  {key}: {v.decode() if isinstance(v, bytes) else v}")
    t = time.time()
    hh, hv, ref, meta = read_window(h5, None, bbox, 1)
    print(f"вікно AOI: {ref['nrow']} x {ref['ncol']} пікс. ({ref['nrow'] * abs(dy) / 1000:.0f} x {ref['ncol'] * dx / 1000:.0f} км), "
          f"прочитано за {time.time() - t:.0f} с, валідних за маскою {meta['valid_share']:.1%}")
    looks_layer = g["numberOfLooks"]
    r, c = looks_layer.shape[0] // 2, looks_layer.shape[1] // 2
    print(f"numberOfLooks у центрі кадру: медіана {np.nanmedian(looks_layer[r:r + 256, c:c + 256]):.1f}")
    with np.errstate(invalid="ignore", divide="ignore"):
        print(f"HH: медіана {10 * np.log10(np.nanmedian(hh)):.1f} дБ; HV: медіана {10 * np.log10(np.nanmedian(hv)):.1f} дБ "
              "(суходіл на L-band: HH приблизно -5..-15, HV приблизно -12..-25)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kakhovka")
    ap.add_argument("--urls", default=str(PROJECT / "data_plan" / "kakhovka_urls_108_D_065.txt"))
    ap.add_argument("--bbox", default="33.4,46.9,35.2,47.8", help="W,S,E,N у градусах")
    ap.add_argument("--looks", type=int, default=4, help="усереднення: 4 дає ~80 м із 20 м")
    ap.add_argument("--levels", default="PROVISIONAL", help="через кому; різні рівні в одному стеку не змішувати")
    ap.add_argument("--limit", type=int, help="взяти лише перші N дат")
    ap.add_argument("--probe", action="store_true", help="показати структуру першого файлу і вийти")
    ap.add_argument("--download", action="store_true", help="завантажити файли повністю замість потоку")
    ap.add_argument("--local-dir", default=str(PROJECT / "data" / "h5"), help="де шукати вже завантажені .h5")
    args = ap.parse_args()

    import h5py
    bbox = tuple(float(v) for v in args.bbox.split(","))
    rows = read_url_list(args.urls, args.levels.split(","))[: args.limit]
    if not rows:
        raise SystemExit(f"У {args.urls} немає рядків рівня {args.levels}")
    local = pathlib.Path(args.local_dir)
    cache = PROJECT / "data" / f"cache_{args.name}"
    cache.mkdir(parents=True, exist_ok=True)
    state = {"ea": None, "fs": None}

    def opened(row):
        path = local / row["url"].rsplit("/", 1)[-1]
        if path.exists():
            return h5py.File(path, "r")
        if state["ea"] is None:
            state["ea"] = login()
            state["fs"] = state["ea"].get_fsspec_https_session()
        if args.download:
            local.mkdir(parents=True, exist_ok=True)
            state["ea"].download([row["url"]], str(local))
            return h5py.File(path, "r")
        return h5py.File(state["fs"].open(row["url"], mode="rb", block_size=8 * 2 ** 20, cache_type="blockcache"), "r")

    if args.probe:
        print("файл:", rows[0]["url"].rsplit("/", 1)[-1])
        with opened(rows[0]) as h5:
            probe(h5, bbox)
        return

    ref, done = None, []
    for row in rows:
        part = cache / f"{row['date']}_L{args.looks}.npz"
        if part.exists():
            z = np.load(part, allow_pickle=False)
            ref = ref or {k: z[k].item() for k in ("epsg", "dx", "dy", "x0", "y0", "nrow", "ncol")}
            done.append((row, part))
            print(f"{row['date']}: з кешу")
            continue
        t = time.time()
        try:
            with opened(row) as h5:
                hh, hv, ref, meta = read_window(h5, ref, bbox, args.looks)
        except SystemExit:
            raise
        except Exception as err:  # обрив мережі тощо: дату пропускаємо, кеш дозволяє повторити
            print(f"{row['date']}: ПОМИЛКА {type(err).__name__}: {err}. Запустіть ще раз, готові дати візьмуться з кешу.")
            continue
        np.savez_compressed(part, hh=hh, hv=hv, crid=meta["crid"] or row["crid"], **ref)
        done.append((row, part))
        print(f"{row['date']}: {hh.shape}, валідних {np.isfinite(hh).mean():.1%}, CRID {meta['crid'] or row['crid']}, {time.time() - t:.0f} с")

    if len(done) < 2:
        raise SystemExit("Менше двох дат, стек не зібрано.")
    parts = [np.load(p, allow_pickle=False) for _, p in done]
    spacing = ref["dx"] * args.looks
    nrow, ncol = parts[0]["hh"].shape
    out = PROJECT / "data" / f"stack_{args.name}.npz"
    np.savez_compressed(
        out, hh=np.stack([p["hh"] for p in parts]), hv=np.stack([p["hv"] for p in parts]),
        dates=np.array([r["date"] for r, _ in done]), crid=np.array([str(p["crid"]) for p in parts]),
        epsg=ref["epsg"], spacing=spacing,
        x=ref["x0"] + ref["dx"] * (args.looks - 1) / 2 + spacing * np.arange(ncol),
        y=ref["y0"] + ref["dy"] * (args.looks - 1) / 2 + ref["dy"] * args.looks * np.arange(nrow),
        source=f"NISAR L2 GCOV, {pathlib.Path(args.urls).name}")
    missing = len(rows) - len(done)
    print(f"\nстек: {out} ({len(done)} дат, {nrow} x {ncol}, крок {spacing:.0f} м)" + (f"; НЕ ПРОЧИТАНО дат: {missing}" if missing else ""))


if __name__ == "__main__":
    main()

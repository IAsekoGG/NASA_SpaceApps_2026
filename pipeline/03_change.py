"""Карта "чи змінилось і коли" зі стеку GCOV.

    python -I -X utf8 Project/pipeline/03_change.py --stack Project/data/stack_kakhovka.npz --name kakhovka

Вхід: npz від 01_fetch_subset.py (hh, hv у лінійній потужності, форма (k, H, W)).
Вихід: Project/data/change_<name>.npz, Project/web/data/change_<name>.json,
       швидкі перегляди PNG у Project/data/quicklook/ (у рідній сітці UTM, не для вебкарти).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sarstats as st  # noqa: E402

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def lonlat_box_to_slices(box, x, y, epsg):
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *box)
    cols = np.nonzero((x >= w) & (x <= e))[0]
    rows = np.nonzero((y >= s) & (y <= n))[0]
    if cols.size == 0 or rows.size == 0:
        raise SystemExit(f"Ділянка {box} не перетинає стек")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--alpha-r", type=float, default=0.01, help="рівень для окремих R_j")
    ap.add_argument("--enl", type=float, help="задати ENL вручну замість оцінки")
    ap.add_argument("--looks", type=int, default=1, help="додаткове усереднення перед тестом")
    ap.add_argument("--stable-bbox", help="W,S,E,N у градусах: ділянка без очікуваних змін для H0")
    args = ap.parse_args()

    z = np.load(args.stack, allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    stack = np.stack([st.multilook(z["hh"], args.looks), st.multilook(z["hv"], args.looks)], axis=1)  # (k, 2, H, W)
    x, y = z["x"][:: args.looks][: stack.shape[3]], z["y"][:: args.looks][: stack.shape[2]]
    spacing = float(z["spacing"]) * args.looks
    k, _, h, w = stack.shape
    valid = (np.isfinite(stack) & (stack > 0)).all(axis=(0, 1))
    stack = np.where(valid[None, None], stack, np.nan)
    print(f"стек: {k} дат {dates[0]} .. {dates[-1]}, {h}x{w} пікс., крок {spacing:.0f} м, валідних {valid.mean():.1%}")
    if len(set(str(c) for c in z["crid"])) > 1:
        print("УВАГА: у стеку різні CRID, це різні рівні калібровки:", sorted(set(str(c) for c in z["crid"])))

    enl_est, enl_per = st.estimate_enl(stack)
    enl = args.enl or enl_est
    print(f"ENL: оцінка {enl_est:.1f} (розкид по датах і каналах {np.nanmin(enl_per):.1f}..{np.nanmax(enl_per):.1f}), використано {enl:.1f}")

    with np.errstate(invalid="ignore", divide="ignore"):
        p = st.omnibus_p(stack, enl)
    p[~valid] = np.nan
    thr = st.bh_threshold(p[valid], args.fdr)
    changed = valid & (p <= thr)

    first = np.zeros((h, w), np.uint8)
    count = np.zeros((h, w), np.uint8)
    soft = np.zeros((h, w), bool)
    if changed.any():
        f, c, s = st.change_points(stack[:, :, changed], enl, thr, args.alpha_r)
        first[changed], count[changed], soft[changed] = f, c, s

    px_km2 = (spacing / 1000) ** 2
    report = st.calibration_report(p[valid], changed[valid])
    summary = {
        "name": args.name, "source": str(z["source"]) if "source" in z else "NISAR L2 GCOV",
        "dates": dates, "crid": sorted(set(str(c) for c in z["crid"])), "spacing_m": spacing,
        "enl_estimated": round(enl_est, 2), "enl_used": round(float(enl), 2),
        "fdr": args.fdr, "p_threshold_bh": thr, "alpha_r": args.alpha_r,
        "area_valid_km2": round(float(valid.sum() * px_km2), 1),
        "area_changed_km2": round(float(changed.sum() * px_km2), 1),
        "share_gradual": round(float(soft[changed].mean()), 3) if changed.any() else 0.0,
        # зміна з індексом i означає: новий режим почався між dates[i-1] та dates[i]
        "first_change_km2_by_date": {dates[i]: round(float((first == i).sum() * px_km2), 1) for i in range(1, k)},
        "calibration": report,
    }
    if args.stable_bbox:
        rs, cs = lonlat_box_to_slices(tuple(float(v) for v in args.stable_bbox.split(",")), x, y, int(z["epsg"]))
        ps = p[rs, cs][valid[rs, cs]]
        summary["stable_area"] = {"bbox": args.stable_bbox, "n_pixels": int(ps.size),
                                  "share_p_below_0.05": round(float((ps < 0.05).mean()), 4),
                                  "p_histogram_20bins": np.histogram(ps, bins=20, range=(0, 1))[0].tolist()}
        print(f"H0 на стабільній ділянці: частка p<0.05 = {summary['stable_area']['share_p_below_0.05']:.3f} (очікуємо ~0.05), n={ps.size}")

    (PROJECT / "data").mkdir(exist_ok=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_db = 10 * np.log10(np.nanmean(stack, axis=0))
    np.savez_compressed(PROJECT / "data" / f"change_{args.name}.npz", p=p.astype(np.float32), changed=changed,
                        first=first, count=count, soft=soft, mean_hh_db=mean_db[0].astype(np.float32),
                        mean_hv_db=mean_db[1].astype(np.float32), x=x, y=y, epsg=int(z["epsg"]), dates=np.array(dates))
    out_json = PROJECT / "web" / "data" / f"change_{args.name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ql = PROJECT / "data" / "quicklook"
    ql.mkdir(exist_ok=True)
    plt.imsave(ql / f"{args.name}_first_change.png", np.ma.masked_equal(first, 0), cmap="viridis", vmin=1, vmax=k - 1)
    plt.imsave(ql / f"{args.name}_mean_hv_db.png", np.clip(mean_db[1], -30, -8), cmap="gray")

    print(f"поріг BH (FDR {args.fdr:.0%}): p <= {thr:.2e}; змінилось {summary['area_changed_km2']} км² "
          f"з {summary['area_valid_km2']} ({changed.sum() / max(valid.sum(), 1):.1%}), поступових {summary['share_gradual']:.0%}")
    print(f"калібровка: upper_ratio {report['upper_ratio']} (добре ~1), частка без змін pi0 {report['pi0_storey']}")
    print("перша зміна, км² по датах:", summary["first_change_km2_by_date"])
    print("записано:", out_json)


if __name__ == "__main__":
    main()

"""Контроль якості стеку ДО тесту змін: покриття, рівні по датах, ENL. Карту змін не рахує.

    python -I -X utf8 Project/pipeline/02_qc.py --stack Project/data/stack_kakhovka.npz --name kakhovka

Вихід: Project/web/data/qc_<name>.json і PNG у Project/data/quicklook/ (рідна сітка UTM).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sarstats as st  # noqa: E402

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    z = np.load(args.stack, allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    stack = np.stack([z["hh"], z["hv"]], axis=1)
    k, _, h, w = stack.shape
    valid_each = np.isfinite(stack).all(axis=1) & (stack > 0).all(axis=1)       # (k, H, W)
    valid = valid_each.all(axis=0)
    stack = np.where(valid[None, None], stack, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(stack)
    med = np.nanmedian(db.reshape(k, 2, -1), axis=2)
    enl_s, _ = st.estimate_enl(stack)
    enl_t, per_t = st.estimate_enl_temporal(stack)

    print(f"стек {k} дат, {h} x {w} пікс., крок {float(z['spacing']):.0f} м, CRID {sorted(set(str(c) for c in z['crid']))}")
    print(f"валідних у всіх датах: {valid.mean():.1%}; по датах: " + " ".join(f"{v.mean():.0%}" for v in valid_each))
    print("дата         HH мед., дБ   HV мед., дБ   HV-HH")
    for i, d in enumerate(dates):
        print(f"{d}   {med[i, 0]:8.2f}   {med[i, 1]:10.2f}   {med[i, 1] - med[i, 0]:6.2f}")
    print(f"розмах медіан між датами: HH {np.ptp(med[:, 0]):.2f} дБ, HV {np.ptp(med[:, 1]):.2f} дБ")
    print(f"ENL: просторова оцінка {enl_s:.1f}, часова {enl_t:.1f} (по парах дат: "
          + " ".join(f"{v:.0f}" for v in np.nanmax(per_t, axis=1)) + ")")

    out = {"dates": dates, "valid_share_all_dates": round(float(valid.mean()), 4),
           "median_db": {d: {"HH": round(float(med[i, 0]), 2), "HV": round(float(med[i, 1]), 2)} for i, d in enumerate(dates)},
           "enl_spatial": round(enl_s, 2), "enl_temporal": round(enl_t, 2),
           "enl_temporal_by_pair": [round(float(v), 1) for v in np.nanmax(per_t, axis=1)]}
    path = PROJECT / "web" / "data" / f"qc_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ql = PROJECT / "data" / "quicklook"
    ql.mkdir(parents=True, exist_ok=True)
    mean_db = np.nanmean(db, axis=0)
    for ch, tag in enumerate(("hh", "hv")):
        lo, hi = np.nanpercentile(mean_db[ch], [2, 98])
        plt.imsave(ql / f"{args.name}_mean_{tag}_db.png", np.clip(mean_db[ch], lo, hi), cmap="gray", vmin=lo, vmax=hi)
    rgb = np.zeros((h, w, 3), np.float32)
    for j, layer in enumerate((mean_db[0], mean_db[1], mean_db[0] - mean_db[1])):
        lo, hi = np.nanpercentile(layer, [2, 98])
        rgb[..., j] = np.nan_to_num((np.clip(layer, lo, hi) - lo) / (hi - lo))
    plt.imsave(ql / f"{args.name}_rgb_hh_hv_ratio.png", rgb[:: max(1, h // 1200), :: max(1, w // 1200)])
    print("записано:", path, "і PNG у", ql)


if __name__ == "__main__":
    main()

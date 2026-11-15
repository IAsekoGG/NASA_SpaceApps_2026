"""Diagnostic figure: radio interference (RFI) in NISAR frame 108_D_065 over Kakhovka.

    python -I Project/tools/rfi_figure.py

Reads Project/data/stack_kakhovka.npz (frequency A, 80 m), aux_kakhovka.npz (masks), the local
GCOV files in Project/data/h5 (frequency B window + noise metadata) and a Sentinel-2 mosaic.
Writes Project/data/quicklook/rfi_diagnosis.png
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GCOV = "/science/LSAR/GCOV/"
CAL = GCOV + "metadata/calibrationInformation/"
MAP_DATES = ["2026-06-21", "2026-07-03", "2026-09-01", "2026-10-19"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CLASSES = [("Built-up (should stay flat)", "#2a78d6", 50), ("Cropland", "#eb6834", 40),
           ("Former reservoir bed", "#1baf7a", None)]
VMIN, VMAX = -25, 5
CMAP = matplotlib.colormaps["gray"].copy()
CMAP.set_bad("#e8d9b5")  # outside radar swath / invalid


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


def h5_for(date: str) -> Path:
    tag = date.replace("-", "")
    return Path([f for f in glob.glob(str(DATA / "h5" / "*.h5")) if f"_{tag}T" in f][0])


def window_b(h, x_rng, y_rng):
    g = h[GCOV + "grids/frequencyB/"]
    x, y = g["xCoordinates"][()], g["yCoordinates"][()]
    cs = np.nonzero((x >= x_rng[0]) & (x <= x_rng[1]))[0]
    rs = np.nonzero((y >= y_rng[0]) & (y <= y_rng[1]))[0]
    sl = (slice(rs[0], rs[-1] + 1), slice(cs[0], cs[-1] + 1))
    m = g["mask"][sl]
    ok = (m >= 1) & (m <= 5)
    out = {}
    for p in ("HHHH", "HVHV"):
        a = g[p][sl].astype("float32")
        a[~ok | ~(a > 0)] = np.nan
        out[p] = a
    return out, x[cs], y[rs]


def noise_median(h, freq, pol, x_rng, y_rng):
    grp = h[CAL + f"frequency{freq}/noiseEquivalentBackscatter/"]
    x, y = grp["xCoordinates"][()], grp["yCoordinates"][()]
    a = grp[pol][()]
    sub = a[np.ix_((y >= y_rng[0]) & (y <= y_rng[1]), (x >= x_rng[0]) & (x <= x_rng[1]))]
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return float(db(np.median(sub))) if sub.size else np.nan


def class_masks(wc, bed):
    return [(bed == 1) if code is None else ((wc == code) & (bed == 0)) for _, _, code in CLASSES]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    z = np.load(DATA / "stack_kakhovka.npz", allow_pickle=False)
    aux = np.load(DATA / "aux_kakhovka.npz", allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    xa, ya = z["x"], z["y"]
    x_rng, y_rng = (xa.min(), xa.max()), (ya.min(), ya.max())
    wc, bed = aux["worldcover"], aux["reservoir_bed"]
    masks_a = class_masks(wc, bed)

    med_a = np.full((len(CLASSES), len(dates)), np.nan)
    med_b = np.full_like(med_a, np.nan)
    noise = {k: [] for k in ("A_HV", "B_HV", "A_HH", "B_HH")}
    maps_b = {}
    for j, d in enumerate(dates):
        for i, m in enumerate(masks_a):
            med_a[i, j] = db(np.nanmedian(z["hh"][j][m]))
        with h5py.File(h5_for(d), "r") as h:
            b, xb, yb = window_b(h, x_rng, y_rng)
            # nearest mask cell from the 80 m stack grid
            ci = np.clip(np.searchsorted(xa, xb), 0, len(xa) - 1)
            ri = np.clip(np.searchsorted(-ya, -yb), 0, len(ya) - 1)
            wcb, bedb = wc[np.ix_(ri, ci)], bed[np.ix_(ri, ci)]
            for i, m in enumerate(class_masks(wcb, bedb)):
                med_b[i, j] = db(np.nanmedian(b["HHHH"][m]))
            for f in ("A", "B"):
                for p in ("HV", "HH"):
                    noise[f"{f}_{p}"].append(noise_median(h, f, p, x_rng, y_rng))
            if d in MAP_DATES:
                maps_b[d] = (db(b["HVHV"]), xb, yb)

    with rasterio.open(DATA / "s2_story" / "s2_truecolor_2026-07_forest.tif") as s2:
        rgb = np.moveaxis(s2.read(), 0, -1)
        b_ = s2.bounds
        s2_ext = (b_.left, b_.right, b_.bottom, b_.top)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    fig = plt.figure(figsize=(17, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.32, wspace=0.24,
                          left=0.04, right=0.94, top=0.90, bottom=0.06)
    fig.suptitle("Radio interference in NISAR frame 108 D 065 over the former Kakhovka reservoir",
                 x=0.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.925, "HV backscatter on 4 dates. Top row: frequency A (20 MHz, the band of our stack). "
             "Middle row: frequency B (5 MHz) from the same files.\nSmooth bright blobs tens of km wide that move "
             "between dates are interference, not ground. Same grey scale in both rows.\nSand colour = outside the radar swath. "
             "Green outline = reservoir bed before 2023 (JRC water occurrence ≥ 50%).",
             fontsize=10.5, color=INK2, va="bottom")
    ext_a = (xa.min() - 40, xa.max() + 40, ya.min() - 40, ya.max() + 40)
    im = None
    for k, d in enumerate(MAP_DATES):
        j = dates.index(d)
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(db(z["hv"][j]), extent=ext_a, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax.contour(xa, ya, bed, levels=[0.5], colors="#1baf7a", linewidths=0.6)
        ax.set_title(f"A  {d}   noise HV {noise['A_HV'][j]:+.0f} dB", fontsize=10, loc="left", color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        axb = fig.add_subplot(gs[1, k])
        hvb, xb, yb = maps_b[d]
        axb.imshow(hvb, extent=(xb.min() - 40, xb.max() + 40, yb.min() - 40, yb.max() + 40), cmap=CMAP,
                   vmin=VMIN, vmax=VMAX, interpolation="nearest")
        axb.contour(xa, ya, bed, levels=[0.5], colors="#1baf7a", linewidths=0.6)
        axb.set_title(f"B  {d}   noise HV {noise['B_HV'][j]:+.0f} dB", fontsize=10, loc="left", color=INK)
        axb.set_xticks([]), axb.set_yticks([])
    cax = fig.add_axes([0.952, 0.42, 0.010, 0.44])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("HV, dB (γ⁰)", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(rgb, extent=s2_ext)
    ax.set_xlim(ext_a[:2]), ax.set_ylim(ext_a[2:])
    ax.contour(xa, ya, bed, levels=[0.5], colors="#1baf7a", linewidths=0.6)
    ax.set_title("Ground truth look: Sentinel-2, summer 2026", fontsize=10, loc="left")
    ax.set_xticks([]), ax.set_yticks([])

    t = np.arange(len(dates))
    lbl = [d[5:] for d in dates]
    for col, (med, name) in enumerate(((med_a, "A"), (med_b, "B")), start=1):
        ax = fig.add_subplot(gs[2, col])
        style_axes(ax)
        for i, (cname, color, _) in enumerate(CLASSES):
            ax.plot(t, med[i], color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=cname, zorder=3)
        ax.set_ylim(-16, 6)
        ax.set_xticks(t, lbl, rotation=45, fontsize=8)
        if col == 1:
            ax.set_ylabel("median HH, dB")
        rng = np.nanmax(med[0]) - np.nanmin(med[0])
        ax.set_title(f"Frequency {name}: median HH by surface\nbuilt-up swings {rng:.1f} dB over the season",
                     fontsize=10, loc="left")
        if col == 1:
            ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK2)

    ax = fig.add_subplot(gs[2, 3])
    style_axes(ax)
    ax.axhspan(-20, -15, color="#e1e0d9", alpha=0.7, lw=0, zorder=1)
    ax.text(0, -21.2, "typical land HV signal (-15..-20 dB)", ha="left", va="top", fontsize=8, color=INK2)
    ax.plot(t, noise["A_HV"], color=INK, linewidth=2, marker="o", markersize=4.5, label="Frequency A", zorder=3)
    ax.plot(t, noise["B_HV"], color=INK2, linewidth=2, linestyle="--", marker="s", markersize=4.5,
            label="Frequency B", zorder=3)
    ax.text(t[-1] + 0.15, noise["A_HV"][-1], "A", color=INK, fontsize=9, va="center")
    ax.text(t[-1] + 0.15, noise["B_HV"][-1], "B", color=INK2, fontsize=9, va="center")
    ax.set_xticks(t, lbl, rotation=45, fontsize=8)
    ax.set_ylabel("noise floor HV, dB")
    ax.set_ylim(-30, 8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title("Noise floor from file metadata\n(noiseEquivalentBackscatter, AOI median)", fontsize=10, loc="left")

    fig.text(0.04, 0.012, "Sources: NISAR L2 GCOV PROVISIONAL (CRID P05023), track 108 D, frame 065; ESA WorldCover 2021; "
             "JRC Global Surface Water; Sentinel-2 L2A (Earth Search). Script: Project/tools/rfi_figure.py",
             fontsize=8, color=MUTED)
    out = DATA / "quicklook" / "rfi_diagnosis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)
    print("noise HV A:", np.round(noise["A_HV"], 1).tolist())
    print("noise HV B:", np.round(noise["B_HV"], 1).tolist())
    print("built-up HH A:", np.round(med_a[0], 1).tolist())
    print("built-up HH B:", np.round(med_b[0], 1).tolist())


if __name__ == "__main__":
    main()

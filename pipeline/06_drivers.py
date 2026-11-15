"""Контекст драйверів для AOI: смуга багаторічної норми, 2026 рік, згода джерел дощу.

Запуск з кореня vault:
    python -I -X utf8 Project/pipeline/06_drivers.py

Читає куби з trend-detective/data, пише Project/web/data/drivers.json.
Це контекст регіону (крок сіток 20-300 км), а не причина зміни в окремому пікселі NISAR.
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "trend-detective" / "data"
OUT = ROOT / "Project" / "web" / "data" / "drivers.json"

AOI = (33.4, 46.9, 35.2, 47.8)            # W, S, E, N: дно Каховського водосховища
GRACE_REGION = (32.5, 46.0, 36.0, 48.5)   # ~300 км, реальна роздільність GRACE
GRACE_CONTROL = (28.0, 49.0, 31.5, 51.5)  # північ центральної України
YEAR = 2026
SEASON = [6, 7, 8, 9, 10]                 # вікно PROVISIONAL-стеку NISAR
NORMAL = (25, 75)                         # межі "норми" для вердикту H6, перцентилі

SOURCES = [
    # ключ, файл, змінна, група, підпис
    ("rain_merra2", "power_grids/native/merra2_PRECTOTCORR.nc", "PRECTOTCORR", "rain", "MERRA-2 precipitation, mm/day"),
    ("rain_imerg", "power_grids/native/imerg_IMERG_PRECTOT.nc", "IMERG_PRECTOT", "rain", "GPM IMERG precipitation, mm/day"),
    ("rain_gpcp", "raw/gpcp/precip.mon.mean.nc", "precip", "rain", "GPCP v2.3 precipitation, mm/day"),
    ("soil_root", "power_grids/native/merra2_GWETROOT.nc", "GWETROOT", "soil", "MERRA-2 root-zone soil wetness, 0-1"),
    ("soil_top", "power_grids/native/merra2_GWETTOP.nc", "GWETTOP", "soil", "MERRA-2 surface soil wetness, 0-1"),
    ("t2m", "power_grids/native/merra2_T2M.nc", "T2M", "temp", "MERRA-2 air temperature at 2 m, C"),
    ("ndvi", "power_grids/native/modis_NDVI.nc", "NDVI", "veg", "MODIS Terra NDVI (GIBS, 0.2 deg)"),
]


def box_mean(path, var, box):
    """Місячний ряд: середнє по клітинках, центри яких у box; якщо таких немає, найближча клітинка."""
    ds = xr.open_dataset(DATA / path, drop_variables=["time_bnds", "time_bounds"])
    da = ds[var] if var in ds else ds[list(ds.data_vars)[-1]]
    w, s, e, n = box
    lon = da["lon"].values
    if lon.max() > 180:  # сітка 0..360
        w, e = w % 360, e % 360
    lat = da["lat"].values
    lon_ok = (lon >= w) & (lon <= e)
    lat_ok = (lat >= s) & (lat <= n)
    if lon_ok.sum() == 0:
        lon_ok = np.arange(lon.size) == np.abs(lon - (w + e) / 2).argmin()
    if lat_ok.sum() == 0:
        lat_ok = np.arange(lat.size) == np.abs(lat - (s + n) / 2).argmin()
    sub = da.isel(lon=np.where(lon_ok)[0], lat=np.where(lat_ok)[0])
    series = sub.mean(["lat", "lon"], skipna=True).to_series().dropna()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
    return series, int(lon_ok.sum() * lat_ok.sum())


def summarise(series):
    hist = series[series.index.year < YEAR]
    cur = series[series.index.year == YEAR]
    clim = hist.groupby(hist.index.month).quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    months = [m for m in SEASON if m in set(cur.index.month)]
    out = {
        "first_year": int(hist.index.year.min()),
        "climatology": {str(m): [round(float(v), 4) for v in clim.loc[m]] for m in clim.index},
        "current": {str(d.month): round(float(v), 4) for d, v in cur.items()},
        "season_months": months,
        "season_percentile": None,
    }
    if months:
        h = hist[hist.index.month.isin(months)]
        per_year = h.groupby(h.index.year).mean()[h.groupby(h.index.year).count() == len(months)]
        value = cur[cur.index.month.isin(months)].mean()
        out.update(season_value=round(float(value), 4), season_median=round(float(per_year.median()), 4),
                   season_n_years=int(per_year.size),
                   season_percentile=round(float((per_year < value).mean() * 100), 1))
    return out


def grace_difference():
    path = next((DATA / "raw" / "grace_csr").glob("*all-corrections*.nc"))
    ds = xr.open_dataset(path, drop_variables=["time_bounds"], decode_times=False)
    da = ds["lwe_thickness"]
    time = pd.to_datetime("2002-01-01") + pd.to_timedelta(ds["time"].values, unit="D")
    lon, lat = da["lon"].values, da["lat"].values

    def region(box):
        w, s, e, n = box
        if lon.max() > 180:
            w, e = w % 360, e % 360
        vals = da.values[:, (lat >= s) & (lat <= n), :][:, :, (lon >= w) & (lon <= e)]
        return pd.Series(np.nanmean(vals.reshape(len(time), -1), axis=1), index=time)

    diff = region(GRACE_REGION) - region(GRACE_CONTROL)
    base = diff["2004":"2020"]
    deseason = diff - base.groupby(base.index.month).mean().reindex(diff.index.month).values
    return {
        "units": "cm water equivalent, region minus control, deseasonalised (2004-2020 monthly means)",
        "region": GRACE_REGION, "control": GRACE_CONTROL,
        "note": "GRACE resolves ~300 km; gap 2017-07..2018-05 between missions; significance not tested here",
        "series": {d.strftime("%Y-%m"): round(float(v), 2) for d, v in deseason.dropna().items()},
    }


def main():
    result = {"aoi": AOI, "year": YEAR, "season": SEASON, "normal_band_percentiles": NORMAL, "variables": {}}
    print(f"{'key':12s} {'cells':>5s} {'from':>5s} {'months':>7s} {'2026':>8s} {'median':>8s} {'pctile':>7s}")
    for key, path, var, group, label in SOURCES:
        series, cells = box_mean(path, var, AOI)
        info = summarise(series)
        info.update(label=label, group=group, cells=cells, last_month=series.index.max().strftime("%Y-%m"))
        result["variables"][key] = info
        m = info["season_months"]
        span = f"{m[0]}-{m[-1]}" if m else "-"
        print(f"{key:12s} {cells:5d} {info['first_year']:5d} {span:>7s} {info.get('season_value', float('nan')):8.3f} "
              f"{info.get('season_median', float('nan')):8.3f} {str(info['season_percentile']):>7s}")

    rain = {k: v["season_percentile"] for k, v in result["variables"].items()
            if v["group"] == "rain" and v["season_percentile"] is not None}
    inside = [k for k, p in rain.items() if NORMAL[0] <= p <= NORMAL[1]]
    below = [k for k, p in rain.items() if p < NORMAL[0]]
    above = [k for k, p in rain.items() if p > NORMAL[1]]
    if len(inside) >= 2:
        verdict = "normal"
    elif len(below) >= 2:
        verdict = "dry"
    elif len(above) >= 2:
        verdict = "wet"
    else:
        verdict = "undetermined"
    result["rain_agreement"] = {"percentiles": rain, "inside_normal": inside, "below": below, "above": above,
                                "verdict": verdict,
                                "rule": "verdict needs at least 2 of 3 rain sources in the same band"}
    result["grace"] = grace_difference()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nrain verdict: {verdict}  {rain}")
    g = pd.Series(result["grace"]["series"])
    print(f"GRACE diff, mean 2021-06..2023-05: {g['2021-06':'2023-05'].mean():.2f} cm; "
          f"2023-07..2025-06: {g['2023-07':'2025-06'].mean():.2f} cm; last month {g.index[-1]}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

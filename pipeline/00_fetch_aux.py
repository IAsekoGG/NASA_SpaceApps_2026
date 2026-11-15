"""Step 00: ancillary layers for the Kakhovka AOI.

Jobs (--jobs, default all):
  worldcover  ESA WorldCover 2021 v200, 10 m land cover (cropland = 40, tree = 10, water = 80, built = 50). No login.
  jrc         JRC Global Surface Water occurrence 1984-2021 (% of time with water), 30 m.
              Reservoir bed before 2023 = occurrence >= 50. No login.
  dswx        OPERA DSWx-S1 and DSWx-HLS water maps (B01_WTR, 30 m) within +-3 days of each NISAR date.
              Earthdata login.
  s2          Sentinel-2 L2A true-colour mosaics (60 m, EPSG:32636) for the story: reservoir full (May 2023),
              just drained (June-July 2023), summer 2024, summer 2026. No login (Earth Search / AWS).

    python -I Project/pipeline/00_fetch_aux.py --jobs worldcover,jrc,dswx,s2
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AOI = (33.4, 46.9, 35.2, 47.8)
UA = {"User-Agent": "spaceapps-rhythm-or-rupture/0.1"}


def fetch(url: str, dest: Path, session=None) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests
    with s.get(url, stream=True, timeout=300, headers=UA) as r:
        if r.status_code != 200:
            return f"FAIL {r.status_code} {url}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for c in r.iter_content(4 << 20):
                f.write(c)
    tmp.replace(dest)
    return f"ok   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)"


def crop_to_aoi(src_path: Path, dst_path: Path) -> None:
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(src_path) as src:
        win = from_bounds(*AOI, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(width=data.shape[1], height=data.shape[0], transform=src.window_transform(win),
                    compress="deflate", tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(data, 1)
    print(f"crop {dst_path.name} {data.shape}")


def job_worldcover() -> None:
    d = DATA / "aux"
    print(fetch("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                "ESA_WorldCover_10m_2021_v200_N45E033_Map.tif", d / "raw" / "worldcover_N45E033.tif"))
    crop_to_aoi(d / "raw" / "worldcover_N45E033.tif", d / "worldcover_2021_aoi.tif")


def job_jrc() -> None:
    d = DATA / "aux"
    print(fetch("https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/"
                "occurrence_30E_50Nv1_4_2021.tif", d / "raw" / "jrc_occurrence_30E_50N.tif"))
    crop_to_aoi(d / "raw" / "jrc_occurrence_30E_50N.tif", d / "jrc_occurrence_aoi.tif")


def nisar_dates() -> list[date]:
    lst = ROOT / "data_plan" / "kakhovka_urls_108_D_065.txt"
    return [date.fromisoformat(l.split("\t")[0]) for l in lst.read_text().splitlines() if l.strip()]


def job_dswx(days: int = 3) -> None:
    import earthaccess
    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()
    nd = nisar_dates()
    jobs = []
    for short, tag in (("OPERA_L3_DSWX-S1_V1", "s1"), ("OPERA_L3_DSWX-HLS_V1", "hls")):
        res = earthaccess.search_data(short_name=short, bounding_box=AOI,
                                      temporal=(str(min(nd) - timedelta(days)), str(max(nd) + timedelta(days))))
        kept = 0
        for g in res:
            ur = g["umm"]["GranuleUR"]
            y, m, d = re.search(r"_(\d{4})(\d{2})(\d{2})T", ur).groups()
            t = date(int(y), int(m), int(d))
            if min(abs((t - n).days) for n in nd) > days:
                continue
            for link in g.data_links():
                if link.endswith("_B01_WTR.tif"):
                    jobs.append((link, DATA / "opera_dswx" / tag / link.rsplit("/", 1)[1]))
                    kept += 1
        print(f"{short}: {len(res)} granules in window, {kept} within +-{days} days of a NISAR date")
    fails = 0
    with ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(as_completed([ex.submit(fetch, u, p, session) for u, p in jobs]), 1):
            msg = f.result()
            fails += msg.startswith("FAIL")
            if msg.startswith("FAIL") or i % 50 == 0:
                print(f"  dswx {i}/{len(jobs)} {msg}", flush=True)
    print(f"dswx done, {len(jobs)} files, failures {fails}")


S2_PERIODS = {
    "2023-05_full": ("2023-05-01", "2023-06-05"),
    "2023-06_drained": ("2023-06-12", "2023-07-20"),
    "2024-07_regrowth": ("2024-06-15", "2024-08-31"),
    "2026-07_forest": ("2026-06-15", "2026-08-31"),
}


def job_s2(res_m: int = 60) -> None:
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from pyproj import Transformer
    from PIL import Image
    tr = Transformer.from_crs(4326, 32636, always_xy=True)
    xs, ys = tr.transform([AOI[0], AOI[2], AOI[0], AOI[2]], [AOI[1], AOI[1], AOI[3], AOI[3]])
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    W, H = int((right - left) / res_m), int((top - bottom) / res_m)
    out_dir = DATA / "s2_story"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, (t0, t1) in S2_PERIODS.items():
        dst = out_dir / f"s2_truecolor_{tag}.tif"
        if dst.exists():
            print(f"skip {dst.name}")
            continue
        body = {"collections": ["sentinel-2-l2a"], "bbox": list(AOI), "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z",
                "limit": 200, "query": {"eo:cloud_cover": {"lt": 20}}}
        items = requests.post("https://earth-search.aws.element84.com/v1/search", json=body, timeout=120,
                              headers=UA).json().get("features", [])
        items.sort(key=lambda it: it["properties"].get("eo:cloud_cover", 100))
        mosaic = np.zeros((3, H, W), dtype="uint8")
        filled = np.zeros((H, W), bool)
        used = []
        from rasterio.transform import from_origin
        transform = from_origin(left, top, res_m, res_m)
        for it in items:
            if filled.mean() > 0.995:
                break
            href = it["assets"]["visual"]["href"]
            with rasterio.open(href) as src, WarpedVRT(src, crs="EPSG:32636", transform=transform, width=W, height=H,
                                                      resampling=Resampling.average) as vrt:
                rgb = vrt.read(out_shape=(3, H, W))
            have = rgb.any(axis=0) & ~filled
            if have.sum() == 0:
                continue
            mosaic[:, have] = rgb[:, have]
            filled |= have
            used.append(f"{it['id']} (cloud {it['properties'].get('eo:cloud_cover', 0):.1f}%)")
        prof = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs="EPSG:32636", transform=transform,
                    compress="deflate", photometric="RGB")
        with rasterio.open(dst, "w", **prof) as f:
            f.write(mosaic)
        Image.fromarray(np.moveaxis(mosaic, 0, -1)).save(dst.with_suffix(".png"))
        (dst.with_suffix(".txt")).write_text("\n".join(used))
        print(f"ok   {dst.name}: {len(items)} candidate scenes, {len(used)} used, filled {filled.mean():.3f}")


def job_regrid() -> None:
    """WorldCover (mode) and JRC occurrence (average) on the NISAR stack grids from step 01."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    aoi_dir = DATA
    stacks = sorted(aoi_dir.glob("stack_kakhovka*.npz"))
    if not stacks:
        print("no data/stack_kakhovka*.npz yet: run 01_fetch_subset.py first")
    for stack in stacks:
        z = np.load(stack, allow_pickle=False)
        x, y, epsg = z["x"], z["y"], int(z["epsg"])
        res = float(x[1] - x[0])
        transform = from_origin(x[0] - res / 2, y[0] + res / 2, res, res)
        out = {}
        for name, path, rs, dtype, nodata in (
                ("worldcover", DATA / "aux" / "worldcover_2021_aoi.tif", Resampling.mode, "uint8", 0),
                ("jrc_occurrence", DATA / "aux" / "jrc_occurrence_aoi.tif", Resampling.average, "float32", np.nan)):
            dst = np.full((len(y), len(x)), nodata, dtype=dtype)
            with rasterio.open(path) as src:
                src_arr = src.read(1)
                src_nodata = src.nodata
                if name == "jrc_occurrence":
                    src_arr = np.where(src_arr > 100, np.nan, src_arr).astype("float32")  # 255 = no data
                    src_nodata = np.nan
                reproject(src_arr, dst, src_transform=src.transform, src_crs=src.crs, src_nodata=src_nodata,
                          dst_transform=transform, dst_crs=f"EPSG:{epsg}", dst_nodata=nodata, resampling=rs)
            out[name] = dst
        out["reservoir_bed"] = (out["jrc_occurrence"] >= 50).astype("uint8")
        out["cropland"] = (out["worldcover"] == 40).astype("uint8")
        dst_path = aoi_dir / stack.name.replace("stack_", "aux_")
        np.savez(dst_path, x=x, y=y, epsg=epsg, **out)
        print(f"{dst_path.name}: reservoir bed {out['reservoir_bed'].sum() * res * res / 1e6:.0f} km2, "
              f"cropland share {out['cropland'].mean():.2f}, classes {np.unique(out['worldcover']).tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="worldcover,jrc,dswx,s2")
    args = ap.parse_args()
    for job in args.jobs.split(","):
        print(f"=== {job}", flush=True)
        try:
            {"worldcover": job_worldcover, "jrc": job_jrc, "dswx": job_dswx, "s2": job_s2,
             "regrid": job_regrid}[job]()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL job {job}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

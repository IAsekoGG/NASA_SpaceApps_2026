# Kakhovka under radio interference

NISAR L-band monitoring of the former Kakhovka Reservoir (Ukraine), June–November 2026, under strong radio-frequency interference (RFI).
NASA Space Apps Challenge 2026.

**Live web app:** https://kakhovka.mriiacraft.pp.ua (mirror: https://znai-dev.github.io/NASA_Space_Apps_2026/)

**Problem.** NISAR L2 GCOV frames over the area (tracks 143 A and 108 D, 22 acquisitions) are contaminated by RFI: smooth blobs 30–60 km across that move between dates and raise the noise floor by 10–30 dB. A standard multitemporal change test on the raw stack flags **98 %** of the area as changed — it maps interference, not the surface.

**Approach.**
1. **Additive RFI model.** Observed power `Y = s + R`. Field boundaries remain visible inside the blobs, so RFI adds to the surface signal; correction is done in linear power, not in dB.
2. **RFI estimation.** In 1 × 1 km blocks, the mean power of dark land pixels (selected by the multi-date median, so the selection is independent of the current date's RFI noise) and of permanent open water, relative to the cleanest date. Calibrated on held-out water.
3. **RFI-aware change test.** The omnibus test (Conradsen et al., 2016) generalised to `Y_t ~ Gamma(n_t, s + R_t)` with an effective number of looks `n_t = 1 / (1/L + ε²R_t²/(s + R_t)²)`. Contaminated dates are down-weighted instead of discarded; for `R = 0` it reduces to the standard test.

**Key results** (all validated on data not used to estimate RFI):

| | Raw | Corrected |
|---|---|---|
| False changes where RFI > 10 × signal (143 A) | 100 % | 3.4 % |
| Cropland HV bias, contaminated vs clean date (108 D) | +10.4 dB | +1.3 dB |
| Water mask vs OPERA DSWx-S1, IoU (Dnipro corridor, 143 A) | 0.34 | 0.51 |
| Cropland changes seen by both independent tracks | — | 4.0× more often than chance |

Details, limitations and all numbers: [`data/README.md`](data/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | Core library: `rfi_clean.py` (RFI estimation and correction), `sarstats.py` (omnibus test), `sarstats_rfi.py` (RFI-aware test), data fetching, self-tests |
| `tools/` | Evaluation and export: `eval_rfi_clean.py`, `run_change_rfi.py`, `export_web.py`, local web server |
| `ipynb/` | Research notebooks (two-date diagnosis, time series, robustness checks) |
| `web/` | Static web application (MapLibre GL, no build step): raw vs corrected swipe, change maps, validation charts, per-class statistics |
| `data/` | `README.md` with the full method and results; small result tables (CSV/JSON/TXT). Raw data and figures are not included: the scripts below regenerate them |

## Reproduce

Requires Python 3.11 with `numpy scipy pandas h5py rasterio pyproj matplotlib scikit-learn earthaccess`, and a NASA [Earthdata Login](https://urs.earthdata.nasa.gov/) in `~/.netrc`.

```bash
python -I tools/download_stack.py            # NISAR L2 GCOV, ~1.7 GB per file (not in this repo)
python -I pipeline/00_fetch_aux.py           # WorldCover, JRC GSW, OPERA DSWx-S1, Sentinel-2
# build the 80 m two-track cache: ipynb/nisar_kakhovka_timeseries.ipynb
python -I tools/eval_rfi_clean.py            # RFI correction + validation  -> data/analysis_rfi_clean/
python -I tools/run_change_rfi.py            # RFI-aware change test
python -I tools/export_web.py                # layers and app.json for the web app
python -I -X utf8 pipeline/selftest_change_rfi.py   # simulation self-test
```

Run the web app locally:

```bash
python tools/serve_static.py --port 8080 --dir web
```

## Data

NISAR L2 GCOV (NASA/JPL, ISRO) · OPERA DSWx-S1 (NASA/JPL) · ESA WorldCover 2021 · JRC Global Surface Water · Sentinel-2 (ESA Copernicus). Basemaps © Esri.

## Ethics

The study area is an active conflict zone. RFI is presented only as a data-quality attribute (intensity relative to signal and a corrupted-data mask); interference sources are not localised or interpreted.

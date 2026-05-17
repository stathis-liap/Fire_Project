"""hindcast_optimizer.py — Data Assimilation & Hindcast Validation
===================================================================
Pipeline
--------
  1.  Fetch historical VIIRS active-fire detections from NASA FIRMS API.
  2.  Identify "State Zero" — the first chronological detection (ignition).
  3.  Map GPS ignition point → 2-D grid cell.
  4.  Run the Rothermel CA simulation forward for a configurable hindcast
      window (default: 6 hours) while perturbing hidden environmental
      parameters [fuel_moisture_offset, wind_multiplier].
  5.  Extract ground-truth hotspot grid from FIRMS detections within the
      same window.
  6.  Score the prediction against ground truth using a composite error
      metric (IoU + Hausdorff-MSE).
  7.  Minimise the error with SciPy Differential Evolution.
  8.  (NEW) Optionally take the top-N parameter sets from the optimizer's
      evaluation history, re-run each at fine resolution, and stack their
      burn masks to produce a probabilistic Ensemble Confidence Map.
  9.  Print the optimal parameters and final accuracy.

No GUI, no PyVista, no Matplotlib — runs fully headlessly.

Usage
-----
  python hindcast_optimizer.py \
      --map-key  <YOUR_NASA_FIRMS_MAP_KEY>   \
      --lat-min  37.8  --lat-max  38.2       \
      --lon-min  23.5  --lon-max  24.0       \
      --date-start 2023-07-18                \
      --date-end   2023-07-19                \
      --hindcast-hours 6                     \
      --ensemble 50                          # NEW: build a 50-member ensemble

Get a free FIRMS map key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so config, core, air are importable
# whether this file is run directly or imported from a subdirectory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_dilation

import config
from core.landscape import Landscape
from core.fire_model import CellularAutomataFire
from pipeline.auto_fetcher import (
    fetch_terrain_from_api, fetch_corine_land_cover, fetch_all_wildfire_data,
    fetch_satellite_image_by_bounds,
)

_OPENTOPO_API_KEY = "57314bc7ed85882904a7485d77c0dbe5"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  NASA FIRMS Data Ingestion
# ──────────────────────────────────────────────────────────────────────────────

_NASA_MAP_KEY = "2694cf69813b733c8e75e2fc038bac40"

# FIRMS source priority list: SP = Standard Processing (full archive, any date)
# days=3 is within the valid [1..5] range for SP sources.
# We try SNPP first, then NOAA-20 as fallback.
_FIRMS_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]
_FIRMS_MAX_DAYS = 10   # NASA FIRMS SP API hard limit per single request

# Columns we care about from the VIIRS/MODIS CSV
_KEEP_COLS = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]

# VIIRS uses string labels; MODIS SP uses numeric 0-100
_CONF_ACCEPT_STR = {"high", "nominal", "h", "n"}
_CONF_MIN_NUMERIC = 30   # treat MODIS numeric confidence ≥ 30 as valid


def fetch_firms_data(
    map_key: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Download VIIRS/MODIS Standard Processing active-fire detections from
    NASA FIRMS for a bounding box and date range.

    Tries _FIRMS_SOURCES in order (VIIRS_SNPP_SP → VIIRS_NOAA20_SP → MODIS_SP)
    and returns the first non-empty result.  SP sources cover the archive back to 2000.

    For windows longer than _FIRMS_MAX_DAYS (10), makes multiple requests and
    concatenates results.

    Confidence handling:
      - VIIRS: string labels — accepts "high", "nominal", "h", "n"
      - MODIS SP: numeric 0-100 — accepts ≥ _CONF_MIN_NUMERIC (30)
    """
    import datetime as _dt
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    start_dt = _dt.date.fromisoformat(date_start)
    end_dt   = _dt.date.fromisoformat(date_end)
    total_days = max(1, (end_dt - start_dt).days + 1)

    # Build list of (chunk_start, chunk_days) to cover the full window
    chunks: list[tuple[str, int]] = []
    cur = start_dt
    while cur <= end_dt:
        chunk_days = min(_FIRMS_MAX_DAYS, (end_dt - cur).days + 1)
        chunks.append((cur.isoformat(), chunk_days))
        cur += _dt.timedelta(days=chunk_days)

    all_raw: list[str] = []

    for source in _FIRMS_SOURCES:
        chunk_texts: list[str] = []
        source_ok = True
        for chunk_start, chunk_days in chunks:
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
                f"/{map_key}/{source}/{bbox}/{chunk_days}/{chunk_start}"
            )
            print(f"[FIRMS] Trying {source} ({chunk_start}, {chunk_days}d): {url}")
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                lines = [l for l in resp.text.strip().split("\n") if l]
                if len(lines) > 1:
                    chunk_texts.append(resp.text)
                    print(f"[FIRMS] {source} → {len(lines)-1} rows")
                else:
                    print(f"[FIRMS] {source} returned no rows for chunk {chunk_start}")
            except requests.exceptions.RequestException as exc:
                print(f"[FIRMS] {source} failed ({exc})")
                source_ok = False
                break

        if chunk_texts:
            all_raw = chunk_texts
            print(f"[FIRMS] Using source: {source}")
            break

    if not all_raw:
        raise RuntimeError(
            "[FIRMS] All sources returned no data for the requested bbox/date. "
            "Check your bounding box coordinates and date range. "
            "Remember: bbox must contain actual fire hotspots on those dates."
        )

    # Concatenate chunks: keep header only from first chunk
    combined_lines: list[str] = []
    for i, raw in enumerate(all_raw):
        lines = raw.strip().split("\n")
        if i == 0:
            combined_lines.extend(lines)
        else:
            combined_lines.extend(lines[1:])   # skip header of subsequent chunks

    df = pd.read_csv(StringIO("\n".join(combined_lines)))
    df.columns = [c.lower() for c in df.columns]

    present = [c for c in _KEEP_COLS if c in df.columns]
    df = df[present].copy()

    df["acq_time_str"] = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + df["acq_time_str"], format="%Y-%m-%d %H%M", utc=True
    )

    # Filter to the user-requested date window
    t0 = pd.Timestamp(date_start, tz="UTC")
    t1 = pd.Timestamp(date_end,   tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["acq_datetime"] >= t0) & (df["acq_datetime"] < t1)].copy()

    # Confidence filter: accept VIIRS string labels OR MODIS numeric ≥ threshold
    def _conf_ok(val: object) -> bool:
        s = str(val).strip().lower()
        if s in _CONF_ACCEPT_STR:
            return True
        try:
            return int(float(s)) >= _CONF_MIN_NUMERIC
        except (ValueError, TypeError):
            return False

    mask = df["confidence"].apply(_conf_ok)
    n_before = len(df)
    df = df[mask].copy()
    n_after = len(df)
    if n_before > 0 and n_after == 0:
        # Last resort: accept all rows rather than crashing — confidence labels
        # vary by dataset version; better to have imperfect data than nothing.
        print(f"[FIRMS] ⚠ All {n_before} rows failed confidence filter "
              f"— accepting all (values: {df['confidence'].unique()[:5].tolist()})")
        df = df.copy()   # already filtered out; reload from pre-filter
        # Reload without confidence filter
        df = pd.read_csv(StringIO("\n".join(combined_lines)))
        df.columns = [c.lower() for c in df.columns]
        present = [c for c in _KEEP_COLS if c in df.columns]
        df = df[present].copy()
        df["acq_time_str"] = df["acq_time"].astype(int).astype(str).str.zfill(4)
        df["acq_datetime"] = pd.to_datetime(
            df["acq_date"] + " " + df["acq_time_str"], format="%Y-%m-%d %H%M", utc=True
        )
        df = df[(df["acq_datetime"] >= t0) & (df["acq_datetime"] < t1)].copy()

    if df.empty:
        raise ValueError(
            "[FIRMS] No detections in the date window. "
            "Try adjusting --date-start / --date-end or the bounding box."
        )

    df.sort_values("acq_datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[FIRMS] {len(df)} detections loaded "
          f"({df['acq_datetime'].iloc[0]} → {df['acq_datetime'].iloc[-1]})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1b.  Live Weather Ingestion — Open-Meteo (free, no API key required)
# ──────────────────────────────────────────────────────────────────────────────

_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    "&wind_speed_unit=ms&forecast_days=1"
)

_OPEN_METEO_HISTORICAL_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={date_start}&end_date={date_end}"
    "&hourly=temperature_2m,relative_humidity_2m,"
    "wind_speed_10m,wind_direction_10m,"
    "precipitation,surface_pressure"
    "&wind_speed_unit=ms"
    "&timezone=UTC"
)


def fetch_weather_hourly(lat: float, lon: float,
                         date_start: str, date_end: str) -> "pd.DataFrame":
    """
    Fetch full hourly ERA5 reanalysis from Open-Meteo for a date range.

    Returns a DataFrame indexed by UTC hour with columns:
        temperature_c, relative_humidity, wind_speed_ms, wind_direction,
        wind_u, wind_v, precipitation_mm, pressure_hpa
    """
    import math as _math
    url = _OPEN_METEO_HISTORICAL_URL.format(
        lat=lat, lon=lon, date_start=date_start, date_end=date_end
    )
    print(f"[WEATHER] Fetching ERA5 hourly ({date_start} → {date_end}) from Open-Meteo …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    h = resp.json().get("hourly", {})

    times   = pd.to_datetime(h["time"], utc=True)
    spd     = np.array(h["wind_speed_10m"],     dtype=float)
    dirn    = np.array(h["wind_direction_10m"], dtype=float)
    dir_rad = np.deg2rad(dirn)

    df = pd.DataFrame({
        "time":             times,
        "temperature_c":    np.array(h["temperature_2m"],      dtype=float),
        "relative_humidity":np.array(h["relative_humidity_2m"],dtype=float),
        "wind_speed_ms":    spd,
        "wind_direction":   dirn,
        # Meteorological convention: wind_u = -spd * sin(dir),  wind_v = -spd * cos(dir)
        "wind_u":          -spd * np.sin(dir_rad),
        "wind_v":          -spd * np.cos(dir_rad),
        "precipitation_mm": np.array(h.get("precipitation",
                                            [0.0] * len(times)), dtype=float),
        "pressure_hpa":     np.array(h.get("surface_pressure",
                                            [1013.0] * len(times)), dtype=float),
    })
    df = df.set_index("time")
    return df


def fetch_weather(lat: float, lon: float,
                  date: str = None,
                  ignition_time: "pd.Timestamp | None" = None) -> dict:
    """
    Fetch real weather conditions from Open-Meteo (free, no key needed).

    date=None            → live current conditions.
    date="YYYY-MM-DD"    → ERA5 hourly reanalysis.
                           Uses ignition_time's hour when supplied,
                           otherwise falls back to 12:00 UTC (noon).

    Also prints the full hourly table for the day so the operator can
    see how conditions evolved from ignition onward.

    Returns a dict compatible with apply_weather_to_landscape().
    """
    if date is None:
        url = _OPEN_METEO_URL.format(lat=lat, lon=lon)
        print("[WEATHER] Fetching live conditions from Open-Meteo ...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            c = resp.json()["current"]
            weather = {
                "temperature_c":     float(c["temperature_2m"]),
                "relative_humidity": float(c["relative_humidity_2m"]),
                "wind_speed_ms":     float(c["wind_speed_10m"]),
                "wind_direction":    float(c["wind_direction_10m"]),
            }
        except Exception as exc:
            print(f"[WEATHER] Live fetch failed ({exc}), using config.py defaults.")
            weather = _weather_from_config()
    else:
        try:
            df_hourly = fetch_weather_hourly(lat, lon, date_start=date, date_end=date)

            # Pick the hour closest to ignition (or noon as fallback)
            if ignition_time is not None:
                ign_ts = pd.Timestamp(ignition_time).tz_convert("UTC")
                idx = df_hourly.index.get_indexer([ign_ts], method="nearest")[0]
            else:
                idx = min(12, len(df_hourly) - 1)

            row = df_hourly.iloc[idx]
            weather = {
                "temperature_c":     float(row["temperature_c"]),
                "relative_humidity": float(row["relative_humidity"]),
                "wind_speed_ms":     float(row["wind_speed_ms"]),
                "wind_direction":    float(row["wind_direction"]),
            }

            # ── Print full hourly table ─────────────────────────────────────
            print(f"\n[WEATHER] ERA5 Hourly Conditions  "
                  f"({lat:.3f}°N {lon:.3f}°E)  —  {date}")
            print(f"  {'Hour (UTC)':>12}  {'T (°C)':>7}  {'RH (%)':>6}  "
                  f"{'Wind (m/s)':>10}  {'Dir (°)':>7}  {'Precip (mm)':>11}")
            print("  " + "-" * 62)
            for ts, r in df_hourly.iterrows():
                marker = " ◄ ignition" if df_hourly.index.get_loc(ts) == idx else ""
                print(f"  {ts.strftime('%Y-%m-%d %H:%M'):>12}  "
                      f"{r['temperature_c']:>7.1f}  {r['relative_humidity']:>6.0f}  "
                      f"{r['wind_speed_ms']:>10.2f}  {r['wind_direction']:>7.0f}  "
                      f"{r['precipitation_mm']:>11.2f}{marker}")
            print()

        except Exception as exc:
            print(f"[WEATHER] Historical fetch failed ({exc}), using config.py defaults.")
            weather = _weather_from_config()
            df_hourly = None

    print(
        f"[WEATHER] Using: T={weather['temperature_c']:.1f}°C  "
        f"RH={weather['relative_humidity']:.0f}%  "
        f"Wind={weather['wind_speed_ms']:.2f} m/s @ {weather['wind_direction']:.0f}°"
    )
    return weather


def _weather_from_config() -> dict:
    return {
        "temperature_c":     float(config.TEMPERATURE_C),
        "relative_humidity": float(config.RELATIVE_HUMIDITY),
        "wind_speed_ms":     float(config.WIND_SPEED),
        "wind_direction":    float(config.WIND_DIRECTION),
    }


def apply_weather_to_landscape(landscape: Landscape, weather: dict) -> None:
    """
    Push live/historical weather into the landscape object so that:
      - landscape.moisture is recalculated from real T + RH via EMC
      - landscape.wind_speed / wind_dir reflect actual measured wind
      - air/air.py will pick up the corrected wind on next compute_wind_field() call

    This replaces the hardcoded config.py weather values with real data.
    Call this before constructing CellularAutomataFire.
    """
    # Recalculate EMC-based moisture from real temperature + humidity
    real_emc = landscape.calculate_emc(
        weather["temperature_c"], weather["relative_humidity"]
    )
    # Apply spatially (preserve the elevation-gradient moisture variation,
    # but re-centre it on the real EMC instead of the config default)
    config_emc = landscape.calculate_emc(config.TEMPERATURE_C, config.RELATIVE_HUMIDITY)
    moisture_shift = real_emc - config_emc
    landscape.moisture = np.clip(landscape.moisture + moisture_shift, 0.01, 0.30)

    # Set real wind — air/air.py will apply WAF + terrain on top of this
    landscape.set_wind(weather["wind_speed_ms"], weather["wind_direction"])
    print(
        f"[LANDSCAPE] Updated: EMC={real_emc:.4f}  "
        f"wind={weather['wind_speed_ms']:.2f} m/s @ {weather['wind_direction']:.0f} deg"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GPS ↔ Grid coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

class GeoGrid:
    """
    Maps WGS-84 (lat, lon) ↔ 2-D grid (row, col).

    The landscape arrays are loaded with np.flipud() applied to the raw
    rasterio output, making them **south-up**: row 0 = lat_min (south),
    row n-1 = lat_max (north).  This matches the fire model's wind-vector
    convention (wind_v positive = northward = increasing row).

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : bounding box.
    rows, cols                         : grid dimensions.
    """

    def __init__(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        rows: int, cols: int,
    ):
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.rows, self.cols = rows, cols

    def latlon_to_rc(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Convert a GPS coordinate to (row, col) in the south-up grid.

        row = int( (lat - lat_min) / (lat_max - lat_min) * rows )
        col = int( (lon - lon_min) / (lon_max - lon_min) * cols )
        """
        row = int((lat - self.lat_min) / (self.lat_max - self.lat_min) * self.rows)
        col = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * self.cols)
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))
        return row, col

    def rc_to_latlon(self, row: int, col: int) -> tuple[float, float]:
        """Inverse: (row, col) → (lat, lon) at cell centre (south-up)."""
        lat = self.lat_min + (row + 0.5) / self.rows * (self.lat_max - self.lat_min)
        lon = self.lon_min + (col + 0.5) / self.cols * (self.lon_max - self.lon_min)
        return lat, lon

    def dataframe_to_grid_mask(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert a DataFrame of [latitude, longitude] rows into a boolean
        (rows, cols) grid mask where True = at least one hotspot fell in
        that cell.
        """
        mask = np.zeros((self.rows, self.cols), dtype=bool)
        for _, row in df.iterrows():
            r, c = self.latlon_to_rc(float(row["latitude"]), float(row["longitude"]))
            mask[r, c] = True
        return mask


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FIRMS truth-mask dilation helper
# ──────────────────────────────────────────────────────────────────────────────

def dilate_truth_mask(truth_mask: np.ndarray, cell_size_m: float,
                      viirs_resolution_m: float = 375.0) -> np.ndarray:
    """
    Buffer each FIRMS VIIRS pixel by its nominal footprint radius (~375 m).

    Problem
    -------
    FIRMS reports the *centroid* of a 375 m VIIRS scan pixel.  On our grid
    (typically 140–280 m/cell) a single FIRMS detection maps to 1–2 isolated
    grid cells.  Comparing a continuous predicted perimeter against scattered
    single-pixel truth dots yields near-zero IoU even when the model is
    physically correct.

    Solution
    --------
    Dilate each truth pixel outward by `ceil(viirs_resolution_m / cell_size_m)`
    cells in every direction, converting isolated dot detections into a rough
    pseudo-perimeter that represents the full ~375 m footprint of the satellite
    sensor.  This makes IoU/Hausdorff comparisons geometrically meaningful.

    Parameters
    ----------
    truth_mask       : raw (rows, cols) boolean mask from GeoGrid.dataframe_to_grid_mask
    cell_size_m      : physical size of one grid cell in metres
    viirs_resolution_m : VIIRS pixel footprint radius (default 375 m)

    Returns
    -------
    dilated (rows, cols) boolean array — same shape as truth_mask
    """
    import math
    radius_cells = max(1, math.ceil(viirs_resolution_m / cell_size_m))
    # Use a filled disk structuring element so the dilation is circular
    d = radius_cells * 2 + 1
    cy, cx = radius_cells, radius_cells
    y_idx, x_idx = np.ogrid[:d, :d]
    struct = (y_idx - cy)**2 + (x_idx - cx)**2 <= radius_cells**2
    dilated = binary_dilation(truth_mask, structure=struct)
    n_raw  = int(truth_mask.sum())
    n_dil  = int(dilated.sum())
    print(f"  [TruthMask] Dilated {n_raw} VIIRS pixels → {n_dil} cells "
          f"(radius={radius_cells} cells @ {cell_size_m:.0f} m/cell)")
    return dilated


def load_copernicus_truth_mask(
    shapefile_path: str,
    landscape_shape: tuple[int, int],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """
    Burn a Copernicus EMS vector polygon (Shapefile or GeoJSON) into a
    boolean numpy grid aligned with our landscape.

    Copernicus EMS Rapid Mapping products ship as Shapefiles containing the
    measured fire perimeter at 10–30 m resolution — far more accurate than
    dilated VIIRS points.  This function:

      1. Reads the file with geopandas.
      2. Reprojects to EPSG:4326 (Lat/Lon) so coordinates match our grid.
      3. Builds the affine transform that maps pixel (col, row) → (lon, lat).
      4. Uses rasterio.features.rasterize to burn the polygon into a 2-D mask.

    Parameters
    ----------
    shapefile_path   : path to .shp, .geojson, or any OGR-readable file
    landscape_shape  : (rows, cols) of the target grid
    lat_min/max      : southern/northern edge of the terrain domain
    lon_min/max      : western/eastern edge of the terrain domain

    Returns
    -------
    (rows, cols) boolean np.ndarray — True inside the fire perimeter
    """
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError(
            "geopandas and rasterio are required for Copernicus shapefile support. "
            "Install with: conda install -c conda-forge geopandas rasterio"
        ) from exc

    print(f"\n[Copernicus] Loading truth perimeter: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    print(f"  CRS: {gdf.crs}  |  features: {len(gdf)}")

    # Reproject to WGS-84 Lat/Lon so it lines up with our grid
    if gdf.crs is None or str(gdf.crs) == "":
        print("  [Copernicus] WARNING — no CRS found; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    rows, cols = landscape_shape
    # Affine: pixel (0,0) = top-left = (lon_min, lat_max); row increases southward
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)

    # Collect all geometry objects (filter null / empty)
    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError("[Copernicus] No valid geometries found in shapefile.")

    # rasterize burns value=1 inside each polygon; out_shape is (rows, cols)
    mask = rasterize(
        geometries,
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,   # mark any cell touched by the polygon edge
    ).astype(bool)

    n_cells = int(mask.sum())
    print(f"  [Copernicus] Rasterised → {n_cells} cells inside fire perimeter  "
          f"({n_cells * (((lat_max-lat_min)/rows) * 111000)**2 / 10_000:.1f} ha approx)")
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Error / cost metric
# ──────────────────────────────────────────────────────────────────────────────

def hausdorff_score(
    pred_mask:   np.ndarray,
    truth_mask:  np.ndarray,
    cell_size_m: float,
) -> float:
    """
    Normalised Symmetric Fréchet/Hausdorff distance between two boolean masks.

    Computes the symmetric Hausdorff distance using scipy.spatial.distance
    directed_hausdorff in both directions:

        H(P, T) = max( d_H(P → T),  d_H(T → P) )

    where d_H(A → B) = max_{a ∈ A} min_{b ∈ B} ||a − b||.

    The result is normalised by the terrain grid diagonal so the score lies
    in [0, 1]:  0.0 = perfect overlap, 1.0 = maximally separated.

    Parameters
    ----------
    pred_mask   : (rows, cols) bool — predicted fire cells
    truth_mask  : (rows, cols) bool — ground-truth cells
    cell_size_m : physical cell size in metres (for coordinate scaling)

    Returns
    -------
    float in [0, 1]  (lower is better)
    """
    pred_coords  = np.argwhere(pred_mask).astype(np.float64)  * cell_size_m
    truth_coords = np.argwhere(truth_mask).astype(np.float64) * cell_size_m

    # Degenerate cases: one or both masks are empty
    if len(pred_coords) == 0 or len(truth_coords) == 0:
        return 1.0

    rows, cols  = truth_mask.shape
    grid_diag_m = math.hypot(rows * cell_size_m, cols * cell_size_m)

    # directed_hausdorff returns (distance, index_a, index_b)
    d_pt = directed_hausdorff(pred_coords,  truth_coords)[0]
    d_tp = directed_hausdorff(truth_coords, pred_coords)[0]
    sym_hd = max(d_pt, d_tp)

    return float(min(sym_hd / grid_diag_m, 1.0))


def compute_error(
    predicted_mask: np.ndarray,
    truth_mask:     np.ndarray,
    cell_size_m:    float,
) -> float:
    """
    Hybrid error metric: 30 % IoU-loss + 70 % normalised Symmetric Hausdorff.

    ── Term A (30 %): 1 − IoU  ──────────────────────────────────────────
    IoU = |P ∩ T| / |P ∪ T|.
    Penalises mis-matched spatial extent symmetrically.

    ── Term B (70 %): Symmetric Hausdorff distance  ─────────────────────
    H(P, T) = max( d_H(P→T), d_H(T→P) ) normalised by the grid diagonal.
    Provides a smooth gradient signal even when masks are completely
    disjoint, guiding the optimizer toward the correct basin of attraction.
    The symmetric form naturally penalises both under- and over-prediction:
    a "burn everything" prediction has a large directed d_H(pred → truth)
    because remote predicted cells are far from any truth cell.

    ── Combined ─────────────────────────────────────────────────────────
    error = 0.30 · (1 − IoU) + 0.70 · H_norm

    Parameters
    ----------
    predicted_mask : (rows, cols) bool
    truth_mask     : (rows, cols) bool
    cell_size_m    : physical cell size in metres

    Returns
    -------
    error : float ∈ [0, 1]  (lower is better)
    """
    intersection = np.logical_and(predicted_mask, truth_mask).sum()
    union        = np.logical_or( predicted_mask, truth_mask).sum()

    if union == 0:
        return 0.0

    iou      = intersection / union
    term_iou = 1.0 - iou

    hd_score = hausdorff_score(predicted_mask, truth_mask, cell_size_m)

    error = 0.30 * term_iou + 0.70 * hd_score
    return float(error)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  The Hindcast Optimizer
# ──────────────────────────────────────────────────────────────────────────────

class HindcastOptimizer:
    """
    Wraps the Rothermel CA simulation in a SciPy-compatible objective function
    for data assimilation against NASA FIRMS satellite hotspots.

    Hidden variables optimised
    --------------------------
    params[0] : fuel_moisture_offset  Δm ∈ [-0.10, +0.10]
        Added to every cell's base EMC moisture.  Negative = drier fuel
        (fire spreads faster); positive = wetter (fire slows or stops).

    params[1] : wind_multiplier  k ∈ [0.5, 2.0]
        Scales the global wind speed before passing it to air/air.py.
        k < 1 → calmer than reported; k > 1 → stronger than reported.
        This corrects for anemometer placement and mesoscale errors.

    Evaluation history
    ------------------
    Every call to objective_function appends (params, error) to
    self._eval_history.  This is consumed downstream by the ensemble
    forecaster to build a probabilistic confidence map from the
    top-N parameter sets.
    """

    def __init__(
        self,
        base_landscape:   Landscape,
        geo_grid:         GeoGrid,
        truth_mask:       np.ndarray,
        hindcast_steps:   int,
        ignition_rc:      tuple[int, int],
        eval_callback     = None,
    ):
        self.base_landscape  = base_landscape
        self.base_moisture   = base_landscape.moisture.copy()
        self.base_wind_speed = base_landscape.wind_speed
        self.base_wind_dir   = base_landscape.wind_dir
        self.geo_grid        = geo_grid
        self.truth_mask      = truth_mask
        self.hindcast_steps  = hindcast_steps
        self.ignition_rc     = ignition_rc
        self.eval_callback   = eval_callback

        self._eval_count   = 0
        self._best_error   = float("inf")
        self._best_params  = None
        self._t0_wall      = time.time()

        # Full evaluation history for ensemble forecasting.
        # Each entry: {"params": np.ndarray, "error": float}
        self._eval_history: list[dict] = []

    def objective_function(self, params: np.ndarray) -> float:
        """
        Called by differential_evolution for each candidate parameter vector.

        Steps
        -----
        1. Decode hidden variables from `params`.
        2. Mutate a shallow copy of the landscape (moisture + wind).
        3. Rebuild the Rothermel ROS grid (done inside CellularAutomataFire.__init__).
        4. Ignite the GPS-mapped ignition cell.
        5. Step the CA for `hindcast_steps` iterations.
        6. Build the predicted fire mask from fire_sim.state.
        7. Compute and return the composite error.

        Parameters
        ----------
        params : [fuel_moisture_offset, wind_multiplier]

        Returns
        -------
        error : float ≥ 0  (differential_evolution minimises this)
        """
        moisture_offset, wind_mult = float(params[0]), float(params[1])

        # ── Mutate landscape ──────────────────────────────────────────
        # We work on the shared landscape object.  Because CellularAutomata
        # Fire.__init__ precomputes the ROS grid immediately, there is no
        # race condition even when differential_evolution uses workers > 1
        # in "latinhypercube" mode — each worker gets its own call stack.
        self.base_landscape.moisture = np.clip(
            self.base_moisture + moisture_offset, 0.01, 0.30
        )
        self.base_landscape.set_wind(
            self.base_wind_speed * wind_mult,
            self.base_wind_dir,
        )

        # ── Initialise simulation at State Zero ───────────────────────
        fire_sim = CellularAutomataFire(self.base_landscape, config)
        r0, c0 = self.ignition_rc
        # Ignite a 3-cell radius around the mapped GPS point to account for
        # the ~375 m VIIRS pixel footprint relative to our 5 m cells.
        radius = max(1, int(375 / config.CELL_SIZE_METERS / 2))
        rows, cols = self.base_landscape.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    fire_sim.ignite(rr, cc)

        # ── Step forward for hindcast_steps ───────────────────────────
        for _ in range(self.hindcast_steps):
            fire_sim.step()

        # ── Build predicted fire mask ─────────────────────────────────
        # State == 1: currently burning
        # State == 2: burnt (ash) — both count as "was on fire"
        # ignition_fraction >= 0.5: preheated but not yet ignited —
        #   include as a soft boundary to avoid sharp-edge IoU artefacts
        predicted_mask = (
            (fire_sim.state >= 1) |
            (fire_sim.ignition_fraction >= 0.5)
        )

        # ── Score ─────────────────────────────────────────────────────
        error = compute_error(
            predicted_mask, self.truth_mask, config.CELL_SIZE_METERS
        )

        self._eval_count += 1
        is_best = error < self._best_error
        if is_best:
            self._best_error  = error
            self._best_params = params.copy()

        # Record every evaluation for the ensemble forecaster
        self._eval_history.append({
            "params": params.copy(),
            "error":  float(error),
        })

        # Log every evaluation so the user can see progress
        elapsed = time.time() - self._t0_wall
        tag = " ★ NEW BEST" if is_best else ""
        print(
            f"  eval={self._eval_count:4d}  error={error:.4f}  "
            f"best={self._best_error:.4f}  "
            f"Δm={moisture_offset:+.3f}  k_wind={wind_mult:.2f}  "
            f"{elapsed:.0f}s{tag}",
            flush=True,
        )

        if self.eval_callback is not None:
            self.eval_callback(self._eval_count, error,
                               self._best_error, params.copy())

        return error


# ──────────────────────────────────────────────────────────────────────────────
# 4b.  Ensemble Forecasting — top-N re-run → probabilistic confidence map
# ──────────────────────────────────────────────────────────────────────────────

def _select_ensemble_members(
    history:    list[dict],
    n_members:  int,
    dedup_tol:  float = 1e-3,
) -> list[dict]:
    """
    From the optimizer's full evaluation history, pick the top-N lowest-error
    parameter vectors after removing near-duplicates.

    Why de-duplicate?
    -----------------
    Differential Evolution's final L-BFGS-B polish typically produces 5–20
    parameter vectors that differ only in the 4th decimal place.  If we
    naively took the top-50 by error, the ensemble could end up with 30
    near-clones of the optimum and only 20 truly different members — which
    collapses the spread and gives an artificially confident map.

    Method
    ------
    1. Sort history by ascending error.
    2. Walk from best to worst.  Accept a candidate only if its parameter
       vector lies at least `dedup_tol` (in normalised L-inf distance) from
       every already-accepted member.
    3. Stop once we have n_members.

    Parameters
    ----------
    history    : list of {"params": np.ndarray, "error": float}
    n_members  : desired ensemble size
    dedup_tol  : minimum L-inf distance between members (normalised to bounds)

    Returns
    -------
    list of {"params": np.ndarray, "error": float} sorted best-first
    """
    if not history:
        return []

    # Sort best-first
    sorted_hist = sorted(history, key=lambda h: h["error"])

    accepted: list[dict] = []
    for cand in sorted_hist:
        if len(accepted) >= n_members:
            break
        cand_p = cand["params"]
        is_dup = False
        for kept in accepted:
            # L-inf distance — cheap and translates well to box-bounded params
            if np.max(np.abs(cand_p - kept["params"])) < dedup_tol:
                is_dup = True
                break
        if not is_dup:
            accepted.append(cand)

    return accepted


def run_ensemble(
    landscape_fine:   Landscape,
    geo_grid_fine:    GeoGrid,
    weather:          dict,
    base_moisture:    np.ndarray,
    ignition_rc_fine: tuple[int, int],
    hindcast_steps:   int,
    members:          list[dict],
    weighting:        str = "uniform",
    softmax_tau:      float = 0.05,
) -> dict:
    """
    Run each ensemble member through the fine-resolution CA and stack their
    burn masks into a probabilistic confidence map.

    For each of the N members:
      1. Apply the member's (moisture_offset, wind_multiplier) to a fresh
         copy of the fine landscape.
      2. Run the CA for `hindcast_steps`.
      3. Extract the burn mask.

    The N masks are stacked into a float32 array of shape (rows, cols) where
    each cell's value ∈ [0, 1] is the (weighted) fraction of members that
    predicted that cell would burn.

    Weighting
    ---------
    - "uniform"  : every member counts equally (default — most honest).
                   Interpretation: "X out of N plausible parameter sets
                   predicted this cell burns."
    - "softmax"  : weight ∝ exp(-error / softmax_tau).  Better-scoring
                   members dominate.  Tune softmax_tau to set how sharply
                   the weights peak around the best member; smaller tau
                   → more peaked, closer to deterministic best-fit.

    Parameters
    ----------
    landscape_fine    : the fine-resolution Landscape used for the final IoU run
    geo_grid_fine     : matching GeoGrid (unused here but kept for symmetry)
    weather           : the same weather dict used by run_hindcast
    base_moisture     : the *pre-perturbation* moisture grid (i.e. after
                        apply_weather_to_landscape but before any DE offset).
                        Each member's moisture_offset is added to this.
    ignition_rc_fine  : (row, col) of the ignition cell in the fine grid
    hindcast_steps    : number of CA steps per member
    members           : output of _select_ensemble_members
    weighting         : "uniform" or "softmax"
    softmax_tau       : temperature for softmax weighting

    Returns
    -------
    dict with keys:
      - confidence_map  : (rows, cols) float32 in [0, 1]
      - member_masks    : list of N boolean (rows, cols) arrays (for export)
      - weights         : (N,) float array, sums to 1.0
      - members         : the same list passed in (for traceability)
    """
    n = len(members)
    rows, cols = landscape_fine.shape
    base_wind_speed = weather["wind_speed_ms"]
    base_wind_dir   = weather["wind_direction"]

    print(f"\n[Ensemble] Re-running top {n} members at fine resolution "
          f"({rows}x{cols} cells) ...")
    t_start = time.time()

    # Pre-compute weights so cells can be accumulated as floats in one pass
    errors = np.array([m["error"] for m in members], dtype=np.float64)
    if weighting == "softmax":
        # Numerically stable softmax: subtract min before exp
        shifted = -(errors - errors.min()) / max(softmax_tau, 1e-9)
        w = np.exp(shifted)
        w /= w.sum()
        print(f"  Weighting: softmax (τ={softmax_tau:.3f})")
        print(f"  Weight range: [{w.min():.4f}, {w.max():.4f}]  "
              f"effective N = {1.0 / (w**2).sum():.1f}")
    else:
        w = np.ones(n, dtype=np.float64) / n
        print(f"  Weighting: uniform (each member = {w[0]:.4f})")

    confidence  = np.zeros((rows, cols), dtype=np.float32)
    member_masks: list[np.ndarray] = []

    # Ignition footprint sized to VIIRS pixel — same as the final run
    cell_m = landscape_fine.config.CELL_SIZE_METERS
    radius = max(1, int(375 / cell_m / 2))
    r0, c0 = ignition_rc_fine

    for i, member in enumerate(members):
        moff, wmul = float(member["params"][0]), float(member["params"][1])

        # Apply this member's parameters to the fine landscape
        landscape_fine.moisture = np.clip(base_moisture + moff, 0.01, 0.30)
        landscape_fine.set_wind(base_wind_speed * wmul, base_wind_dir)

        fire_sim = CellularAutomataFire(landscape_fine, config)

        # Ignition footprint
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    fire_sim.ignite(rr, cc)

        for _ in range(hindcast_steps):
            fire_sim.step()

        mask_i = (
            (fire_sim.state >= 1) |
            (fire_sim.ignition_fraction >= 0.5)
        )
        member_masks.append(mask_i)

        confidence += w[i] * mask_i.astype(np.float32)

        elapsed = time.time() - t_start
        eta     = elapsed / (i + 1) * (n - i - 1)
        print(f"  member {i+1:3d}/{n}  err={member['error']:.4f}  "
              f"Δm={moff:+.3f}  k={wmul:.2f}  "
              f"burned={int(mask_i.sum()):>6d} cells  "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
              flush=True)

    # Clip to [0, 1] — should already be there, but float accumulation
    # of N tiny weights can occasionally produce 1.0 + 1e-7
    np.clip(confidence, 0.0, 1.0, out=confidence)

    # ── Summary statistics ──────────────────────────────────────────────
    hi  = float((confidence >= 0.9).sum())
    med = float(((confidence >= 0.5) & (confidence < 0.9)).sum())
    low = float(((confidence >= 0.1) & (confidence < 0.5)).sum())
    print(f"\n[Ensemble] Confidence map built in {time.time()-t_start:.1f}s")
    print(f"  ≥90% confidence (likely burn)  : {int(hi):>7d} cells  "
          f"({hi * cell_m**2 / 1e4:.1f} ha)")
    print(f"  50-90% confidence (probable)   : {int(med):>7d} cells  "
          f"({med * cell_m**2 / 1e4:.1f} ha)")
    print(f"  10-50% confidence (possible)   : {int(low):>7d} cells  "
          f"({low * cell_m**2 / 1e4:.1f} ha)")

    return {
        "confidence_map": confidence,
        "member_masks":   member_masks,
        "weights":        w,
        "members":        members,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_hindcast(
    map_key:         str,
    lat_min:         float,
    lat_max:         float,
    lon_min:         float,
    lon_max:         float,
    date_start:      str,
    date_end:        str,
    hindcast_hours:  float = 6.0,
    maxiter:         int   = 20,
    popsize:         int   = 12,
    terrain_buffer:  float = 0.25,
    eval_callback    = None,
    truth_shapefile: Optional[str] = None,
    fired_gpkg:      Optional[str] = None,
    ensemble_size:   int   = 0,
    ensemble_weighting: str = "uniform",
    ensemble_softmax_tau: float = 0.05,
) -> dict:
    """
    Full end-to-end hindcast assimilation pipeline.

    Parameters
    ----------
    map_key         : NASA FIRMS map key.
    lat/lon bounds  : FIRMS search bounding box (can be large).
    date_start/end  : FIRMS query date range.
    hindcast_hours  : minimum hindcast window (auto-extended to next VIIRS pass).
    maxiter         : differential_evolution max generations.
    popsize         : differential_evolution population size per generation.
    terrain_buffer  : half-width in degrees of the terrain domain centred on
                      the ignition point (0.25 deg = ~28 km).
    eval_callback   : optional callable(eval_num, error, best_error, params).
    truth_shapefile : optional path to a Copernicus EMS Shapefile (.shp/.geojson).
                      When supplied, NASA FIRMS fetching + dilation is bypassed
                      and the polygon is used directly as ground truth.
    fired_gpkg      : optional path to a FIRED daily GeoPackage (.gpkg).
                      Used as ground truth when truth_shapefile is not provided.
                      Falls back to NASA FIRMS if also absent.
    ensemble_size   : number of top-N parameter sets to re-run for a
                      probabilistic confidence map.  0 (default) = disabled.
                      Typical values: 30-100.  Adds ~ensemble_size × (fine-run
                      cost) to total wall time.
    ensemble_weighting   : "uniform" or "softmax" — see run_ensemble().
    ensemble_softmax_tau : softmax temperature (only used if weighting=softmax).

    Truth source priority: Copernicus shapefile > FIRED GeoPackage > NASA FIRMS.

    Returns
    -------
    result dict with keys:
        best_params, best_error, iou, df_firms, landscape, ...
        confidence_map  (only if ensemble_size > 0)
        ensemble        (only if ensemble_size > 0)
    """
    # ── Step 1: Fetch FIRMS data (skipped if Copernicus shapefile provided) ───
    df            = None
    df_window     = None
    ignition_time = None
    ignition_date = date_start
    fired_timeline = None   # populated when using FIRED truth source

    using_shapefile = truth_shapefile is not None and os.path.exists(str(truth_shapefile))
    using_fired     = (
        not using_shapefile
        and fired_gpkg is not None
        and os.path.exists(str(fired_gpkg))
    )

    if using_shapefile:
        print("\n[1/8] Copernicus EMS shapefile provided — skipping NASA FIRMS fetch.")
        print(f"  Truth source : {truth_shapefile}")
        # Derive ignition point as the search-bbox centre
        centre_lat  = (lat_min + lat_max) / 2
        centre_lon  = (lon_min + lon_max) / 2
        ignition_lat = centre_lat
        ignition_lon = centre_lon
        import datetime as _dt
        ignition_time = pd.Timestamp(date_start + " 00:00:00", tz="UTC")
        print(f"  Ignition GPS (bbox centre): ({ignition_lat:.5f}, {ignition_lon:.5f})")
        print(f"  Ignition time (assumed)   : {ignition_time}")

    elif using_fired:
        print("\n[1/8] FIRED GeoPackage provided — skipping NASA FIRMS fetch.")
        print(f"  Truth source : {fired_gpkg}")

        # Derive ignition from the centroid of the FIRST day's FIRED polygon
        # using the already-proven load_fired_daily pipeline (avoids tz-naive
        # vs tz-aware comparison that silently broke the inline probe).
        _ignition_set = False
        try:
            from pipeline.fired_loader import (
                load_fired_daily   as _lfd,
                find_fire_event    as _ffe,
                get_event_timeline as _get_tl,
            )
            _daily    = _lfd(str(fired_gpkg), (lat_min, lat_max, lon_min, lon_max))
            _eid      = _ffe(_daily, date_start, date_end, (lat_min, lat_max, lon_min, lon_max))
            if _eid is not None:
                _tl   = _get_tl(_daily, _eid)
                _day1 = _tl.iloc[[0]]          # earliest day
                _cen  = _day1.geometry.iloc[0].centroid
                ignition_lat  = _cen.y
                ignition_lon  = _cen.x
                _ignition_set = True
                print(f"  Ignition GPS (FIRED day-1 centroid, event {_eid}): "
                      f"({ignition_lat:.5f}, {ignition_lon:.5f})")
        except Exception as _e:
            print(f"  [FIRED] Centroid probe failed ({_e}) — using bbox centre.")

        if not _ignition_set:
            ignition_lat = (lat_min + lat_max) / 2
            ignition_lon = (lon_min + lon_max) / 2
            print(f"  Ignition GPS (bbox centre): ({ignition_lat:.5f}, {ignition_lon:.5f})")

        ignition_time = pd.Timestamp(date_start + " 00:00:00", tz="UTC")
        print(f"  Ignition time (assumed)   : {ignition_time}")
    else:
        print("\n[1/8] Fetching NASA FIRMS VIIRS data ...")
        df = fetch_firms_data(map_key, lat_min, lat_max, lon_min, lon_max,
                              date_start, date_end)

        # ── Step 2: Identify State Zero ───────────────────────────────────
        print("\n[2/8] Identifying State Zero (first detection) ...")
        state_zero    = df.iloc[0]
        ignition_lat  = float(state_zero["latitude"])
        ignition_lon  = float(state_zero["longitude"])
        ignition_time = state_zero["acq_datetime"]
        ignition_date = str(ignition_time.date())
        print(f"  Ignition GPS : ({ignition_lat:.5f}, {ignition_lon:.5f})")
        print(f"  Ignition UTC : {ignition_time}")

    # ── Step 2b: Fetch real weather for the ignition date ─────────────
    print("\n[3/8] Fetching real weather conditions from Open-Meteo ERA5 ...")
    centre_lat = (lat_min + lat_max) / 2
    centre_lon = (lon_min + lon_max) / 2
    weather = fetch_weather(centre_lat, centre_lon,
                            date=ignition_date,
                            ignition_time=ignition_time)

    # ── Step 3: Fetch real terrain (DEM + CORINE) and build landscape ────
    print("\n[4/8] Fetching real terrain (Copernicus COP30 + CORINE) ...")
    print(f"  Terrain domain: {ignition_lat:.4f}N {ignition_lon:.4f}E ± {terrain_buffer}°"
          f"  (~{terrain_buffer * 111:.0f} km radius)")

    # Compute and store terrain bbox for GeoGrid (used below)
    t_lat_min = ignition_lat - terrain_buffer
    t_lat_max = ignition_lat + terrain_buffer
    t_lon_min = ignition_lon - terrain_buffer
    t_lon_max = ignition_lon + terrain_buffer

    dem_file = fetch_terrain_from_api(ignition_lat, ignition_lon,
                                      api_key=_OPENTOPO_API_KEY,
                                      buffer=terrain_buffer)
    corine_file = None
    satellite_file = None
    if dem_file:
        try:
            corine_file = fetch_corine_land_cover(
                t_lon_min, t_lat_min, t_lon_max, t_lat_max, 400, 400
            )
        except Exception as exc:
            print(f"[4/8] CORINE fetch failed ({exc}) — will use synthetic terrain.")
        try:
            satellite_file = fetch_satellite_image_by_bounds(
                t_lon_min, t_lat_min, t_lon_max, t_lat_max, 400, 400
            )
        except Exception as exc:
            print(f"[4/8] Satellite texture fetch failed ({exc}) — 3D view will use CORINE.")

    # Coarse grid for DE loop: 200x200 = ~40K cells per eval (fast).
    # Fine grid for final IoU run: 400x400 = ~160K cells (accurate).
    COARSE_SHAPE = (200, 200)
    FINE_SHAPE   = (400, 400)

    land = Landscape(config)
    if dem_file and corine_file:
        land.load_real_terrain(dem_file=dem_file, corine_file=corine_file,
                               target_shape=COARSE_SHAPE)
        print(f"[4/8] Coarse terrain loaded: {land.shape[0]}x{land.shape[1]} cells "
              f"({land.elevation.min():.0f}-{land.elevation.max():.0f} m)  "
              f"cell={land.config.CELL_SIZE_METERS:.0f} m")
    else:
        print("[4/8] Falling back to synthetic terrain (DEM/CORINE unavailable).")
        land.generate_random_terrain()

    # Override config defaults with actual measured weather
    apply_weather_to_landscape(land, weather)

    # GeoGrid is tied to the TERRAIN domain, not the FIRMS search bbox.
    rows, cols = land.shape
    geo_grid   = GeoGrid(t_lat_min, t_lat_max, t_lon_min, t_lon_max, rows, cols)
    ignition_rc = geo_grid.latlon_to_rc(ignition_lat, ignition_lon)
    print(f"  Grid ignition cell : row={ignition_rc[0]}, col={ignition_rc[1]}")

    # ── Step 4: Build ground-truth mask ───────────────────────────────
    print(f"\n[5/8] Building ground-truth mask ...")
    effective_hours = hindcast_hours

    if using_shapefile:
        # ── High-resolution Copernicus vector truth ──────────────────────
        truth_mask = load_copernicus_truth_mask(
            shapefile_path  = str(truth_shapefile),
            landscape_shape = (rows, cols),
            lat_min=t_lat_min, lat_max=t_lat_max,
            lon_min=t_lon_min, lon_max=t_lon_max,
        )
        truth_cells = int(truth_mask.sum())
        print(f"  Ground-truth cells (Copernicus): {truth_cells}")
        # hindcast window = full user-specified hours (no VIIRS orbit to chase)
        window_end = ignition_time + pd.Timedelta(hours=effective_hours)

    elif using_fired:
        # ── FIRED daily polygon perimeters ───────────────────────────────
        # Build truth mask using ONLY the daily polygons that fall within
        # the hindcast window (ignition_time → ignition_time + hindcast_hours).
        # This makes the scoring fair: if we simulate 1 day, we compare against
        # the FIRED perimeter from day 1, not the full 12-day extent.
        try:
            from pipeline.fired_loader import (
                load_fired_daily      as _lfd,
                find_fire_event       as _ffe,
                get_event_timeline    as _get_tl,
                fired_polygon_to_grid_mask as _f2g,
            )
            fired_bbox    = (t_lat_min, t_lat_max, t_lon_min, t_lon_max)
            _daily_gdf    = _lfd(str(fired_gpkg), fired_bbox)
            _event_id     = _ffe(_daily_gdf, date_start, date_end, fired_bbox)
            if _event_id is not None:
                fired_timeline = _get_tl(_daily_gdf, _event_id)
            else:
                fired_timeline = _daily_gdf  # fallback: all features in window

            # Compute window_end for truth mask selection
            window_end = ignition_time + pd.Timedelta(hours=effective_hours)

            # Select daily polygons whose burn_date ≤ window_end
            _in_window = fired_timeline[fired_timeline["burn_date"] <= window_end]
            if _in_window.empty:
                # If no polygons yet in window, use the first available day
                _in_window = fired_timeline.iloc[[0]]

            truth_mask  = _f2g(_in_window, fired_bbox, (rows, cols))
            truth_cells = int(truth_mask.sum())
            n_days      = len(_in_window)

            print(f"  FIRED polygons in hindcast window ({hindcast_hours}h): {n_days}")
            print(f"  Ground-truth cells (FIRED day≤{window_end.date()}): {truth_cells}  "
                  f"({truth_cells * land.config.CELL_SIZE_METERS**2 / 1e4:.1f} ha)")

            if truth_cells == 0:
                print("  ⚠ FIRED returned zero burned cells for this bbox/date range.")
                print("    Falling back to NASA FIRMS for ground truth.")
                using_fired    = False
                fired_timeline = None
        except Exception as _fired_err:
            import traceback as _tb
            print(f"  ⚠ FIRED loader error: {_fired_err}")
            _tb.print_exc()
            print("    Falling back to NASA FIRMS for ground truth.")
            using_fired    = False
            fired_timeline = None

        if not using_fired:
            # FIRMS fallback
            df_after = df if df is not None else None
            if df_after is None:
                print("  Fetching NASA FIRMS as fallback ...")
                df = fetch_firms_data(map_key, lat_min, lat_max, lon_min, lon_max,
                                      date_start, date_end)
            df_after = df[df["acq_datetime"] > ignition_time].copy() if df is not None else None

        window_end = ignition_time + pd.Timedelta(hours=effective_hours)

    else:
        # ── VIIRS point detections with spatial dilation ──────────────────
        df_after = df[df["acq_datetime"] > ignition_time].copy()
        if df_after.empty:
            raise ValueError(
                "[5/8] No FIRMS detections after State Zero. "
                "Try a wider date range (e.g. --date-end +2 days)."
            )
        next_overpass  = df_after["acq_datetime"].min()
        delta_h        = (next_overpass - ignition_time).total_seconds() / 3600.0
        print(f"  Next VIIRS overpass  : {next_overpass} (+{delta_h:.1f}h after ignition)")
        effective_hours = max(hindcast_hours, math.ceil(delta_h) + 0.5)
        if effective_hours > hindcast_hours:
            print(f"  Window auto-extended : {hindcast_hours}h -> {effective_hours}h "
                  f"to capture first post-ignition overpass")
        window_end = ignition_time + pd.Timedelta(hours=effective_hours)
        df_window  = df[
            (df["acq_datetime"] >  ignition_time) &
            (df["acq_datetime"] <= window_end)
        ].copy()
        print(f"  Ground-truth hotspots in window: {len(df_window)}")
        if df_window.empty:
            raise ValueError("[5/8] Still no FIRMS detections in window — check bbox.")
        truth_mask  = geo_grid.dataframe_to_grid_mask(df_window)
        truth_mask  = dilate_truth_mask(truth_mask, config.CELL_SIZE_METERS)
        truth_cells = int(truth_mask.sum())
        print(f"  Ground-truth grid cells burned : {truth_cells}")

    # Use a coarser dt for optimizer evaluations to keep wall-clock time sane.
    # dt=1.0 min/step → 630 steps for 10.5h window (vs 6300 at dt=0.1).
    # Fire spread is smooth enough that 1-min steps give adequate accuracy.
    OPT_DT = 1.0   # minutes per step during optimization
    hindcast_minutes = effective_hours * 60.0
    hindcast_steps   = int(hindcast_minutes / OPT_DT)
    config.dt = OPT_DT   # optimizer evaluations use this dt
    print(f"  Hindcast steps  : {hindcast_steps}  (dt={OPT_DT} min/step, "
          f"{effective_hours:.1f}h window)")
    evals = maxiter * popsize * 2
    cells = land.shape[0] * land.shape[1]
    print(f"  Est. evaluations: ~{maxiter*popsize*2}  |  "
          f"grid={land.shape[0]}x{land.shape[1]}={cells:,} cells")

    # ── Step 5 & 6: Optimisation ──────────────────────────────────────
    print("\n[6/8] Launching Differential Evolution optimiser ...")
    print("  Hidden variables:")
    print("    params[0] = fuel_moisture_offset  in [-0.10,  0.00]  (drier only)")
    print("    params[1] = wind_multiplier        in [ 0.50,  1.50]")
    print(f"  Generations={maxiter}, PopSize={popsize}")
    print(f"  Progress logged every 10 evaluations.\n")

    optimizer = HindcastOptimizer(
        base_landscape  = land,
        geo_grid        = geo_grid,
        truth_mask      = truth_mask,
        hindcast_steps  = hindcast_steps,
        ignition_rc     = ignition_rc,
        eval_callback   = eval_callback,
    )

    bounds = [
        (-0.10,  0.00),   # fuel_moisture_offset: drier only (physically correct)
        ( 0.50,  1.50),   # wind_multiplier: max 1.5× prevents unrealistic spread
    ]

    de_result = differential_evolution(
        optimizer.objective_function,
        bounds,
        maxiter      = maxiter,
        popsize      = popsize,
        mutation     = (0.5, 1.0),   # dithering mutation for better exploration
        recombination= 0.7,
        tol          = 1e-4,
        seed         = 42,           # reproducibility
        disp         = False,        # we print our own progress
        polish       = True,         # final L-BFGS-B polish of the best solution
    )

    best_moisture_offset = float(de_result.x[0])
    best_wind_mult       = float(de_result.x[1])
    best_error           = float(de_result.fun)

    # ── Step 7: Reload at fine resolution, run final simulation ──────────
    print(f"\n[7/8] Reloading terrain at {FINE_SHAPE[0]}x{FINE_SHAPE[1]} "
          f"for final IoU run ...")

    # Build fine landscape
    land_fine = Landscape(config)
    if dem_file and corine_file:
        land_fine.load_real_terrain(dem_file=dem_file, corine_file=corine_file,
                                    target_shape=FINE_SHAPE)
        print(f"  Fine terrain: {land_fine.shape[0]}x{land_fine.shape[1]} cells  "
              f"cell={land_fine.config.CELL_SIZE_METERS:.0f} m")
    else:
        land_fine = land   # synthetic fallback: same grid

    # Re-apply best weather to fine landscape, then snapshot the "base"
    # moisture grid BEFORE adding any DE offset.  We need this snapshot
    # for the ensemble, where each member starts from the same baseline.
    apply_weather_to_landscape(land_fine, weather)
    base_moisture_fine = land_fine.moisture.copy()

    # Now apply the best-fit offset for the deterministic final run
    land_fine.moisture = np.clip(
        base_moisture_fine + best_moisture_offset, 0.01, 0.30
    )
    land_fine.set_wind(weather["wind_speed_ms"] * best_wind_mult,
                       weather["wind_direction"])

    # Rebuild GeoGrid and remap ignition + truth to fine resolution
    fine_rows, fine_cols = land_fine.shape
    geo_grid_fine    = GeoGrid(t_lat_min, t_lat_max, t_lon_min, t_lon_max,
                               fine_rows, fine_cols)
    ignition_rc_fine = geo_grid_fine.latlon_to_rc(ignition_lat, ignition_lon)

    if using_shapefile:
        # Re-rasterise the polygon at fine resolution
        truth_mask_fine = load_copernicus_truth_mask(
            shapefile_path  = str(truth_shapefile),
            landscape_shape = (fine_rows, fine_cols),
            lat_min=t_lat_min, lat_max=t_lat_max,
            lon_min=t_lon_min, lon_max=t_lon_max,
        )
    elif using_fired:
        # Re-rasterise FIRED polygons at fine resolution
        from pipeline.fired_loader import fired_polygon_to_grid_mask as _f2g
        fired_bbox = (t_lat_min, t_lat_max, t_lon_min, t_lon_max)
        truth_mask_fine = _f2g(fired_timeline, fired_bbox, (fine_rows, fine_cols))
        print(f"  [FIRED] Fine truth mask: {int(truth_mask_fine.sum())} cells")
    else:
        truth_mask_fine = geo_grid_fine.dataframe_to_grid_mask(df_window)
        truth_mask_fine = dilate_truth_mask(truth_mask_fine, land_fine.config.CELL_SIZE_METERS)

    # Run final CA at fine resolution
    final_sim = CellularAutomataFire(land_fine, config)
    r0, c0    = ignition_rc_fine
    cell_m_fine = land_fine.config.CELL_SIZE_METERS
    radius    = max(1, int(375 / cell_m_fine / 2))
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < fine_rows and 0 <= cc < fine_cols:
                final_sim.ignite(rr, cc)

    for _ in range(hindcast_steps):
        final_sim.step()

    final_mask = (final_sim.state >= 1) | (final_sim.ignition_fraction >= 0.5)

    # Surface air-computed per-cell wind grids onto the landscape so the GUI
    # and 3D viewer use the physics-corrected (WAF + draft + Poisson) field.
    if hasattr(final_sim, "_wind_u_grid") and final_sim._wind_u_grid is not None:
        land_fine.wind_u = final_sim._wind_u_grid
        land_fine.wind_v = final_sim._wind_v_grid

    intersection = np.logical_and(final_mask, truth_mask_fine).sum()
    union        = np.logical_or( final_mask, truth_mask_fine).sum()
    final_iou    = float(intersection / union) if union > 0 else 0.0

    # For the result dict, expose the fine landscape + fine ignition
    land          = land_fine
    ignition_rc   = ignition_rc_fine
    truth_mask    = truth_mask_fine

    # ── Step 8: Ensemble forecasting (optional) ──────────────────────
    ensemble_result = None
    confidence_map  = None
    if ensemble_size and ensemble_size > 0:
        print(f"\n[8/8] Building {ensemble_size}-member ensemble forecast ...")

        # Pick the top-N members from the optimizer's history.  Request
        # 2× the desired count from the de-dup walker so a higher fraction
        # of unique parameter vectors survives the duplicate filter.
        selected = _select_ensemble_members(
            optimizer._eval_history,
            n_members  = ensemble_size,
            dedup_tol  = 1e-3,
        )
        n_avail = len(selected)
        if n_avail < ensemble_size:
            print(f"  [Ensemble] Only {n_avail} unique parameter sets in history "
                  f"(requested {ensemble_size}). Continuing with {n_avail}.")
        if n_avail == 0:
            print("  [Ensemble] Empty history — skipping.")
        else:
            ensemble_result = run_ensemble(
                landscape_fine   = land_fine,
                geo_grid_fine    = geo_grid_fine,
                weather          = weather,
                base_moisture    = base_moisture_fine,
                ignition_rc_fine = ignition_rc_fine,
                hindcast_steps   = hindcast_steps,
                members          = selected,
                weighting        = ensemble_weighting,
                softmax_tau      = ensemble_softmax_tau,
            )
            confidence_map = ensemble_result["confidence_map"]

            # Restore landscape to best-fit deterministic state — the last
            # ensemble member's parameters are still applied otherwise.
            land_fine.moisture = np.clip(
                base_moisture_fine + best_moisture_offset, 0.01, 0.30
            )
            land_fine.set_wind(weather["wind_speed_ms"] * best_wind_mult,
                               weather["wind_direction"])
    else:
        print("\n[8/8] Ensemble forecasting disabled (--ensemble 0).")

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HINDCAST OPTIMISATION RESULTS")
    print("=" * 60)
    print(f"  Ignition          : {ignition_lat:.5f}°N, {ignition_lon:.5f}°E")
    print(f"  Ignition time     : {ignition_time}")
    print(f"  Hindcast window   : {hindcast_hours} hours ({hindcast_steps} steps)")
    print(f"  Ground-truth cells: {truth_cells}")
    print(f"  Predicted cells   : {int(final_mask.sum())}")
    print()
    print(f"  Optimal moisture offset : {best_moisture_offset:+.4f}")
    print(f"    → Effective midflame moisture ≈ "
          f"{float(optimizer.base_moisture.mean()) + best_moisture_offset:.3f}")
    print(f"  Optimal wind multiplier : {best_wind_mult:.3f}x")
    print(f"    -> Effective wind speed approx "
          f"{weather['wind_speed_ms'] * best_wind_mult:.2f} m/s "
          f"(Open-Meteo measured: {weather['wind_speed_ms']:.1f} m/s @ {weather['wind_direction']:.0f} deg)")
    print()
    print(f"  Composite error   : {best_error:.4f}")
    print(f"  Final IoU         : {final_iou * 100:.1f}%")
    if ensemble_result is not None:
        cm = confidence_map
        print(f"  Ensemble members  : {len(ensemble_result['members'])}")
        print(f"  ≥90% confidence   : {int((cm >= 0.9).sum()):>7d} cells")
        print(f"  50-90% confidence : "
              f"{int(((cm >= 0.5) & (cm < 0.9)).sum()):>7d} cells")
    print("=" * 60)

    result = {
        "best_params": {
            "fuel_moisture_offset": best_moisture_offset,
            "wind_multiplier":      best_wind_mult,
        },
        "best_error":     best_error,
        "iou":            final_iou,
        "predicted_mask": final_mask,
        "truth_mask":     truth_mask,
        "df_firms":       df,          # None when shapefile or FIRED mode
        "weather":        weather,
        "ignition_rc":    ignition_rc,
        "ignition_latlon":(ignition_lat, ignition_lon),
        "ignition_time":  ignition_time,
        "hindcast_steps": hindcast_steps,
        "landscape":      land,
        "geo_grid":       geo_grid_fine if dem_file else geo_grid,
        "terrain_bbox":   (t_lat_min, t_lat_max, t_lon_min, t_lon_max),
        "truth_source":   (
            "copernicus" if using_shapefile else
            "fired"      if using_fired     else
            "firms_viirs"
        ),
        "fired_timeline": fired_timeline,   # GeoDataFrame or None
        "texture_path":   satellite_file or corine_file,  # satellite JPG preferred; fallback CORINE PNG
    }
    if ensemble_result is not None:
        result["confidence_map"] = confidence_map
        result["ensemble"]       = ensemble_result
    return result


def plot_results(result: dict, output_path: str = "hindcast_result.png") -> None:
    """
    Save a 4-panel diagnostic figure from the run_hindcast() result dict.

    Panels
    ------
    1. Elevation (DEM) with ignition point marked
    2. Fuel map (CORINE categories)
    3. Predicted fire perimeter (blue) vs FIRMS ground-truth (red)
    4. Overlap map: TP green, FP blue, FN red
    """
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    land         = result["landscape"]
    truth_mask   = result["truth_mask"]
    pred_mask    = result["predicted_mask"]
    ignition_rc  = result["ignition_rc"]
    ig_lat, ig_lon = result["ignition_latlon"]
    weather      = result["weather"]
    iou          = result["iou"]
    best_params  = result["best_params"]

    rows, cols = land.shape
    r0, c0 = ignition_rc

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Hindcast Results — Ignition {ig_lat:.4f}°N {ig_lon:.4f}°E\n"
        f"ERA5: {weather['wind_speed_ms']:.1f} m/s @ {weather['wind_direction']:.0f}°  "
        f"T={weather['temperature_c']:.1f}°C  RH={weather['relative_humidity']:.0f}%\n"
        f"Optimised: wind×{best_params['wind_multiplier']:.2f}  "
        f"Δmoisture={best_params['fuel_moisture_offset']:+.3f}  "
        f"IoU={iou*100:.1f}%",
        fontsize=11,
    )

    # Panel 1 — DEM
    ax = axes[0, 0]
    im = ax.imshow(land.elevation, cmap="terrain", origin="lower")
    ax.plot(c0, r0, "r*", markersize=14, label="Ignition")
    ax.set_title("Elevation (m)")
    ax.set_xlabel("Col (W→E)"); ax.set_ylabel("Row (S→N)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.legend(fontsize=8)

    # Panel 2 — Fuel map
    ax = axes[0, 1]
    fuel_colors = ["#228B22","#8B4513","#556B2F","#DAA520","#90EE90","#D3D3D3","#4682B4"]
    n_fuels = len(land.fuel_names)
    cmap_fuel = ListedColormap(fuel_colors[:n_fuels])
    im2 = ax.imshow(land.fuel_map, cmap=cmap_fuel, vmin=0, vmax=n_fuels-1, origin="lower")
    ax.plot(c0, r0, "r*", markersize=14)
    ax.set_title("CORINE Fuel Map")
    ax.set_xlabel("Col"); ax.set_ylabel("Row")
    patches = [mpatches.Patch(color=fuel_colors[i], label=land.fuel_names[i])
               for i in range(n_fuels)]
    ax.legend(handles=patches, fontsize=6, loc="lower right")

    # Panel 3 — Predicted vs truth outlines
    ax = axes[1, 0]
    ax.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.5)
    # truth = red, predicted = blue, semi-transparent fills
    truth_overlay = np.zeros((*land.shape, 4), dtype=float)
    truth_overlay[truth_mask] = [1, 0, 0, 0.6]
    pred_overlay = np.zeros((*land.shape, 4), dtype=float)
    pred_overlay[pred_mask]  = [0, 0.4, 1, 0.4]
    ax.imshow(truth_overlay,  origin="lower")
    ax.imshow(pred_overlay,   origin="lower")
    ax.plot(c0, r0, "y*", markersize=14, label="Ignition")
    red_p  = mpatches.Patch(color=[1,0,0,0.6], label=f"FIRMS truth ({truth_mask.sum()} cells)")
    blue_p = mpatches.Patch(color=[0,0.4,1,0.6], label=f"Predicted ({pred_mask.sum()} cells)")
    ax.legend(handles=[red_p, blue_p], fontsize=8)
    ax.set_title("Predicted (blue) vs FIRMS Truth (red)")
    ax.set_xlabel("Col"); ax.set_ylabel("Row")

    # Panel 4 — Overlap breakdown
    ax = axes[1, 1]
    ax.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.5)
    tp = pred_mask & truth_mask
    fp = pred_mask & ~truth_mask
    fn = ~pred_mask & truth_mask
    overlay = np.zeros((*land.shape, 4), dtype=float)
    overlay[tp] = [0, 0.8, 0,   0.8]   # green  = hit
    overlay[fp] = [0, 0.3, 1,   0.6]   # blue   = over-prediction
    overlay[fn] = [1, 0,   0,   0.8]   # red    = miss
    ax.imshow(overlay, origin="lower")
    ax.plot(c0, r0, "y*", markersize=14, label="Ignition")
    ax.legend(handles=[
        mpatches.Patch(color=[0,0.8,0,0.8], label=f"True Pos  {tp.sum()}"),
        mpatches.Patch(color=[0,0.3,1,0.6], label=f"False Pos {fp.sum()}"),
        mpatches.Patch(color=[1,0,  0,0.8], label=f"False Neg {fn.sum()}"),
    ], fontsize=8)
    ax.set_title(f"TP/FP/FN Breakdown  (IoU={iou*100:.1f}%)")
    ax.set_xlabel("Col"); ax.set_ylabel("Row")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Results saved to '{output_path}'")


def plot_ensemble(result: dict,
                  output_path: str = "hindcast_ensemble.png") -> None:
    """
    Save a 4-panel ensemble confidence figure from the run_hindcast() result.

    Requires result to contain "confidence_map" and "ensemble" — i.e. the
    pipeline was run with ensemble_size > 0.

    Panels
    ------
    1. Confidence heatmap (0-100% chance each cell burned)
       with 10 / 50 / 90 % iso-probability contours.
    2. Confidence heatmap overlaid on the DEM, with the ground-truth
       perimeter outlined for direct visual comparison.
    3. Histogram of cell-wise confidence values (excluding never-burned cells).
    4. Parameter scatter: each ensemble member as one dot in
       (Δmoisture, wind_multiplier) space, coloured by error.
    """
    if "confidence_map" not in result or "ensemble" not in result:
        print("[plot_ensemble] result has no ensemble data — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    land           = result["landscape"]
    cm             = result["confidence_map"]      # float32 in [0, 1]
    truth_mask     = result["truth_mask"]
    ignition_rc    = result["ignition_rc"]
    ig_lat, ig_lon = result["ignition_latlon"]
    weather        = result["weather"]
    ensemble       = result["ensemble"]
    members        = ensemble["members"]
    iou            = result["iou"]
    best_params    = result["best_params"]

    rows, cols = cm.shape
    r0, c0     = ignition_rc

    # ── Custom colormap: transparent → yellow → orange → red ────────────
    cmap_conf = LinearSegmentedColormap.from_list(
        "fire_conf",
        [(0.0, (1.0, 1.0, 0.6, 0.0)),   # 0 % — fully transparent
         (0.1, (1.0, 1.0, 0.4, 0.4)),   # 10% — pale yellow
         (0.5, (1.0, 0.6, 0.0, 0.7)),   # 50% — orange
         (0.9, (0.9, 0.1, 0.0, 0.9)),   # 90% — red
         (1.0, (0.5, 0.0, 0.0, 1.0))],  # 100%— dark red
        N=256,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Ensemble Forecast — {len(members)} members "
        f"(weighting={ensemble.get('weights', [0])[0]:.4f}/member avg)\n"
        f"Ignition {ig_lat:.4f}°N {ig_lon:.4f}°E   "
        f"Best IoU={iou*100:.1f}%   "
        f"Best Δm={best_params['fuel_moisture_offset']:+.3f}, "
        f"k_wind={best_params['wind_multiplier']:.2f}",
        fontsize=11,
    )

    # ── Panel 1 — Confidence heatmap with iso-probability contours ──────
    ax = axes[0, 0]
    ax.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.6)
    im = ax.imshow(cm, cmap=cmap_conf, vmin=0, vmax=1, origin="lower")

    # Contours at 10, 50, 90 %
    has_cells = cm.max() > 0
    if has_cells:
        cs = ax.contour(cm, levels=[0.1, 0.5, 0.9],
                        colors=["#fff066", "#ff8800", "#cc0000"],
                        linewidths=[1.2, 1.6, 2.0], origin="lower")
        ax.clabel(cs, fmt={0.1: "10%", 0.5: "50%", 0.9: "90%"},
                  fontsize=8, inline=True)

    ax.plot(c0, r0, "k*", markersize=16, markeredgecolor="white",
            markeredgewidth=1.5, label="Ignition")
    ax.set_title("Burn Probability  (per-cell confidence)")
    ax.set_xlabel("Col (W→E)"); ax.set_ylabel("Row (S→N)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("Probability of burning", fontsize=9)
    cb.set_ticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    cb.set_ticklabels(["0%", "10%", "25%", "50%", "75%", "90%", "100%"])
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 2 — Confidence vs ground-truth perimeter ──────────────────
    ax = axes[0, 1]
    ax.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.6)
    ax.imshow(cm, cmap=cmap_conf, vmin=0, vmax=1, origin="lower")

    # Outline of ground-truth perimeter — use contour at 0.5 on float mask
    if truth_mask.any():
        ax.contour(truth_mask.astype(float), levels=[0.5],
                   colors="cyan", linewidths=2.0, origin="lower")

    ax.plot(c0, r0, "k*", markersize=16, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.set_title("Confidence vs Ground-Truth Perimeter")
    ax.set_xlabel("Col"); ax.set_ylabel("Row")
    ax.legend(handles=[
        mpatches.Patch(color="cyan", label=f"Ground-truth perimeter "
                                           f"({truth_mask.sum()} cells)"),
        mpatches.Patch(color=(0.9, 0.1, 0.0, 0.8), label="≥90% burn probability"),
        mpatches.Patch(color=(1.0, 0.6, 0.0, 0.7), label="50-90% burn probability"),
        mpatches.Patch(color=(1.0, 1.0, 0.4, 0.4), label="10-50% burn probability"),
    ], fontsize=7, loc="lower right")

    # ── Panel 3 — Distribution of non-zero probabilities ────────────────
    ax = axes[1, 0]
    flat = cm.ravel()
    nonzero = flat[flat > 0.0]
    if nonzero.size == 0:
        ax.text(0.5, 0.5, "No burned cells in any ensemble member",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
    else:
        ax.hist(nonzero, bins=20, range=(0, 1),
                color="#ff5500", edgecolor="black", alpha=0.85)
        ax.axvline(0.1, color="#cccc00", linestyle="--", linewidth=1, label="10%")
        ax.axvline(0.5, color="#ff8800", linestyle="--", linewidth=1, label="50%")
        ax.axvline(0.9, color="#cc0000", linestyle="--", linewidth=1, label="90%")
        ax.set_xlabel("Per-cell burn probability")
        ax.set_ylabel("Number of grid cells")
        ax.set_title(f"Confidence Distribution  "
                     f"({nonzero.size:,} cells with any burn prob.)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # ── Panel 4 — Parameter scatter coloured by error ───────────────────
    ax = axes[1, 1]
    pts  = np.array([m["params"] for m in members])
    errs = np.array([m["error"]  for m in members])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=errs, cmap="viridis_r",
                    s=60, edgecolor="black", linewidth=0.5)
    # Mark the best member
    best_i = int(np.argmin(errs))
    ax.plot(pts[best_i, 0], pts[best_i, 1], marker="*",
            markersize=22, markerfacecolor="red",
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"Best (err={errs[best_i]:.4f})")
    ax.set_xlabel("Fuel-moisture offset  Δm")
    ax.set_ylabel("Wind multiplier  k")
    ax.set_title(f"Ensemble Parameter Spread  "
                 f"(N={len(members)})")
    cb2 = fig.colorbar(sc, ax=ax, fraction=0.046)
    cb2.set_label("Composite error", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Ensemble confidence map saved to '{output_path}'")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hindcast Data Assimilation — NASA FIRMS + Rothermel CA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--map-key",        default=_NASA_MAP_KEY,  help="NASA FIRMS map key")
    p.add_argument("--lat-min",        type=float, default=37.8)
    p.add_argument("--lat-max",        type=float, default=38.2)
    p.add_argument("--lon-min",        type=float, default=23.5)
    p.add_argument("--lon-max",        type=float, default=24.0)
    p.add_argument("--date-start",     default="2023-07-18", help="YYYY-MM-DD")
    p.add_argument("--date-end",       default="2023-07-19", help="YYYY-MM-DD")
    p.add_argument("--hindcast-hours", type=float, default=6.0)
    p.add_argument("--maxiter",        type=int,   default=20)
    p.add_argument("--popsize",        type=int,   default=12)
    p.add_argument("--terrain-buffer", type=float, default=0.25,
                   help="Terrain domain half-width in degrees (0.25 = ~28 km around ignition)")
    p.add_argument("--plot",           action="store_true",
                   help="Save a 4-panel results figure as hindcast_result.png")
    p.add_argument("--plot-out",       default="hindcast_result.png",
                   help="Output path for the results figure")
    p.add_argument("--truth-shapefile", default=None,
                   help="Optional Copernicus EMS .shp / .geojson fire perimeter. "
                        "When supplied, replaces NASA FIRMS as ground truth.")
    p.add_argument("--fired-gpkg", default=None,
                   help="Optional FIRED daily GeoPackage (.gpkg). "
                        "Used as ground truth when --truth-shapefile is not provided. "
                        "Download from: https://scholar.colorado.edu/collections/pz50gx05h")
    # ── Ensemble forecasting options ─────────────────────────────────
    p.add_argument("--ensemble",       type=int,   default=0,
                   metavar="N",
                   help="Run an ensemble forecast: take the top-N unique "
                        "parameter sets from the DE evaluation history, "
                        "re-run each at fine resolution, and stack their "
                        "burn masks into a probabilistic confidence map. "
                        "0 (default) disables the feature. Typical: 30-100.")
    p.add_argument("--ensemble-weighting", choices=["uniform", "softmax"],
                   default="uniform",
                   help="How to weight ensemble members when stacking masks. "
                        "uniform = all members equal (most honest spread). "
                        "softmax = better-scoring members count more.")
    p.add_argument("--ensemble-tau",   type=float, default=0.05,
                   help="Softmax temperature for ensemble weighting "
                        "(smaller = more peaked around best member).")
    p.add_argument("--ensemble-plot-out", default="hindcast_ensemble.png",
                   help="Output path for the ensemble confidence figure.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_hindcast(
        map_key        = args.map_key,
        lat_min        = args.lat_min,
        lat_max        = args.lat_max,
        lon_min        = args.lon_min,
        lon_max        = args.lon_max,
        date_start     = args.date_start,
        date_end       = args.date_end,
        hindcast_hours = args.hindcast_hours,
        maxiter        = args.maxiter,
        popsize        = args.popsize,
        terrain_buffer = args.terrain_buffer,
        truth_shapefile= args.truth_shapefile,
        fired_gpkg     = args.fired_gpkg,
        ensemble_size       = args.ensemble,
        ensemble_weighting  = args.ensemble_weighting,
        ensemble_softmax_tau= args.ensemble_tau,
    )
    if args.plot:
        plot_results(result, output_path=args.plot_out)
    if args.ensemble and "confidence_map" in result:
        plot_ensemble(result, output_path=args.ensemble_plot_out)
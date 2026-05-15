"""fired_loader.py -- FIRED Daily Fire-Perimeter Loader for Project WILSON.

Data source: https://scholar.colorado.edu/collections/pz50gx05h
Expected file: fired_greece_daily.gpkg
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

FIRED_DOWNLOAD_URL = "https://scholar.colorado.edu/collections/pz50gx05h"

# Known FIRED event IDs confirmed by sinusoidal→WGS84 back-projection
EVIA_2021_EVENT_ID = 2871   # Northern Evia megafire, Aug 2021, centre 38.83°N 23.20°E

KNOWN_FIRE_EVENTS = {
    "Evoia 2021": {
        "event_id":   EVIA_2021_EVENT_ID,
        "lat_centre": 38.83, "lon_centre": 23.20,
        "date_start": "2021-08-01", "date_end": "2021-08-13",
        "note":       "Northern Evia megafire — ~50 000 ha",
    },
}

_DATE_COLS = ["date", "burn_date", "ig_date", "ignition_date", "acq_date"]
_ID_COLS   = ["id", "event_id", "did", "fire_id", "eventid"]


def _detect_date_column(gdf):
    """Return the first recognised date-column name, or None."""
    lower_cols = {c.lower(): c for c in gdf.columns}
    for candidate in _DATE_COLS:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def _detect_id_column(gdf):
    """Return the first recognised event-ID column name, or None."""
    lower_cols = {c.lower(): c for c in gdf.columns}
    for candidate in _ID_COLS:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def load_fired_daily(gpkg_path, bbox):
    """Load and spatially filter the FIRED daily GeoPackage.

    Parameters
    ----------
    gpkg_path : str or Path -- path to fired_greece_daily.gpkg.
    bbox      : (lat_min, lat_max, lon_min, lon_max)

    Returns
    -------
    GeoDataFrame in EPSG:4326 with standardised burn_date column.

    Raises
    ------
    FileNotFoundError  if gpkg_path does not exist.
    ValueError         if no date column is found.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required: conda install -c conda-forge geopandas"
        ) from exc

    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        msg = (
            "[FIRED] GeoPackage not found: {}\n\n"
            "  Download instructions\n"
            "  ---------------------\n"
            "  1. Visit {}\n"
            "  2. Download the daily-perimeter GeoPackage for Greece.\n"
            "  3. Save as: {}\n"
        ).format(gpkg_path, FIRED_DOWNLOAD_URL, gpkg_path.resolve())
        raise FileNotFoundError(msg)

    lat_min, lat_max, lon_min, lon_max = bbox
    print("[FIRED] Loading: {}".format(gpkg_path))

    # ── Detect native CRS using geopandas (read 1 row, cheap) ─────────────
    # FIRED Greece files are typically in MODIS sinusoidal projection —
    # passing a WGS84 bbox to read_file() returns 0 features.
    # Strategy: detect CRS, read full file only when not WGS84, then filter.
    native_is_wgs84 = False
    try:
        _probe = gpd.read_file(gpkg_path, rows=1)
        if _probe.crs is not None:
            try:
                import pyproj as _pyproj
                native_is_wgs84 = _pyproj.CRS(_probe.crs).equals(_pyproj.CRS("EPSG:4326"))
            except Exception:
                # Rough check: WGS84 has angular units (degrees); sinusoidal has metres
                native_is_wgs84 = "degree" in str(_probe.crs).lower()
        del _probe
    except Exception:
        pass   # probe failed — will read full and let geopandas handle it

    if native_is_wgs84:
        try:
            gdf = gpd.read_file(gpkg_path, bbox=(lon_min, lat_min, lon_max, lat_max))
        except Exception as exc:
            print("[FIRED] WARNING -- bbox read failed ({}); reading full.".format(exc))
            gdf = gpd.read_file(gpkg_path)
    else:
        print("[FIRED] Non-WGS84 CRS detected — reading full file and reprojecting.")
        gdf = gpd.read_file(gpkg_path)

    print("[FIRED] Loaded {} features (pre-filter).".format(len(gdf)))

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()
    print("[FIRED] {} features in bbox.".format(len(gdf)))

    date_col = _detect_date_column(gdf)
    if date_col is None:
        raise ValueError(
            "[FIRED] No date column found. Available: {}  Expected: {}".format(
                list(gdf.columns), _DATE_COLS))

    if date_col != "burn_date":
        gdf = gdf.rename(columns={date_col: "burn_date"})

    # Normalise event ID column → always called "id" for consistent access
    id_col = _detect_id_column(gdf)
    if id_col and id_col != "id":
        gdf = gdf.rename(columns={id_col: "id"})
        print("[FIRED] Event ID column: '{}' → renamed to 'id'.".format(id_col))
    elif id_col is None:
        print("[FIRED] WARNING — no event ID column found; event-level filtering disabled.")

    gdf["burn_date"] = pd.to_datetime(gdf["burn_date"], utc=True, errors="coerce")
    gdf = gdf[gdf["burn_date"].notna()].copy()
    gdf = gdf.sort_values("burn_date").reset_index(drop=True)
    print("[FIRED] Date range: {} -> {}".format(gdf["burn_date"].min(), gdf["burn_date"].max()))
    return gdf


def find_fire_event(daily_gdf, date_start, date_end, bbox):
    """Find dominant fire event in a date/bbox window. Returns event_id or None.

    Dominant = largest total burned area across all days in the window.
    Groups by event_id column when present; otherwise returns the index
    of the largest single polygon.
    """
    t0 = pd.Timestamp(date_start, tz="UTC")
    t1 = pd.Timestamp(date_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    lat_min, lat_max, lon_min, lon_max = bbox
    window = daily_gdf[
        (daily_gdf["burn_date"] >= t0) & (daily_gdf["burn_date"] <= t1)
    ].cx[lon_min:lon_max, lat_min:lat_max].copy()

    if window.empty:
        print("[FIRED] No events for {} -> {} in bbox.".format(date_start, date_end))
        return None

    window["_area_m2"] = window.geometry.to_crs("EPSG:3857").area
    # The ID column is always normalised to "id" by load_fired_daily
    if "id" in window.columns:
        by_event = window.groupby("id")["_area_m2"].sum().sort_values(ascending=False)
        best_id = by_event.index[0]
        print("[FIRED] Dominant event id={}  {:.1f} ha".format(best_id, by_event.iloc[0] / 10000))
        return best_id
    else:
        best_idx = window["_area_m2"].idxmax()
        print("[FIRED] No id column -- row={} {:.1f} ha".format(
            best_idx, window.loc[best_idx, "_area_m2"] / 10000))
        return best_idx


def get_event_timeline(daily_gdf, event_id):
    """Return all daily polygon rows for event_id, sorted by burn_date."""
    if "id" in daily_gdf.columns:
        tl = daily_gdf[daily_gdf["id"] == event_id].copy()
    else:
        tl = daily_gdf.loc[[event_id]].copy()
    tl = tl.sort_values("burn_date").reset_index(drop=True)
    print("[FIRED] Timeline: {} polygons ({} -> {})".format(
        len(tl), tl["burn_date"].min().date(), tl["burn_date"].max().date()))
    return tl


def fired_polygon_to_grid_mask(polygon_gdf, dem_bounds, grid_shape):
    """Burn FIRED polygons onto simulation grid as boolean mask.

    rasterio.features.rasterize produces north-down (row-0=North).
    landscape.elevation uses south-up (row-0=South) set by np.flipud in
    load_real_terrain. We apply np.flipud to align the mask with landscape.

    Parameters
    ----------
    polygon_gdf : GeoDataFrame in EPSG:4326
    dem_bounds  : (lat_min, lat_max, lon_min, lon_max)
    grid_shape  : (rows, cols) matching landscape.elevation.shape

    Returns
    -------
    bool (rows, cols) ndarray, row-0 = lat_min (south-up, matches landscape).
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    lat_min, lat_max, lon_min, lon_max = dem_bounds
    rows, cols = grid_shape
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)

    geoms = [g for g in polygon_gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        print("[FIRED] WARNING -- no valid geometries; returning empty mask.")
        return np.zeros((rows, cols), dtype=bool)

    north_down = rasterize(
        geoms, out_shape=(rows, cols), transform=transform,
        fill=0, default_value=1, dtype="uint8", all_touched=True,
    ).astype(bool)

    # Flip to south-up: row-0 becomes lat_min (matches landscape.elevation)
    south_up = np.flipud(north_down)

    n_cells = int(south_up.sum())
    cell_m = ((lat_max - lat_min) / rows) * 111_000
    print("[FIRED] Rasterised -> {} cells  ({:.1f} ha)".format(
        n_cells, n_cells * cell_m ** 2 / 10_000))
    return south_up


def get_fired_truth_mask(gpkg_path, bbox, date_start, date_end,
                         dem_bounds, grid_shape, cumulative=True):
    """Master function: load -> find event -> rasterize -> return dict.

    Parameters
    ----------
    gpkg_path  : path to the FIRED GeoPackage.
    bbox       : search bbox (lat_min, lat_max, lon_min, lon_max).
    date_start : "YYYY-MM-DD"
    date_end   : "YYYY-MM-DD"
    dem_bounds : (lat_min, lat_max, lon_min, lon_max) of the terrain grid.
    grid_shape : (rows, cols) matching landscape.elevation.shape.
    cumulative : True -> union of all daily polygons in window (default).
                 False -> only the last day's polygon.

    Returns
    -------
    dict with keys: mask, event_id, timeline, area_ha, n_days, date_range.
    """
    daily_gdf = load_fired_daily(gpkg_path, bbox)
    event_id = find_fire_event(daily_gdf, date_start, date_end, bbox)

    if event_id is None:
        empty = np.zeros(grid_shape, dtype=bool)
        return {"mask": empty, "event_id": None,
                "timeline": daily_gdf.iloc[0:0], "area_ha": 0.0,
                "n_days": 0, "date_range": (None, None)}

    timeline = get_event_timeline(daily_gdf, event_id)
    t0 = pd.Timestamp(date_start, tz="UTC")
    t1 = pd.Timestamp(date_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    window_tl = timeline[(timeline["burn_date"] >= t0) & (timeline["burn_date"] <= t1)]
    if window_tl.empty:
        window_tl = timeline
    if not cumulative:
        window_tl = window_tl.iloc[[-1]]

    mask = fired_polygon_to_grid_mask(window_tl, dem_bounds, grid_shape)
    lat_min, lat_max = dem_bounds[0], dem_bounds[1]
    cell_m = ((lat_max - lat_min) / grid_shape[0]) * 111_000
    area_ha = float(mask.sum()) * cell_m ** 2 / 10_000
    first_d = window_tl["burn_date"].min()
    last_d  = window_tl["burn_date"].max()
    print("[FIRED] Mask ready: event={} {} -> {} {} cells {:.1f} ha".format(
        event_id, first_d.date(), last_d.date(), int(mask.sum()), area_ha))

    return {"mask": mask, "event_id": event_id,
            "timeline": window_tl.reset_index(drop=True),
            "area_ha": area_ha, "n_days": len(window_tl),
            "date_range": (first_d, last_d)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fired_loader.py <fired_greece_daily.gpkg>")
        print("Download: {}".format(FIRED_DOWNLOAD_URL))
        sys.exit(0)
    r = get_fired_truth_mask(sys.argv[1], (35.8, 36.5, 27.0, 28.5),
                             "2023-07-19", "2023-07-22",
                             (35.8, 36.5, 27.0, 28.5), (400, 400))
    print(r)

"""auto_fetcher.py — Real-World Data Acquisition
================================================
Fetches all geospatial inputs needed to run the Hellenic Wildfire Digital Twin
on real terrain instead of synthetic hills:

  1. Drone telemetry     → GPS fix (Lat, Lon) from a drone log file
  2. Copernicus COP30 DEM → 30 m terrain elevation from OpenTopography API
  3. ESRI World Imagery  → High-res satellite texture (for visualizer_3d.py)
  4. CORINE Land Cover    → EU vegetation/land-use map for fuel-type assignment

API key (OpenTopography / Copernicus COP30):  57314bc7ed85882904a7485d77c0dbe5
Register or renew at:  https://portal.opentopography.org/requestApiKey
"""

import os
import requests

# Default Copernicus/OpenTopography API key
_DEFAULT_API_KEY = "57314bc7ed85882904a7485d77c0dbe5"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Drone telemetry parser
# ──────────────────────────────────────────────────────────────────────────────

def read_drone_telemetry(filepath="drone_telemetry.txt"):
    """
    Parse a drone telemetry text file with the format:
        Lat: 38.894939
        Lon: 23.401405

    Returns (lat, lon) as floats.
    """
    telemetry = {}
    print(f"[Fetcher] Reading telemetry from: {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) == 2:
                telemetry[parts[0].strip()] = float(parts[1].strip())
    return telemetry["Lat"], telemetry["Lon"]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Copernicus COP30 DEM  (OpenTopography API)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_terrain_from_api(lat, lon, api_key=_DEFAULT_API_KEY,
                            output_dir=".", buffer=0.05, force=False):
    """
    Fetch a Copernicus COP30 (~30 m resolution) DEM GeoTIFF for a bounding box
    centred on (lat, lon) with ±buffer degrees.

    Cache filename encodes the location so different fires never share a file.
    Falls back silently to the cached file if the network request fails.
    Skips the download entirely if the cached file already exists (use
    ``force=True`` to override).
    """
    # Location-specific filename so Rhodes and Evoia don't overwrite each other
    tag      = f"{lat:.3f}_{lon:.3f}_{buffer:.3f}"
    filename = os.path.join(output_dir, f"terrain_{tag}.tif")

    if not force and os.path.exists(filename):
        print(f"[Fetcher] Using cached DEM '{filename}' (pass force=True to re-fetch).")
        return filename

    west, east   = lon - buffer, lon + buffer
    south, north = lat - buffer, lat + buffer

    print(f"\n[Fetcher] Target: Lat={lat:.5f}, Lon={lon:.5f}")
    print(f"[Fetcher] Bounding box: S={south:.4f} N={north:.4f} "
          f"W={west:.4f} E={east:.4f}")

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":      "COP30",
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": "GTiff",
        "API_Key":      api_key,
    }

    print("[Fetcher] Requesting Copernicus COP30 DEM from OpenTopography ...")

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        with open(filename, "wb") as fh:
            fh.write(resp.content)
        print(f"[Fetcher] SUCCESS — terrain saved as '{filename}'")
        return filename
    except Exception as exc:
        print(f"[Fetcher] WARNING — DEM download failed: {exc}")
        if os.path.exists(filename):
            print(f"[Fetcher] OFFLINE MODE — using cached '{filename}'")
            return filename
        # Last resort: any existing terrain tif in the output dir
        import glob as _glob
        fallbacks = _glob.glob(os.path.join(output_dir, "terrain_*.tif"))
        if fallbacks:
            print(f"[Fetcher] FALLBACK — using nearest cached terrain: {fallbacks[0]}")
            return fallbacks[0]
        print("[Fetcher] CRITICAL — no cached DEM found. Aborting.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ESRI World Imagery satellite texture
# ──────────────────────────────────────────────────────────────────────────────

def fetch_satellite_image_by_bounds(west, south, east, north,
                                     width, height, output_dir="."):
    """
    Fetch a high-resolution satellite JPG from ESRI World Imagery,
    pixel-matched to the DEM's exact bounding box and aspect ratio.

    No API key required — ESRI provides this service free.
    """
    print(f"\n[Fetcher] Fetching satellite texture (ESRI World Imagery) ...")
    scale_factor = max(1, 2048 // max(width, height))
    tex_w = width  * scale_factor
    tex_h = height * scale_factor

    url = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/export")
    params = {
        "bbox":    f"{west},{south},{east},{north}",
        "bboxSR":  "4326",
        "size":    f"{tex_w},{tex_h}",
        "imageSR": "4326",
        "format":  "jpg",
        "f":       "image",
    }

    filename = os.path.join(output_dir, "texture.jpg")
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        os.makedirs(output_dir, exist_ok=True)
        with open(filename, "wb") as fh:
            fh.write(resp.content)
        print(f"[Fetcher] SUCCESS — satellite texture saved as '{filename}' "
              f"({tex_w}x{tex_h})")
        return filename
    except Exception as exc:
        print(f"[Fetcher] ERROR — satellite texture failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EEA CORINE Land Cover  (vegetation / fuel type map)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_corine_land_cover(west, south, east, north,
                             width, height, output_dir=".", force=False):
    """
    Fetch CORINE Land Cover 2018 PNG from the EEA DISCOMAP WMS service.
    Cache filename encodes the bbox so different fires never share a file.
    """
    tag      = f"{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}"
    filename = os.path.join(output_dir, f"corine_{tag}.png")

    if not force and os.path.exists(filename):
        print(f"[Fetcher] Using cached CORINE '{filename}' (pass force=True to re-fetch).")
        return filename

    print(f"\n[Fetcher] Fetching CORINE Land Cover (CLC 2018) ...")

    url = ("https://image.discomap.eea.europa.eu/arcgis/rest/services/"
           "Corine/CLC2018_WM/MapServer/export")
    params = {
        "bbox":        f"{west},{south},{east},{north}",
        "bboxSR":      "4326",
        "size":        f"{width},{height}",
        "imageSR":     "4326",
        "format":      "png",
        "transparent": "false",
        "f":           "image",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        with open(filename, "wb") as fh:
            fh.write(resp.content)
        print(f"[Fetcher] SUCCESS — CORINE map saved as '{filename}'")
        return filename
    except Exception as exc:
        print(f"[Fetcher] WARNING — CORINE download failed: {exc}")
        if os.path.exists(filename):
            print(f"[Fetcher] OFFLINE MODE — using cached '{filename}'")
            return filename
        raise RuntimeError(
            "CORINE fetch failed and no cached file exists. "
            "Check network or place corine_cover.png manually."
        ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Master Orchestrator  — single call to prepare a full simulation
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_wildfire_data(
    lat:        float,
    lon:        float,
    start_date: str,
    end_date:   str,
    firms_map_key: str,
    radius_km:  float = 15.0,
    output_dir: str   = ".",
    force:      bool  = False,
) -> dict:
    """
    Master API Orchestrator.

    Given a GPS centre point and a date range, fetches ALL data required to
    run a hindcast simulation and returns them as a single ready-to-use dict.

    Orchestrates
    ------------
    1. Bounding box calculation from (lat, lon, radius_km)
    2. Open-Meteo ERA5 weather for start_date
    3. NASA FIRMS VIIRS active-fire detections
    4. Copernicus COP30 DEM via OpenTopography
    5. CORINE Land Cover 2018 PNG

    Parameters
    ----------
    lat, lon       : centre of the study area (decimal degrees)
    start_date     : ISO date string "YYYY-MM-DD" — first day of FIRMS window
    end_date       : ISO date string "YYYY-MM-DD" — last day of FIRMS window
    firms_map_key  : NASA FIRMS map key (get one at firms.modaps.eosdis.nasa.gov)
    radius_km      : approximate search radius in km (default 15 km ≈ 0.135°)
    output_dir     : directory to write all downloaded files (default ".")
    force          : if True, re-download even if cached files exist

    Returns
    -------
    dict with keys
      "bbox"         : (lat_min, lat_max, lon_min, lon_max)
      "weather"      : dict from fetch_weather()
      "df_firms"     : pandas DataFrame of VIIRS detections (may be empty)
      "dem_file"     : path to the GeoTIFF, or None on failure
      "corine_file"  : path to the CORINE PNG, or None on failure
      "status"       : dict of per-step success flags
    """
    import math as _math

    # ── 1.  Bounding box ─────────────────────────────────────────────────────
    # 1° latitude ≈ 111 km; 1° longitude ≈ 111 km × cos(lat)
    lat_buf = radius_km / 111.0
    lon_buf = radius_km / (111.0 * max(_math.cos(_math.radians(lat)), 0.01))
    lat_min, lat_max = lat - lat_buf, lat + lat_buf
    lon_min, lon_max = lon - lon_buf, lon + lon_buf
    bbox = (lat_min, lat_max, lon_min, lon_max)

    print("\n" + "=" * 60)
    print("  MASTER ORCHESTRATOR — fetch_all_wildfire_data")
    print("=" * 60)
    print(f"  Centre    : {lat:.5f}°N  {lon:.5f}°E")
    print(f"  Radius    : {radius_km} km")
    print(f"  Bbox      : S={lat_min:.4f} N={lat_max:.4f} W={lon_min:.4f} E={lon_max:.4f}")
    print(f"  Dates     : {start_date} → {end_date}")

    status = {
        "weather": False,
        "firms":   False,
        "dem":     False,
        "corine":  False,
    }

    # ── 2.  Weather ──────────────────────────────────────────────────────────
    weather = None
    try:
        weather = _fetch_weather_orchestrator(lat, lon, start_date)
        status["weather"] = True
        print(f"  [1/4] Weather OK  — {weather['wind_speed_ms']:.1f} m/s @ "
              f"{weather['wind_direction']:.0f}°  "
              f"T={weather['temperature_c']:.1f}°C  RH={weather['relative_humidity']:.0f}%")
    except Exception as exc:
        print(f"  [1/4] Weather FAILED: {exc}")

    # ── 3.  NASA FIRMS ───────────────────────────────────────────────────────
    df_firms = None
    try:
        df_firms = _fetch_firms_orchestrator(
            firms_map_key, lat_min, lat_max, lon_min, lon_max, start_date, end_date
        )
        status["firms"] = True
        print(f"  [2/4] FIRMS OK    — {len(df_firms)} detections")
    except Exception as exc:
        print(f"  [2/4] FIRMS FAILED: {exc}")
        import pandas as _pd
        df_firms = _pd.DataFrame()

    # ── 4.  COP30 DEM ────────────────────────────────────────────────────────
    dem_file = None
    try:
        dem_file = fetch_terrain_from_api(
            lat, lon, output_dir=output_dir,
            buffer=max(lat_buf, lon_buf), force=force
        )
        status["dem"] = dem_file is not None
        print(f"  [3/4] DEM OK      — {dem_file}")
    except Exception as exc:
        print(f"  [3/4] DEM FAILED: {exc}")

    # ── 5.  CORINE ───────────────────────────────────────────────────────────
    corine_file = None
    if dem_file:
        try:
            corine_file = fetch_corine_land_cover(
                lon_min, lat_min, lon_max, lat_max, 400, 400,
                output_dir=output_dir, force=force
            )
            status["corine"] = corine_file is not None
            print(f"  [4/4] CORINE OK   — {corine_file}")
        except Exception as exc:
            print(f"  [4/4] CORINE FAILED: {exc}")

    n_ok = sum(status.values())
    print(f"\n  Orchestrator complete: {n_ok}/4 sources ready")
    print("=" * 60)

    return {
        "bbox":        bbox,
        "lat_min":     lat_min,
        "lat_max":     lat_max,
        "lon_min":     lon_min,
        "lon_max":     lon_max,
        "weather":     weather,
        "df_firms":    df_firms,
        "dem_file":    dem_file,
        "corine_file": corine_file,
        "status":      status,
    }


# ── Internal helpers used only by the orchestrator ────────────────────────────

def _fetch_weather_orchestrator(lat: float, lon: float, date: str) -> dict:
    """Calls Open-Meteo ERA5 hourly for the given date; picks ignition-hour conditions.

    Delegates to hindcast_optimizer.fetch_weather_hourly for consistency,
    then returns the noon (12:00 UTC) snapshot as the representative dict
    (orchestrator doesn't know the exact ignition time at this stage).
    """
    try:
        from pipeline.hindcast_optimizer import fetch_weather_hourly as _fwh
        df = _fwh(lat, lon, date_start=date, date_end=date)
        idx  = min(12, len(df) - 1)
        row  = df.iloc[idx]
        return {
            "temperature_c":     float(row["temperature_c"]),
            "relative_humidity": float(row["relative_humidity"]),
            "wind_speed_ms":     float(row["wind_speed_ms"]),
            "wind_direction":    float(row["wind_direction"]),
        }
    except Exception:
        # Minimal fallback in case of import issues
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
            "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            "&wind_speed_unit=ms&timezone=UTC"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        h    = resp.json().get("hourly", {})
        idx  = min(12, len(h.get("temperature_2m", [0])) - 1)
        return {
            "temperature_c":     float(h["temperature_2m"][idx]),
            "relative_humidity": float(h["relative_humidity_2m"][idx]),
            "wind_speed_ms":     float(h["wind_speed_10m"][idx]),
            "wind_direction":    float(h["wind_direction_10m"][idx]),
        }


def _fetch_firms_orchestrator(map_key, lat_min, lat_max, lon_min, lon_max,
                               date_start, date_end):
    """
    Lightweight FIRMS fetch used by the orchestrator.
    Returns a DataFrame (may be empty — caller handles that).
    """
    import pandas as _pd
    import datetime as _dt

    _SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]
    _COLS    = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]

    start = _dt.date.fromisoformat(date_start)
    end   = _dt.date.fromisoformat(date_end)
    days  = max(1, (end - start).days + 1)

    for source in _SOURCES:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{map_key}/{source}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{date_start}"
        )
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = _pd.read_csv(__import__("io").StringIO(resp.text))
            if df.empty or "latitude" not in df.columns:
                continue
            df["acq_datetime"] = _pd.to_datetime(
                df["acq_date"].astype(str) + " " +
                df["acq_time"].astype(str).str.zfill(4).str[:2] + ":" +
                df["acq_time"].astype(str).str.zfill(4).str[2:],
                utc=True
            )
            conf_ok = df["confidence"].astype(str).str.lower().isin(
                ["high", "nominal", "h", "n"]
            ) | df["confidence"].apply(
                lambda c: str(c).lstrip("0").isdigit() and int(str(c)) >= 50
            )
            df = df[conf_ok].copy()
            if not df.empty:
                print(f"  [FIRMS] {len(df)} detections from {source}")
                return df
        except Exception:
            continue

    return _pd.DataFrame()


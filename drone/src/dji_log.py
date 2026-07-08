"""dji_log.py — DJI GO 4 flight-record telemetry for the fire pipeline
=====================================================================
Loads a telemetry CSV exported from a DJI GO 4 ``.txt`` flight record
(via https://github.com/lvauvillier/dji-log-parser or a PhantomHelp /
Airdata export) and answers "where was the drone / camera at time t?"
so cached video frames can be geolocated by geometry.py.

.TXT AUTO-DECODE
----------------
DJI GO 4 ``.txt`` flight records are proprietary encrypted binaries.
If you pass a ``.txt`` path directly, this module will attempt to
decode it automatically in this order:
  1. Run the ``dji-log-parser`` npm CLI (``npx -y dji-log-parser``).
  2. Look for a companion CSV with the same base name
     (e.g. ``DJIFlightRecord_2026-07-08_[15-30-54].csv``).
  3. Look for any ``ExportCSV_*.csv`` file in the same directory.
  4. Look for any ``flight_*.csv`` / ``*.csv`` in the same directory.
  5. Raise an informative error if nothing works.

Column names differ between exporters, so headers are matched fuzzily
(lowercased, non-alphanumerics stripped) against known candidates.

Telemetry conventions produced here (see geometry.py):
  • The Spark gimbal stabilises the camera independently of the body,
    so we feed geometry.py the GIMBAL attitude, not the aircraft's.
  • DJI gimbal pitch: 0° = horizontal, -90° = straight down.
    geometry.py models a nadir body-fixed camera where positive
    telemetry.pitch tilts the ray backward, hence
        telemetry.pitch = -(90 + gimbal_pitch)
    (gimbal -90° → 0 = nadir; gimbal 0° → -90 = forward horizon).
  • The Spark gimbal has no yaw axis: camera heading = aircraft yaw.
  • ``height`` is barometric height above the take-off point, used as
    alt_agl (valid over roughly flat terrain around the launch site).
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from geometry import DroneTelemetry


# ── TXT → CSV auto-decoder ────────────────────────────────────────────────────

def _is_dji_binary(path: str) -> bool:
    """Return True if the file is a DJI GO 4 proprietary binary (not a text CSV)."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
        # DJI binaries are not valid UTF-8 CSV — they start with non-printable bytes
        # and the first 8 bytes encode the total record count as a little-endian int.
        return not all(0x09 <= b <= 0x7E or b in (0x0A, 0x0D) for b in header[:8])
    except OSError:
        return False


def _try_npm_decode(txt_path: str, out_csv: str) -> bool:
    """Try decoding the .txt with the dji-log-parser npm CLI.
    Returns True on success."""
    npx = shutil.which("npx")
    if not npx:
        return False
    try:
        result = subprocess.run(
            [npx, "-y", "dji-log-parser", txt_path, "--output", out_csv],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and os.path.isfile(out_csv)
    except Exception:
        return False


# Required telemetry fields that every valid DJI CSV must supply
_REQUIRED_FIELDS = ("lat", "lon", "height", "gimbal_pitch")


def _csv_has_required_columns(csv_path: str) -> bool:
    """Return True if the CSV has all required telemetry columns AND contains
    at least one row with a valid (non-zero) GPS fix.

    Uses the same fuzzy normalisation as _find_columns() so it works
    regardless of the exporter (OSD.latitude vs latitude vs lat, etc.).
    """
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return False
            # Flatten any space/tab-separated headers into individual tokens
            flat_headers: list[str] = []
            for h in reader.fieldnames:
                flat_headers.extend(h.split())
            found = _find_columns(flat_headers)
            missing = [f for f in _REQUIRED_FIELDS if f not in found]
            if missing:
                print(f"[DJILog]   skip {Path(csv_path).name!r}: missing cols {missing}")
                return False

            # Check that at least one row has a non-zero GPS coordinate.
            # We read up to 200 rows to avoid loading huge files.
            lat_col = found["lat"]
            lon_col = found["lon"]
            has_gps = False
            for i, row in enumerate(reader):
                if i > 200:
                    break
                try:
                    lat = float(row.get(lat_col, "0") or "0")
                    lon = float(row.get(lon_col, "0") or "0")
                    if abs(lat) > 1e-4 and abs(lon) > 1e-4:
                        has_gps = True
                        break
                except (TypeError, ValueError):
                    continue

            if not has_gps:
                print(f"[DJILog]   skip {Path(csv_path).name!r}: all GPS rows are 0,0 — no fix")
                return False
        return True
    except Exception as exc:
        print(f"[DJILog]   skip {Path(csv_path).name!r}: {exc}")
        return False



def _find_companion_csv(txt_path: str) -> Optional[str]:
    """Search for a usable (fully-featured) exported CSV near the .txt file.

    Each candidate is validated for required telemetry columns before being
    accepted.  Priority order:
      1. Same stem with .csv extension  (e.g. DJIFlightRecord_X.csv)
      2. flight_*.csv in the same folder  ← full DJI Assistant export
      3. ExportCSV_*.csv in the same folder  ← may be a minimal export
      4. Any other *.csv (alphabetically last = newest)
    """
    p = Path(txt_path)
    folder = p.parent

    def _try(candidates: list[Path], label: str) -> Optional[str]:
        for c in candidates:
            print(f"[DJILog] Checking {label}: {c.name}")
            if _csv_has_required_columns(str(c)):
                print(f"[DJILog] ✓ Accepted {label}: {c}")
                return str(c)
        return None

    # 1. Same stem
    companion = folder / (p.stem + ".csv")
    if companion.is_file():
        result = _try([companion], "companion")
        if result:
            return result

    # 2. flight_*.csv — full DJI Assistant / dji-log-parser exports
    flight_csvs = sorted(folder.glob("flight_*.csv"))
    if flight_csvs:
        result = _try(flight_csvs[::-1], "flight CSV")  # newest first
        if result:
            return result

    # 3. ExportCSV_*.csv — DJI GO 4 app share-sheet (often minimal)
    export_csvs = sorted(folder.glob("ExportCSV_*.csv"))
    if export_csvs:
        result = _try(export_csvs[::-1], "ExportCSV")
        if result:
            return result

    # 4. Any CSV at all (newest first)
    any_csvs = sorted(folder.glob("*.csv"))[::-1]
    result = _try(any_csvs, "fallback CSV")
    return result


def resolve_log_path(path: str) -> str:
    """If *path* is a DJI GO 4 binary .txt, decode it to a CSV and return the
    CSV path.  Otherwise return *path* unchanged.

    Call this before constructing DJILogBook when the user may pass a .txt.
    """
    if not path.lower().endswith(".txt"):
        return path  # already a CSV or other text format

    if not _is_dji_binary(path):
        return path  # it's a plain-text (uncommon legacy format)

    print(f"[DJILog] Detected DJI GO 4 binary TXT: {path}")
    print("[DJILog] Attempting auto-decode ...")

    # Strategy 1: npm dji-log-parser
    tmp_csv = os.path.join(tempfile.gettempdir(), Path(path).stem + "_decoded.csv")
    if _try_npm_decode(path, tmp_csv):
        print(f"[DJILog] ✓ Decoded via dji-log-parser → {tmp_csv}")
        return tmp_csv

    # Strategy 2-4: companion CSV in the same folder
    companion = _find_companion_csv(path)
    if companion:
        print(f"[DJILog] ✓ Using companion CSV: {companion}")
        return companion

    raise FileNotFoundError(
        f"Cannot decode DJI binary TXT '{path}'.\n"
        "Please do ONE of the following:\n"
        "  a) Install Node.js and run:  npx -y dji-log-parser\n"
        "  b) Place the exported CSV (same stem + .csv) next to the .txt file\n"
        "  c) Use DJI Assistant / AirData to export a CSV and pass it directly"
    )

# ── Fuzzy header candidates (normalised: lowercase, alphanumeric only) ────────
_COLUMN_CANDIDATES: Dict[str, Sequence[str]] = {
    "time_ms":      ("timemillisecond", "timemilliseconds", "flytimems"),
    "time_s":       ("osdflytime", "flytime", "timesecond", "flytimesec"),
    "datetime":     ("customdatetime", "customupdatetime", "datetimeutc",
                     "customdatelocal", "datetime"),
    "lat":          ("osdlatitude", "latitude", "lat"),
    "lon":          ("osdlongitude", "longitude", "lon", "lng"),
    "height":       ("osdheight", "heightabovetakeoff", "heightfeet",
                     "osdaltitude", "altitudeabovesealevel", "height"),
    "gimbal_pitch": ("gimbalpitch", "gimbalpitchdegrees"),
    "gimbal_roll":  ("gimbalroll", "gimbalrolldegrees"),
    "gimbal_yaw":   ("gimbalyaw", "gimbalyawdegrees"),
    "osd_yaw":      ("osdyaw", "yaw", "compassheadingdegrees", "heading"),
    "is_video":     ("camerainfoisvideo", "cameraisvideo", "isvideo",
                     "recordingstate", "isrecording"),
}

_TRUE_VALUES = {"true", "1", "yes", "on", "record", "recording"}


def _normalise(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def _find_columns(headers: Sequence[str]) -> Dict[str, str]:
    """Map logical field → actual CSV header, first candidate wins."""
    norm = {_normalise(h): h for h in headers}
    found: Dict[str, str] = {}
    for field, candidates in _COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in norm:
                found[field] = norm[cand]
                break
    return found


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_datetime_s(value: str) -> float:
    """Best-effort ISO/exporter datetime → epoch seconds (nan on failure)."""
    value = (value or "").strip()
    for fmt in (None, "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
        try:
            dt = datetime.fromisoformat(value) if fmt is None \
                else datetime.strptime(value, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return math.nan


class DJILogBook:
    """Time-indexed telemetry from one exported flight-record CSV.

    You may pass either a decoded CSV **or** a raw DJI GO 4 ``.txt`` binary.
    If a ``.txt`` is given, ``resolve_log_path()`` is called automatically to
    find or produce a usable CSV before loading.
    """

    def __init__(self, csv_path: str):
        csv_path = resolve_log_path(csv_path)
        self.path = csv_path
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV: {csv_path}")
            self.columns = _find_columns(reader.fieldnames)
            rows = list(reader)

        missing = [f for f in ("lat", "lon", "height", "gimbal_pitch")
                   if f not in self.columns]
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required telemetry columns: "
                f"{missing}.\nHeaders found: {reader.fieldnames}"
            )

        self._build_arrays(rows)
        print(f"[DJILog] {csv_path}: {len(self.t)} records, "
              f"{self.t[-1] - self.t[0]:.1f}s of flight, "
              f"columns={sorted(self.columns)}")

    # ── construction ──────────────────────────────────────────────────────────
    def _row_time_s(self, row: dict, idx: int) -> float:
        cols = self.columns
        if "time_ms" in cols:
            return _parse_float(row[cols["time_ms"]]) / 1000.0
        if "time_s" in cols:
            return _parse_float(row[cols["time_s"]])
        if "datetime" in cols:
            return _parse_datetime_s(row[cols["datetime"]])
        return idx * 0.1  # flight records tick at 10 Hz

    def _build_arrays(self, rows: List[dict]) -> None:
        cols = self.columns

        def col(field: str, default: float = math.nan) -> np.ndarray:
            if field not in cols:
                return np.full(len(rows), default)
            return np.array([_parse_float(r[cols[field]]) for r in rows])

        t     = np.array([self._row_time_s(r, i) for i, r in enumerate(rows)])
        lat   = col("lat")
        lon   = col("lon")
        hgt   = col("height")
        g_pit = col("gimbal_pitch")
        g_rol = col("gimbal_roll", default=0.0)
        yaw   = col("gimbal_yaw")
        if np.isnan(yaw).all():
            yaw = col("osd_yaw", default=0.0)

        if "is_video" in cols:
            rec = np.array([str(r[cols["is_video"]]).strip().lower()
                            in _TRUE_VALUES for r in rows])
        else:
            rec = np.zeros(len(rows), dtype=bool)

        # Some exports store GPS in radians — Greece would read as lat≈0.66.
        finite = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
        if finite.any() and np.abs(lat[finite]).max() <= math.pi / 2 \
                and np.abs(lon[finite]).max() <= math.pi:
            lat, lon = np.degrees(lat), np.degrees(lon)
            print("[DJILog] GPS looked like radians — converted to degrees.")

        ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
              & np.isfinite(hgt) & np.isfinite(g_pit)
              & (np.abs(lat) > 1e-6))  # drop pre-GPS-lock rows at (0,0)
        if not ok.any():
            raise ValueError(f"No usable telemetry rows in {self.path}")

        order = np.argsort(t[ok], kind="stable")

        def pick(a: np.ndarray) -> np.ndarray:
            return a[ok][order]

        self.t            = pick(t) - t[ok][order][0]   # seconds from first fix
        self.lat          = pick(lat)
        self.lon          = pick(lon)
        self.height       = pick(hgt)
        self.gimbal_pitch = pick(g_pit)
        self.gimbal_roll  = np.nan_to_num(pick(g_rol))
        self.is_video     = pick(rec.astype(float)) > 0.5
        # Unwrap yaw so interpolation doesn't sweep through 360°→0° jumps.
        yaw_ok = np.nan_to_num(pick(yaw))
        self.yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_ok)))

    # ── queries ───────────────────────────────────────────────────────────────
    def first_recording_time(self) -> Optional[float]:
        """Seconds (log clock) when video recording first switched on."""
        idx = np.flatnonzero(self.is_video)
        return float(self.t[idx[0]]) if idx.size else None

    def telemetry_at(self, t_s: float) -> DroneTelemetry:
        """Linearly interpolated camera-frame telemetry at log time t_s."""
        t_s = float(np.clip(t_s, self.t[0], self.t[-1]))
        gimbal_pitch = float(np.interp(t_s, self.t, self.gimbal_pitch))
        return DroneTelemetry(
            lat=float(np.interp(t_s, self.t, self.lat)),
            lon=float(np.interp(t_s, self.t, self.lon)),
            alt_agl=float(np.interp(t_s, self.t, self.height)),
            roll=float(np.interp(t_s, self.t, self.gimbal_roll)),
            pitch=-(90.0 + gimbal_pitch),
            yaw=float(np.interp(t_s, self.t, self.yaw_unwrapped)) % 360.0,
        )

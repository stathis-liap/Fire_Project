"""hindcast_runner.py — Async wrapper for hindcast optimization

Wraps hindcast_optimizer.run_hindcast() + plot_results() for WebSocket delivery.
Runs the (CPU-heavy) optimization in a thread pool, then renders the
diagnostic figure and returns it as base64 PNG.
"""

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.hindcast_optimizer import run_hindcast, plot_results

_DATA_DIR = _ROOT / "data"
_FIRED_GPKG = _DATA_DIR / "fired_greece_2000_to_2024_daily.gpkg"

_NASA_MAP_KEY = "2694cf69813b733c8e75e2fc038bac40"

# Known historical fires — bbox is the FIRMS search box, dates are the
# FIRED/FIRMS query window (auto-extended internally by run_hindcast).
FIRE_PRESETS = {
    "Evoia 2021": {
        "lat_min": 38.50, "lat_max": 39.10,
        "lon_min": 22.70, "lon_max": 23.70,
        "date_start": "2021-08-03", "date_end": "2021-08-05",
    },
    "Rhodes 2023": {
        "lat_min": 35.70, "lat_max": 36.50,
        "lon_min": 27.50, "lon_max": 28.50,
        "date_start": "2023-07-18", "date_end": "2023-07-20",
    },
    "Evros 2023": {
        "lat_min": 40.50, "lat_max": 41.30,
        "lon_min": 25.20, "lon_max": 26.40,
        "date_start": "2023-08-21", "date_end": "2023-08-23",
    },
}


async def run_hindcast_async(
    fire_name: str,
    date_start: str = "",
    date_end: str = "",
    hindcast_hours: float = 6.0,
    maxiter: int = 20,
    popsize: int = 12,
    terrain_buf: float = 0.25,
    progress_callback=None,
) -> dict:
    """
    Run hindcast optimization for a named historical fire.

    Returns dict with keys: error, best_iou, best_params, heatmap_b64, overlay_b64.
    """
    if fire_name not in FIRE_PRESETS:
        return {
            "error": f"Unknown fire: {fire_name}. Expected one of {list(FIRE_PRESETS.keys())}",
            "best_iou": None, "best_params": None,
            "heatmap_b64": None, "overlay_b64": None,
        }

    preset = FIRE_PRESETS[fire_name]
    ds = date_start or preset["date_start"]
    de = date_end or preset["date_end"]

    fired_gpkg = str(_FIRED_GPKG) if _FIRED_GPKG.exists() else None

    async def _progress(step, status, message, **kwargs):
        if progress_callback:
            await progress_callback(
                type="hindcast_progress", step=step, status=status,
                message=message, **kwargs,
            )

    STEP_NAMES = [
        "Fetching fire perimeter data",
        "Identifying ignition point",
        "Fetching ERA5 weather",
        "Fetching terrain (DEM + CORINE)",
        "Building ground-truth mask",
        "Running Differential Evolution optimizer",
        "Rendering diagnostic figure",
    ]

    await _progress(1, "running", STEP_NAMES[0])

    def _run_sync():
        return run_hindcast(
            map_key=_NASA_MAP_KEY,
            lat_min=preset["lat_min"], lat_max=preset["lat_max"],
            lon_min=preset["lon_min"], lon_max=preset["lon_max"],
            date_start=ds, date_end=de,
            hindcast_hours=hindcast_hours,
            maxiter=maxiter, popsize=popsize,
            terrain_buffer=terrain_buf,
            fired_gpkg=fired_gpkg,
            ignition_day=1,
        )

    try:
        # Fire off a lightweight ticker so the browser sees step progress
        # while the (blocking) optimizer runs in a worker thread — since
        # run_hindcast has no callback hook for intermediate step numbers,
        # we approximate by advancing through the known step list on a timer.
        stop_ticker = asyncio.Event()

        async def _ticker():
            for i, name in enumerate(STEP_NAMES[1:6], start=2):
                try:
                    await asyncio.wait_for(stop_ticker.wait(), timeout=max(2.0, hindcast_hours))
                    break
                except asyncio.TimeoutError:
                    await _progress(i, "running", name)

        ticker_task = asyncio.create_task(_ticker())

        result = await asyncio.to_thread(_run_sync)

        stop_ticker.set()
        try:
            await ticker_task
        except Exception:
            pass

        await _progress(7, "running", STEP_NAMES[6])

        # Render diagnostic figure to a temp PNG, then base64-encode it
        heatmap_b64 = ""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "hindcast_result.png")
                await asyncio.to_thread(plot_results, result, out_path)
                with open(out_path, "rb") as f:
                    heatmap_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as exc:
            print(f"[hindcast_runner] plot_results failed: {exc}")

        best_params = result.get("best_params", {})
        # Ensure JSON-serialisable (numpy floats etc.)
        best_params_clean = {k: float(v) for k, v in best_params.items()}

        return {
            "error": None,
            "best_iou": float(result.get("iou", 0.0)),
            "best_params": best_params_clean,
            "heatmap_b64": heatmap_b64,
            "overlay_b64": "",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        await _progress(None, "error", f"Hindcast failed: {exc}")
        return {
            "error": str(exc),
            "best_iou": None, "best_params": None,
            "heatmap_b64": None, "overlay_b64": None,
        }

"""
fire_info_panel.py
=================
Encapsulates fire event generation for the WILSON simulation server.

This module produces structured fire event messages that the client
renders in the footer info panel.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from core.fire_model import _compass_bearing

# Cumulative burned-area tiers (hectares) that trigger a size-classification
# event. Kept coarse (4 tiers, not 8) so the log reports meaningful jumps in
# fire severity instead of every arbitrary threshold.
MILESTONES = [5, 50, 200, 1000]
_SIZE_CLASS = {5: "small fire", 50: "growing fire", 200: "large fire", 1000: "major fire"}
RAPID_SPREAD_THRESHOLD = 30.0   # m/min, internal comparison only (display uses km/h)
RAPID_SPREAD_COOLDOWN_S = 30.0
SLOWDOWN_FACTOR = 0.60
MIN_ACTIVE_FOR_SLOWDOWN = 2.0
INTERVENTION_RECENT_S = 90.0

# ── Spread-report thresholds (sim time, not wall time) ──────────────────────
REPORT_INTERVAL_MIN   = 60.0    # periodic "heading X because Y" report
DIR_CHANGE_COOLDOWN_MIN = 20.0  # min sim-minutes between "fire turned" reports
MIN_REPORT_ROS        = 0.5     # m/min — below this the front isn't really moving
STALL_WINDOW_MIN      = 45.0    # growth window watched for a stall
STALL_GROWTH_DAA      = 5.0     # < this growth inside the window = "stopped growing"
WIND_ALIGN_DEG        = 60.0    # spread within this angle of the wind = wind-driven
WIND_DRIVEN_MIN_MS    = 3.0     # weaker wind than this can't be called the driver
UPHILL_DELTA_M        = 6.0     # mean climb over the look-ahead = slope-driven
DRY_FUEL_MOISTURE     = 0.10    # drier than this = "running through dry fuel"
DAMP_FUEL_MOISTURE    = 0.22    # wetter than this = too damp to catch
BARRIER_NONCOMB_FRAC  = 0.50    # this much bare/urban ground ahead = natural barrier
CALM_WIND_MS          = 1.5

_DIR_WORD    = {"N": "NORTH", "E": "EAST", "S": "SOUTH", "W": "WEST"}
_DIR_BEARING = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

# Icon names (assets/icons/<name>.png) — resolved to <img> tags client-side.
_ICONS: dict[str, str] = {
    "info": "play",
    "weather": "wind",
    "ignition": "flame",
    "milestone": "bar-chart",
    "danger": "zap",
    "warn": "alert-triangle",
    "ok": "check-circle",
    "intervention": "wrench",
}


def _fmt_time(mins: float) -> str:
    h = int(mins // 60)
    m = int(round(mins % 60))
    return f"{h}h {m}min" if h > 0 else f"{m} min"


def _fuel_word(name: str) -> str:
    """'Aleppo_Pine' -> 'aleppo pine' — readable fuel name."""
    return name.replace("_", " ").lower()


def _wind_word(speed_ms: float) -> str:
    if speed_ms >= 10.0:
        return "very strong"
    if speed_ms >= 6.0:
        return "strong"
    return "moderate"


def _pace_word(ros_m_min: float) -> str:
    kmh = ros_m_min * 0.06
    if kmh >= 1.0:
        return "FAST"
    if kmh >= 0.3:
        return "steadily"
    return "slowly"


def _bearing_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _event_dict(event_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "fire_event",
        "event_type": event_type,
        "icon": _ICONS.get(event_type, _ICONS["info"]),
        "message": message,
    }


class FireInfoPanel:
    """Server-side fire panel event state and message generator."""

    def __init__(self) -> None:
        self._started = False
        self._milestone = 0
        self._last_intv: str | None = None
        self._last_intv_ts = 0.0
        self._last_ros_ts = 0.0
        self._humidity = 25.0
        self._last_active = 0.0
        self._extinguished = False
        self._cell_daa = 0.0   # area of one grid cell, in daa — for cells → area display
        # Spread-report state (times in sim minutes)
        self._last_report_min = -1e9
        self._last_dir: str | None = None
        self._area_history: list[tuple[float, float]] = []   # (sim_min, total_daa)
        self._stalled = False

    def init_started(self, rows: int, cols: int, cell_m: float) -> dict[str, Any]:
        self._cell_daa = (cell_m * cell_m) / 1000.0   # 1 daa = 1000 m²
        area_daa = rows * cols * self._cell_daa
        return _event_dict(
            "info",
            f"Simulation started — monitoring {area_daa:,.0f} daa",
        )

    def _cells_to_daa(self, n_cells: int) -> float:
        return n_cells * self._cell_daa

    def weather_update(self,
                       wind_speed_ms: float,
                       wind_direction: float,
                       temperature_c: float,
                       relative_humidity: float) -> dict[str, Any]:
        self._humidity = relative_humidity
        return _event_dict(
            "weather",
            (
                f"Weather change — {wind_speed_ms:.1f} m/s from {_compass_bearing(wind_direction)}"
                f", Humidity: {relative_humidity:.0f}%, {temperature_c:.0f}°C"
            ),
        )

    def intervention(self, action: str, affected_cells: int) -> dict[str, Any]:
        self._last_intv = action
        self._last_intv_ts = time.monotonic()
        area_daa = self._cells_to_daa(affected_cells)
        if action == "firebreak":
            return _event_dict(
                "intervention",
                f"Firebreak cut — {area_daa:.1f} daa cleared.",
            )
        if action == "water_drop":
            return _event_dict(
                "intervention",
                f"Water drop — {area_daa:.1f} daa cooled.",
            )
        if action in ("suppression_line", "containment_line"):
            return _event_dict(
                "intervention",
                f"Containment line holding — {area_daa:.1f} daa protected.",
            )
        return _event_dict("info", f"Intervention: {action} — {area_daa:.1f} daa.")

    def process_frame(self,
                      burned_ha: float,
                      active_ha: float,
                      minutes_since_ignition: float,
                      ros: dict[str, float]) -> List[dict[str, Any]]:
        events: List[dict[str, Any]] = []
        now = time.monotonic()

        if not self._started and active_ha > 0:
            self._started = True
            events.append(_event_dict("ignition", "Fire ignition started"))

        for mha in MILESTONES:
            if burned_ha >= mha and self._milestone < mha:
                self._milestone = mha
                label = _SIZE_CLASS.get(mha, "").upper()
                events.append(_event_dict(
                    "milestone",
                    f"Fire classified as {label} — {mha * 10:.0f} daa burned "
                    f"({_fmt_time(minutes_since_ignition)})",
                ))

        ros_max = max(ros.get("N", 0.0), ros.get("E", 0.0), ros.get("S", 0.0), ros.get("W", 0.0))
        if ros_max > RAPID_SPREAD_THRESHOLD and now - self._last_ros_ts > RAPID_SPREAD_COOLDOWN_S:
            self._last_ros_ts = now
            max_dir = max(ros, key=ros.get)
            events.append(_event_dict(
                "danger",
                f"Rapid spread toward {_DIR_WORD.get(max_dir, max_dir)} — {ros_max * 0.06:.1f} km/h",
            ))

        if self._last_active > MIN_ACTIVE_FOR_SLOWDOWN and active_ha > 0 and active_ha < self._last_active * SLOWDOWN_FACTOR:
            reason = "Fuel depletion — spread slowing"
            if self._last_intv == "firebreak" and now - self._last_intv_ts < INTERVENTION_RECENT_S:
                reason = "Firebreak contained the spread"
            elif self._last_intv == "water_drop" and now - self._last_intv_ts < INTERVENTION_RECENT_S:
                reason = "Increased moisture slowed the fire"
            elif self._humidity > 55:
                reason = f"High humidity ({self._humidity}%) — slowdown"
            events.append(_event_dict("warn", reason))

        if self._last_active > 0.5 and active_ha == 0 and self._started and not self._extinguished:
            self._extinguished = True
            events.append(_event_dict(
                "ok",
                f"Fire extinguished — Total: {burned_ha * 10:.1f} daa",
            ))

        self._last_active = active_ha
        return events

    # ── Spread direction & stall analysis ────────────────────────────────────
    # Turns the sampled front conditions (see server._spread_context) into
    # plain-language reports for the operations log. Deliberately a chain of
    # simple threshold checks, ordered by how decisive each factor is.

    def _heading_reason(self, ctx: Dict[str, Any]) -> str:
        wind_ms  = ctx.get("wind_speed_ms", 0.0)
        gap      = _bearing_gap(_DIR_BEARING[ctx["dir"]],
                                ctx.get("wind_toward_deg", 0.0))
        if wind_ms >= WIND_DRIVEN_MIN_MS and gap <= WIND_ALIGN_DEG:
            return f"pushed by a {_wind_word(wind_ms)} wind"
        if ctx.get("elev_delta_m", 0.0) >= UPHILL_DELTA_M:
            return "climbing uphill (fires accelerate on slopes)"
        if (ctx.get("moisture_ahead", 1.0) < DRY_FUEL_MOISTURE
                and ctx.get("fuel_ahead")):
            return f"running through dry {_fuel_word(ctx['fuel_ahead'])}"
        if ctx.get("fuel_ahead"):
            return f"following the {_fuel_word(ctx['fuel_ahead'])} in its path"
        return "following the driest fuel available"

    def _stall_reason(self, ctx: Dict[str, Any]) -> str:
        now = time.monotonic()
        if (self._last_intv in ("firebreak", "containment_line", "water_drop")
                and now - self._last_intv_ts < INTERVENTION_RECENT_S * 4):
            return "your intervention is holding it back"
        if ctx.get("noncomb_frac", 0.0) > BARRIER_NONCOMB_FRAC:
            return "it has reached ground with nothing to burn (roads, buildings or bare rock)"
        if ctx.get("moisture_ahead", 0.0) > DAMP_FUEL_MOISTURE:
            return "the vegetation ahead is too damp to catch"
        if self._humidity > 55:
            return f"high air humidity ({self._humidity:.0f}%) is protecting the vegetation"
        if ctx.get("wind_speed_ms", 99.0) < CALM_WIND_MS:
            return "the wind has died down and the flames can't reach new fuel"
        return "it is running out of dry fuel"

    def spread_report(self,
                      ctx: Dict[str, Any] | None,
                      total_ha: float,
                      minutes: float) -> List[dict[str, Any]]:
        events: List[dict[str, Any]] = []
        if ctx is None or not self._started or self._extinguished:
            return events

        # ── Stall / regrowth detection over a sliding sim-time window ────────
        self._area_history.append((minutes, total_ha * 10.0))
        cutoff = minutes - STALL_WINDOW_MIN * 1.5
        self._area_history = [(t, a) for t, a in self._area_history if t >= cutoff]
        window = [(t, a) for t, a in self._area_history
                  if t >= minutes - STALL_WINDOW_MIN]
        if len(window) >= 2 and window[-1][0] - window[0][0] >= STALL_WINDOW_MIN * 0.8:
            growth_daa = window[-1][1] - window[0][1]
            if not self._stalled and growth_daa < STALL_GROWTH_DAA:
                self._stalled = True
                events.append(_event_dict(
                    "warn",
                    f"Fire has stopped growing — {self._stall_reason(ctx)}",
                ))
            elif self._stalled and growth_daa >= STALL_GROWTH_DAA * 3:
                self._stalled = False
                self._last_report_min = -1e9   # force a fresh heading report
                events.append(_event_dict(
                    "danger",
                    f"Fire is on the move again — watch the {_DIR_WORD[ctx['dir']]} side",
                ))

        # ── Periodic / direction-change heading report ────────────────────────
        if self._stalled or ctx["ros_m_min"] < MIN_REPORT_ROS:
            return events

        pace  = _pace_word(ctx["ros_m_min"])
        speed = (f"moving {pace}" if pace != "FAST"
                 else f"moving FAST ({ctx['ros_m_min'] * 0.06:.1f} km/h)")
        due   = minutes - self._last_report_min >= REPORT_INTERVAL_MIN

        if not ctx.get("dominant", True):
            # No single front is winning — an "all directions" note beats a
            # noisy stream of compass flips.
            if due or self._last_report_min < 0:
                if ctx.get("wind_speed_ms", 0.0) < WIND_DRIVEN_MIN_MS:
                    why = "no strong wind to steer it, so it burns outward evenly"
                elif ctx.get("fuel_ahead"):
                    why = f"dry {_fuel_word(ctx['fuel_ahead'])} on every side"
                else:
                    why = "dry fuel on every side"
                events.append(_event_dict(
                    "danger" if pace == "FAST" else "milestone",
                    f"Fire is spreading in ALL directions, {speed} — {why}",
                ))
                self._last_report_min = minutes
                self._last_dir        = None
            return events

        dir_changed = (self._last_dir is not None
                       and ctx["dir"] != self._last_dir
                       and minutes - self._last_report_min >= DIR_CHANGE_COOLDOWN_MIN)
        if dir_changed or due or self._last_dir is None:
            head = (f"Fire turned {_DIR_WORD[ctx['dir']]}" if dir_changed
                    else f"Fire is heading {_DIR_WORD[ctx['dir']]}")
            events.append(_event_dict(
                "danger" if pace == "FAST" else "milestone",
                f"{head}, {speed} — {self._heading_reason(ctx)}",
            ))
            self._last_report_min = minutes
            self._last_dir        = ctx["dir"]

        return events

    def simulation_completed(self, count: int) -> dict[str, Any]:
        return _event_dict("ok", "Simulation complete — timeline ready to review")

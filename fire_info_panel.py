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

MILESTONES = [5, 10, 25, 50, 100, 250, 500, 1000]
RAPID_SPREAD_THRESHOLD = 30.0
RAPID_SPREAD_COOLDOWN_S = 30.0
SLOWDOWN_FACTOR = 0.60
MIN_ACTIVE_FOR_SLOWDOWN = 2.0
INTERVENTION_RECENT_S = 90.0

_ICONS: dict[str, str] = {
    "info": "🚀",
    "weather": "💨",
    "ignition": "🔥",
    "milestone": "📊",
    "danger": "⚡",
    "warn": "📉",
    "ok": "✅",
    "intervention": "🛠",
}


def _fmt_time(mins: float) -> str:
    h = int(mins // 60)
    m = int(round(mins % 60))
    return f"{h}h {m}min" if h > 0 else f"{m} min"


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

    def init_started(self, rows: int, cols: int, cell_m: float) -> dict[str, Any]:
        return _event_dict(
            "info",
            f"Simulation started — Grid: {cols}x{rows} cells, {round(cell_m)} m/cell",
        )

    def weather_update(self,
                       wind_speed_ms: float,
                       wind_direction: float,
                       temperature_c: float,
                       relative_humidity: float) -> dict[str, Any]:
        self._humidity = relative_humidity
        return _event_dict(
            "weather",
            (
                f"Weather change — {wind_speed_ms:.1f} m/s from {wind_direction:.0f}°"
                f", Humidity: {relative_humidity:.0f}%, {temperature_c:.0f}°C"
            ),
        )

    def intervention(self, action: str, affected_cells: int) -> dict[str, Any]:
        self._last_intv = action
        self._last_intv_ts = time.monotonic()
        if action == "firebreak":
            return _event_dict(
                "intervention",
                f"Firebreak applied: {affected_cells} cells cleared.",
            )
        if action == "water_drop":
            return _event_dict(
                "intervention",
                f"Water drop: {affected_cells} cells affected.",
            )
        if action in ("suppression_line", "containment_line"):
            return _event_dict(
                "intervention",
                f"Containment line deployed: {affected_cells} cells under protection.",
            )
        return _event_dict("info", f"Intervention: {action} — {affected_cells} cells.")

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
                events.append(_event_dict(
                    "milestone",
                    f"Burned area: {mha} ha ({_fmt_time(minutes_since_ignition)})",
                ))

        ros_max = max(ros.get("N", 0.0), ros.get("E", 0.0), ros.get("S", 0.0), ros.get("W", 0.0))
        if ros_max > RAPID_SPREAD_THRESHOLD and now - self._last_ros_ts > RAPID_SPREAD_COOLDOWN_S:
            self._last_ros_ts = now
            max_dir = max(ros, key=ros.get)
            events.append(_event_dict(
                "danger",
                f"Rapid spread toward {max_dir} — {ros_max:.0f} m/min",
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
                f"Fire extinguished — Total: {burned_ha:.1f} ha",
            ))

        self._last_active = active_ha
        return events

    def simulation_completed(self, count: int) -> dict[str, Any]:
        return _event_dict("info", f"Simulation completed — {count} snapshots")

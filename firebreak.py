from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Coordinate = Tuple[float, float]
ZoneID = int

FIREBREAK_DIR_NAME = "firebreak_zones"
DEFAULT_FILE_PREFIX = "firebreak"
SUPPORTED_EXTENSIONS = (".csv", ".txt")


@dataclass
class FirebreakZone:
    zone_id: ZoneID
    points: List[Coordinate] = field(default_factory=list)
    label: str = ""
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "label": self.label,
            "created_at": self.created_at,
            "points": [list(pt) for pt in self.points],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirebreakZone":
        return cls(
            zone_id=int(data["zone_id"]),
            points=[(float(lat), float(lon)) for lat, lon in data.get("points", [])],
            label=str(data.get("label", "")),
            created_at=float(data.get("created_at", time.time())),
            meta=dict(data.get("meta", {})),
        )

    def add_point(self, coordinate: Coordinate) -> None:
        self.points.append(coordinate)

    def update_points(self, points: Iterable[Coordinate]) -> None:
        self.points = [tuple(pt) for pt in points]

    def remove_point(self, index: int) -> None:
        if index < 0 or index >= len(self.points):
            raise IndexError("Point index out of range")
        del self.points[index]

    def clear_points(self) -> None:
        self.points.clear()

    def is_valid(self) -> bool:
        return len(self.points) >= 2


class FirebreakManager:
    def __init__(self) -> None:
        self.storage_dir = Path(__file__).resolve().parent / FIREBREAK_DIR_NAME
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.zones: List[FirebreakZone] = []
        self.active_zone_id: Optional[ZoneID] = None
        self._undo_stack: List[List[Dict[str, Any]]] = []
        self._redo_stack: List[List[Dict[str, Any]]] = []
        self._next_zone_id: ZoneID = 1

    def _snapshot(self) -> List[Dict[str, Any]]:
        return [zone.to_dict() for zone in self.zones]

    def _push_history(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def _restore_snapshot(self, snapshot: List[Dict[str, Any]]) -> None:
        self.zones = [FirebreakZone.from_dict(item) for item in snapshot]
        self.active_zone_id = self.zones[-1].zone_id if self.zones else None
        self._next_zone_id = max((zone.zone_id for zone in self.zones), default=0) + 1

    def list_saved_files(self) -> List[Path]:
        return sorted(
            [path for path in self.storage_dir.iterdir()
             if path.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=lambda p: p.name,
        )

    def create_zone(self, label: str = "") -> FirebreakZone:
        self._push_history()
        zone = FirebreakZone(zone_id=self._next_zone_id, label=label)
        self._next_zone_id += 1
        self.zones.append(zone)
        self.active_zone_id = zone.zone_id
        return zone

    def select_zone(self, zone_id: ZoneID) -> Optional[FirebreakZone]:
        if any(zone.zone_id == zone_id for zone in self.zones):
            self.active_zone_id = zone_id
            return self.get_zone(zone_id)
        return None

    def get_zone(self, zone_id: ZoneID) -> Optional[FirebreakZone]:
        return next((zone for zone in self.zones if zone.zone_id == zone_id), None)

    def add_point_to_active_zone(self, coordinate: Coordinate) -> FirebreakZone:
        if self.active_zone_id is None:
            raise ValueError("No active firebreak zone selected")
        zone = self.get_zone(self.active_zone_id)
        if zone is None:
            raise ValueError(f"Active zone {self.active_zone_id} not found")
        self._push_history()
        zone.add_point(coordinate)
        return zone

    def update_zone_points(self, zone_id: ZoneID, points: Iterable[Coordinate]) -> FirebreakZone:
        zone = self.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} not found")
        self._push_history()
        zone.update_points(points)
        return zone

    def delete_zone(self, zone_id: ZoneID) -> bool:
        zone = self.get_zone(zone_id)
        if zone is None:
            return False
        self._push_history()
        self.zones = [z for z in self.zones if z.zone_id != zone_id]
        if self.active_zone_id == zone_id:
            self.active_zone_id = self.zones[-1].zone_id if self.zones else None
        return True

    def clear_all(self) -> None:
        if not self.zones:
            return
        self._push_history()
        self.zones.clear()
        self.active_zone_id = None

    def finish_active_zone(self) -> Optional[FirebreakZone]:
        if self.active_zone_id is None:
            return None
        zone = self.get_zone(self.active_zone_id)
        self.active_zone_id = None
        return zone

    def has_zones(self) -> bool:
        return bool(self.zones)

    def save(self, filename: Optional[str] = None) -> Path:
        if filename:
            export_path = self.storage_dir / filename
        else:
            timestamp = int(time.time())
            export_path = self.storage_dir / f"{DEFAULT_FILE_PREFIX}_{timestamp}.csv"

        export_path = export_path.with_suffix(export_path.suffix or ".csv")
        if export_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            export_path = export_path.with_suffix(".csv")

        if export_path.suffix.lower() == ".csv":
            self._save_csv(export_path)
        else:
            self._save_txt(export_path)

        return export_path

    def _save_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["zone_id", "label", "point_index", "lat", "lon"])
            for zone in self.zones:
                for idx, point in enumerate(zone.points):
                    writer.writerow([zone.zone_id, zone.label, idx, point[0], point[1]])

    def _save_txt(self, path: Path) -> None:
        data = {"zones": [zone.to_dict() for zone in self.zones]}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def load(self, filename: str) -> Path:
        path = self.storage_dir / filename
        if not path.exists():
            saved = self.list_saved_files()
            if not saved:
                raise FileNotFoundError(
                    f"No saved firebreak zones found in '{self.storage_dir}'. "
                    "Please add a .csv or .txt file there or design a new zone."
                )
            raise FileNotFoundError(f"File '{filename}' not found in '{self.storage_dir}'.")

        if path.suffix.lower() == ".csv":
            self._load_csv(path)
        else:
            self._load_txt(path)

        return path

    def _load_csv(self, path: Path) -> None:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            zones: Dict[ZoneID, FirebreakZone] = {}
            for row in reader:
                zone_id = int(row["zone_id"])
                label = row.get("label", "")
                lat = float(row["lat"])
                lon = float(row["lon"])
                if zone_id not in zones:
                    zones[zone_id] = FirebreakZone(zone_id=zone_id, label=label)
                zones[zone_id].add_point((lat, lon))
        self._push_history()
        self.zones = list(zones.values())
        self.active_zone_id = self.zones[-1].zone_id if self.zones else None
        self._next_zone_id = max((zone.zone_id for zone in self.zones), default=0) + 1

    def _load_txt(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        zones = [FirebreakZone.from_dict(item) for item in data.get("zones", [])]
        self._push_history()
        self.zones = zones
        self.active_zone_id = self.zones[-1].zone_id if self.zones else None
        self._next_zone_id = max((zone.zone_id for zone in self.zones), default=0) + 1

    def to_geojson(self) -> Dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"zone_id": zone.zone_id, "label": zone.label},
                    "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in zone.points]},
                }
                for zone in self.zones if zone.is_valid()
            ],
        }

    def apply(
        self,
        sim: Any,
        geo_grid: Any,
        strength: float = 1.0,
        effect_radius_m: float = 0.0,
        water_application: float = 0.0,
        humidity_boost: float = 0.0,
        decay_hours: float = 0.0,
    ) -> int:
        if not self.zones:
            return 0

        line_mask = self._rasterize_zones(geo_grid)
        if hasattr(sim, "apply_containment_line"):
            return sim.apply_containment_line(
                line_mask,
                strength=float(strength),
                effect_radius_cells=max(1, int(round(effect_radius_m / getattr(sim, "cell_m", 1.0)))),
                water_application=float(water_application),
                humidity_boost=float(humidity_boost),
                decay_rate=self._decay_rate(sim, strength, decay_hours),
            )

        if hasattr(sim, "apply_firebreak_mask"):
            return sim.apply_firebreak_mask(line_mask)

        raise NotImplementedError("Simulation object does not support firebreak application.")

    def _decay_rate(self, sim: Any, strength: float, decay_hours: float) -> float:
        dt_minutes = float(getattr(sim, "dt", 1.0))
        if decay_hours > 0.0 and dt_minutes > 0.0:
            steps_lifetime = max(1.0, decay_hours * 60.0 / dt_minutes)
            return float(strength) / steps_lifetime
        return 0.0

    def _rasterize_zones(self, geo_grid: Any) -> Any:
        if not hasattr(geo_grid, "latlon_to_rc"):
            raise AttributeError("geo_grid must provide latlon_to_rc(lat, lon)")

        mask = None
        for zone in self.zones:
            if not zone.is_valid():
                continue
            for first, second in zip(zone.points, zone.points[1:]):
                r0, c0 = geo_grid.latlon_to_rc(first[0], first[1])
                r1, c1 = geo_grid.latlon_to_rc(second[0], second[1])
                if mask is None:
                    rows, cols = getattr(geo_grid, "rows", None), getattr(geo_grid, "cols", None)
                    if rows is None or cols is None:
                        raise ValueError("geo_grid must provide rows and cols attributes for rasterization")
                    mask = self._empty_mask(rows, cols)
                self._bresenham_line(mask, r0, c0, r1, c1)

        if mask is None:
            raise ValueError("No valid firebreak zones to rasterize")
        return mask

    def _empty_mask(self, rows: int, cols: int) -> Any:
        import numpy as np
        return np.zeros((rows, cols), dtype=bool)

    def _bresenham_line(self, mask: Any, r0: int, c0: int, r1: int, c1: int) -> None:
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        step_r = 1 if r0 < r1 else -1
        step_c = 1 if c0 < c1 else -1
        err = dr - dc

        while True:
            if 0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1]:
                mask[r0, c0] = True
            if r0 == r1 and c0 == c1:
                break
            err2 = err * 2
            if err2 > -dc:
                err -= dc
                r0 += step_r
            if err2 < dr:
                err += dr
                c0 += step_c


__all__ = ["FirebreakManager", "FirebreakZone"]

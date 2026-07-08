"""
pipeline/threat_advisor.py
===========================
Live threat assessment for the running simulation:

  • Tracks every named settlement (OSM place nodes; falls back to residential
    polygon centroids when the places layer is unavailable).
  • Every assessment computes distance from the active fire front, closing
    speed, and projected arrival time (ETA) per settlement, and assigns a
    danger level: safe → watch → warning → critical → impact.
  • On level escalation it produces plain-language report lines, action
    suggestions (aerial water drops / containment line on the approach axis),
    and an escape route along the OSM road graph that avoids the fire
    (Dijkstra with fire-proximity edge penalties).

The server forwards:
  {"type": "threat_update", threats, suggestions, routes}   → map overlays
  {"type": "fire_event", ...}                                → language feed
"""

from __future__ import annotations

import heapq
import json
import math
import os
from typing import Optional

import numpy as np

_LEVELS = ["safe", "watch", "warning", "critical", "impact"]
_LEVEL_RANK = {lv: i for i, lv in enumerate(_LEVELS)}

# Assessment thresholds
_ETA_WATCH_MIN    = 360.0   # projected arrival within 6 h  → watch
_ETA_WARNING_MIN  = 120.0   # within 2 h                    → warning
_ETA_CRITICAL_MIN = 30.0    # within 30 min                 → critical
_DIST_WARNING_M   = 1500.0
_DIST_CRITICAL_M  = 500.0

_MAX_THREATS_SENT = 12      # payload cap: nearest N settlements
_MAX_ROUTES       = 3       # escape routes computed per assessment

# Road classes used for evacuation routing (service tracks excluded)
_ROUTABLE = {"motorway", "trunk", "primary", "secondary", "tertiary",
             "residential", "unclassified"}
_ROAD_WORD = {"motorway": "the motorway", "trunk": "the trunk road",
              "primary": "the main road", "secondary": "the secondary road",
              "tertiary": "the local road", "residential": "village streets",
              "unclassified": "the country road"}


def _compass(bearing_deg: float) -> str:
    dirs = ["north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west"]
    return dirs[int((bearing_deg + 22.5) / 45) % 8]


def _bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing (deg) from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(la2)
    y = (math.cos(la1) * math.sin(la2)
         - math.sin(la1) * math.cos(la2) * math.cos(dlon))
    return math.degrees(math.atan2(x, y)) % 360.0


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = la2 - la1
    dlo = math.radians(lon2 - lon1)
    a = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _fmt_eta(minutes: float) -> str:
    if minutes < 60:
        return f"~{int(round(minutes))} min"
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    return f"~{h}h {m:02d}m"


def _fmt_dist(m: float) -> str:
    return f"{m/1000:.1f} km" if m >= 1000 else f"{int(round(m/10)*10)} m"


class ThreatAdvisor:
    """Per-connection advisor. Build once after the landscape; call
    assess(sim, minutes) periodically from the frame loop."""

    def __init__(self, geo_grid, cell_m: float,
                 osm_path: Optional[str] = None,
                 places_path: Optional[str] = None):
        self.geo = geo_grid
        self.cell_m = float(cell_m)
        self.settlements: list[dict] = []
        self._history: dict[str, dict] = {}    # name → {minutes, dist, ema, level}
        self._route_cache: dict[str, dict] = {}

        # Road graph
        self._node_coord: list[tuple[float, float]] = []   # (lat, lon)
        self._adj: list[list[tuple[int, float, str]]] = [] # idx → [(nbr, m, class)]
        self._exit_nodes: set[int] = set()
        self._node_rc = np.zeros((0, 2), dtype=np.int32)

        if osm_path and os.path.exists(osm_path):
            try:
                self._build_road_graph(osm_path)
            except Exception as exc:
                print(f"[Advisor] Road graph build failed: {exc}")

        self._load_settlements(places_path, osm_path)

    # ── Construction ─────────────────────────────────────────────────────────

    def _load_settlements(self, places_path, osm_path) -> None:
        geo = self.geo
        feats = []
        if places_path and os.path.exists(places_path):
            try:
                feats = json.load(open(places_path)).get("features", [])
            except Exception:
                feats = []

        for f in feats:
            try:
                lon, lat = f["geometry"]["coordinates"][:2]
                p = f.get("properties", {})
                if not (geo.lat_min <= lat <= geo.lat_max
                        and geo.lon_min <= lon <= geo.lon_max):
                    continue
                self.settlements.append({
                    "name":  p.get("name") or "settlement",
                    "place": p.get("place", "village"),
                    "lat": float(lat), "lon": float(lon),
                })
            except Exception:
                continue

        # Fallback: unnamed residential polygons → centroid "settlements"
        if not self.settlements and osm_path and os.path.exists(osm_path):
            try:
                n = 0
                for f in json.load(open(osm_path)).get("features", []):
                    p = f.get("properties", {})
                    g = f.get("geometry", {}) or {}
                    if p.get("landuse") != "residential" or g.get("type") != "Polygon":
                        continue
                    ring = np.asarray(g["coordinates"][0], dtype=np.float64)
                    if len(ring) < 8:      # skip tiny hamlet fragments
                        continue
                    lon, lat = float(ring[:, 0].mean()), float(ring[:, 1].mean())
                    if not (geo.lat_min <= lat <= geo.lat_max
                            and geo.lon_min <= lon <= geo.lon_max):
                        continue
                    n += 1
                    self.settlements.append({
                        "name": f"Settlement {n}", "place": "village",
                        "lat": lat, "lon": lon,
                    })
            except Exception as exc:
                print(f"[Advisor] Residential fallback failed: {exc}")

        # Deduplicate close pairs (same village as node + polygon)
        kept = []
        for s in self.settlements:
            if all(_haversine_m(s["lat"], s["lon"], k["lat"], k["lon"]) > 400
                   for k in kept):
                kept.append(s)
        self.settlements = kept

        for s in self.settlements:
            s["rc"] = self.geo.latlon_to_rc(s["lat"], s["lon"])
            s["node"] = self._nearest_node(s["lat"], s["lon"])
            # Unique key — Greek villages frequently share names
            s["key"] = f"{s['name']}@{s['lat']:.3f},{s['lon']:.3f}"
        print(f"[Advisor] Tracking {len(self.settlements)} settlements, "
              f"{len(self._node_coord)} road nodes, "
              f"{len(self._exit_nodes)} map exits")

    def _build_road_graph(self, osm_path: str) -> None:
        geo = self.geo
        node_id: dict[tuple[int, int], int] = {}

        def _nid(lat, lon):
            key = (int(round(lat * 1e5)), int(round(lon * 1e5)))
            i = node_id.get(key)
            if i is None:
                i = len(self._node_coord)
                node_id[key] = i
                self._node_coord.append((lat, lon))
                self._adj.append([])
            return i

        for f in json.load(open(osm_path)).get("features", []):
            p = f.get("properties", {})
            g = f.get("geometry", {}) or {}
            hw = p.get("highway", "")
            if p.get("feature_type") != "road" or hw not in _ROUTABLE:
                continue
            if g.get("type") != "LineString":
                continue
            coords = g["coordinates"]
            prev = None
            for lon, lat in coords:
                cur = _nid(lat, lon)
                if prev is not None and prev != cur:
                    d = _haversine_m(*self._node_coord[prev], lat, lon)
                    self._adj[prev].append((cur, d, hw))
                    self._adj[cur].append((prev, d, hw))
                prev = cur

        # Map-boundary exits: road nodes in the outer 4 % margin of the domain
        mlat = (geo.lat_max - geo.lat_min) * 0.04
        mlon = (geo.lon_max - geo.lon_min) * 0.04
        for i, (lat, lon) in enumerate(self._node_coord):
            if (lat < geo.lat_min + mlat or lat > geo.lat_max - mlat
                    or lon < geo.lon_min + mlon or lon > geo.lon_max - mlon):
                self._exit_nodes.add(i)

        # Precompute grid rc per node for fast fire-distance lookups
        self._node_rc = np.array(
            [self.geo.latlon_to_rc(lat, lon) for lat, lon in self._node_coord],
            dtype=np.int32).reshape(-1, 2)

    def _nearest_node(self, lat, lon) -> Optional[int]:
        if not self._node_coord:
            return None
        arr = np.asarray(self._node_coord, dtype=np.float64)
        coslat = math.cos(math.radians(lat))
        d2 = ((arr[:, 0] - lat)) ** 2 + ((arr[:, 1] - lon) * coslat) ** 2
        i = int(np.argmin(d2))
        # Reject if the nearest road is further than ~2 km (off-network)
        if _haversine_m(lat, lon, *self._node_coord[i]) > 2000:
            return None
        return i

    # ── Assessment ───────────────────────────────────────────────────────────

    def assess(self, sim, minutes: float) -> Optional[dict]:
        """Return {threats, events, suggestions, routes} or None when idle."""
        if not self.settlements:
            return None
        burning = (sim.state == 1)
        if not burning.any():
            return None

        rows, cols = sim.state.shape
        try:
            from scipy.ndimage import distance_transform_edt
            dist_cells, idx = distance_transform_edt(
                ~burning, return_indices=True)
        except Exception:
            return None
        dist_m = dist_cells * self.cell_m

        events, suggestions, routes, threats = [], [], [], []

        for s in self.settlements:
            r, c = s["rc"]
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            d = float(dist_m[r, c])
            h = self._history.setdefault(
                s["key"], {"minutes": minutes, "dist": d, "ema": 0.0,
                           "level": "safe"})

            dt = minutes - h["minutes"]
            if dt > 0.01:
                closing = (h["dist"] - d) / dt          # m per sim-minute
                h["ema"] = 0.5 * h["ema"] + 0.5 * closing
                h["minutes"], h["dist"] = minutes, d
            closing = h["ema"]

            eta = d / closing if closing > 1.0 else None
            already_burned = sim.state[r, c] == 2

            if already_burned or d <= self.cell_m:
                level = "impact"
            elif (eta is not None and eta <= _ETA_CRITICAL_MIN) or \
                 (d <= _DIST_CRITICAL_M and closing > 0):
                level = "critical"
            elif (eta is not None and eta <= _ETA_WARNING_MIN) or \
                 (d <= _DIST_WARNING_M and closing > 0):
                level = "warning"
            elif eta is not None and eta <= _ETA_WATCH_MIN:
                level = "watch"
            else:
                level = "safe"

            threats.append({
                "name": s["name"], "key": s["key"], "place": s["place"],
                "lat": s["lat"], "lon": s["lon"],
                "distance_m": round(d), "closing_m_min": round(closing, 1),
                "eta_min": round(eta, 1) if eta is not None else None,
                "level": level,
            })

            # ── Language + suggestions on escalation ─────────────────────────
            prev = h["level"]
            if _LEVEL_RANK[level] > _LEVEL_RANK[prev]:
                # nearest point of the fire front to this settlement
                fr, fc = int(idx[0][r, c]), int(idx[1][r, c])
                f_lat, f_lon = self._rc_to_latlon(fr, fc)
                brg = _bearing(s["lat"], s["lon"], f_lat, f_lon)
                from_dir = _compass(brg)
                eta_txt = f", projected arrival {_fmt_eta(eta)}" if eta else ""

                if level == "watch":
                    events.append(self._ev("warn", "clock",
                        f"{s['name']} — fire {_fmt_dist(d)} to the {from_dir}, "
                        f"closing at ~{closing:.0f} m/min{eta_txt}. "
                        f"Monitoring approach."))
                elif level == "warning":
                    events.append(self._ev("warn", "alert-triangle",
                        f"WARNING {s['name']} — fire {_fmt_dist(d)} away"
                        f"{eta_txt}. Recommend suppression on the approach "
                        f"and preparing evacuation."))
                elif level == "critical":
                    events.append(self._ev("danger", "zap",
                        f"CRITICAL {s['name']} — fire {_fmt_dist(d)} away"
                        f"{eta_txt}. Evacuate now via the marked route."))
                elif level == "impact":
                    events.append(self._ev("danger", "flame",
                        f"Fire has reached {s['name']}."))

                if level in ("warning", "critical"):
                    suggestions.extend(self._make_suggestions(s, fr, fc, d))
            elif _LEVEL_RANK[level] < _LEVEL_RANK[prev] - 1 or \
                    (prev in ("warning", "critical") and level in ("safe", "watch")):
                events.append(self._ev("ok", "check-circle",
                    f"{s['name']} no longer under immediate threat "
                    f"(front stalled or receding)."))
                self._route_cache.pop(s["key"], None)
            h["level"] = level

        # ── Escape routes for the most-threatened settlements ────────────────
        hot = sorted((t for t in threats
                      if t["level"] in ("warning", "critical")),
                     key=lambda t: _LEVEL_RANK[t["level"]] * -1000
                                   + (t["eta_min"] or 9999))
        for t in hot[:_MAX_ROUTES]:
            route = self._escape_route(t, dist_m)
            if route:
                routes.append(route)
                if route.pop("_is_new", False):
                    events.append(self._ev("info", "map-pin", route["message"]))

        threats.sort(key=lambda t: (-_LEVEL_RANK[t["level"]],
                                    t["eta_min"] if t["eta_min"] is not None else 1e9,
                                    t["distance_m"]))
        return {
            "threats":     threats[:_MAX_THREATS_SENT],
            "events":      events,
            "suggestions": suggestions,
            "routes":      routes,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        geo = self.geo
        lat = geo.lat_min + (r + 0.5) * (geo.lat_max - geo.lat_min) / geo.rows
        lon = geo.lon_min + (c + 0.5) * (geo.lon_max - geo.lon_min) / geo.cols
        return lat, lon

    @staticmethod
    def _ev(event_type: str, icon: str, message: str) -> dict:
        return {"type": "fire_event", "event_type": event_type,
                "icon": icon, "message": message}

    def _make_suggestions(self, s: dict, fr: int, fc: int, dist: float) -> list:
        """Water-drop point + containment line on the fire→settlement axis."""
        f_lat, f_lon = self._rc_to_latlon(fr, fc)
        # point 35 % of the way from the front toward the settlement
        w_lat = f_lat + 0.35 * (s["lat"] - f_lat)
        w_lon = f_lon + 0.35 * (s["lon"] - f_lon)
        brg_f2s = _bearing(f_lat, f_lon, s["lat"], s["lon"])
        out = [{
            "kind": "water_drop", "for": s["name"],
            "lat": round(w_lat, 5), "lon": round(w_lon, 5), "radius_m": 435,
            "message": (f"Suggest aerial water drops "
                        f"{_fmt_dist(dist * 0.35)} {_compass(brg_f2s)} of the "
                        f"front to slow the head approaching {s['name']}."),
        }]
        # containment line perpendicular to the approach axis, 60 % of the way
        c_lat = f_lat + 0.6 * (s["lat"] - f_lat)
        c_lon = f_lon + 0.6 * (s["lon"] - f_lon)
        half_m = 600.0
        perp = math.radians((brg_f2s + 90.0) % 360.0)
        dlat = (half_m / 111_320.0) * math.cos(perp)
        dlon = (half_m / (111_320.0 * math.cos(math.radians(c_lat)))) * math.sin(perp)
        out.append({
            "kind": "containment_line", "for": s["name"],
            "points": [{"lat": round(c_lat - dlat, 5), "lon": round(c_lon - dlon, 5)},
                       {"lat": round(c_lat + dlat, 5), "lon": round(c_lon + dlon, 5)}],
            "message": (f"Suggest a containment line across the approach, "
                        f"~{_fmt_dist(dist * 0.6)} out from the front, "
                        f"to shield {s['name']}."),
        })
        return out

    # ── Escape routing ───────────────────────────────────────────────────────

    def _edge_penalty(self, fire_dist_m: float) -> float:
        if fire_dist_m <= 300.0:
            return 1e6      # effectively blocked
        if fire_dist_m <= 1000.0:
            return 25.0
        if fire_dist_m <= 2000.0:
            return 6.0
        return 1.0

    def _escape_route(self, threat: dict, dist_m: np.ndarray) -> Optional[dict]:
        name = threat["name"]
        key  = threat["key"]
        s = next((x for x in self.settlements if x["key"] == key), None)
        if s is None or s.get("node") is None or not self._exit_nodes:
            return None

        cached = self._route_cache.get(key)
        if cached is not None:
            # Reuse until any route node comes within 300 m of the fire
            rc = cached["_node_rc"]
            rows, cols = dist_m.shape
            ok = True
            for r, c in rc:
                if 0 <= r < rows and 0 <= c < cols and dist_m[r, c] <= 300.0:
                    ok = False
                    break
            if ok:
                out = dict(cached)
                out.pop("_node_rc", None)
                out["_is_new"] = False
                return out
            self._route_cache.pop(key, None)

        rows, cols = dist_m.shape
        rc = self._node_rc
        in_grid = ((rc[:, 0] >= 0) & (rc[:, 0] < rows)
                   & (rc[:, 1] >= 0) & (rc[:, 1] < cols))
        node_fire = np.full(len(self._node_coord), 1e9, dtype=np.float64)
        node_fire[in_grid] = dist_m[rc[in_grid, 0], rc[in_grid, 1]]

        start = s["node"]
        INF = float("inf")
        best = {start: 0.0}
        prev: dict[int, int] = {}
        pq = [(0.0, start)]
        goal = None
        while pq:
            cost, u = heapq.heappop(pq)
            if cost > best.get(u, INF):
                continue
            if u in self._exit_nodes and node_fire[u] > 2000.0:
                goal = u
                break
            for v, d, hw in self._adj[u]:
                pen = self._edge_penalty(min(node_fire[u], node_fire[v]))
                if pen >= 1e6:
                    continue
                nc = cost + d * pen
                if nc < best.get(v, INF):
                    best[v] = nc
                    prev[v] = u
                    heapq.heappush(pq, (nc, v))
        if goal is None:
            return {
                "for": name, "level": threat["level"], "coords": [],
                "length_km": 0, "direction": "",
                "message": (f"No clear road route out of {name} — all mapped "
                            f"roads pass near the fire. Shelter in place and "
                            f"await ground guidance."),
                "_is_new": cached is None,
            }

        # Reconstruct path
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()

        coords = [[round(self._node_coord[i][1], 5),
                   round(self._node_coord[i][0], 5)] for i in path]
        length_m = sum(
            _haversine_m(self._node_coord[a][0], self._node_coord[a][1],
                         self._node_coord[b][0], self._node_coord[b][1])
            for a, b in zip(path, path[1:]))
        # Dominant road class for language
        hw_len: dict[str, float] = {}
        for a, b in zip(path, path[1:]):
            for v, d, hw in self._adj[a]:
                if v == b:
                    hw_len[hw] = hw_len.get(hw, 0.0) + d
                    break
        dom_hw = max(hw_len, key=hw_len.get) if hw_len else "unclassified"
        end_lat, end_lon = self._node_coord[goal]
        direction = _compass(_bearing(s["lat"], s["lon"], end_lat, end_lon))

        # Thin the polyline for payload (keep endpoints)
        if len(coords) > 80:
            step = max(1, len(coords) // 79)
            coords = coords[::step] + [coords[-1]]

        route = {
            "for": name, "level": threat["level"], "coords": coords,
            "length_km": round(length_m / 1000, 1), "direction": direction,
            "message": (f"Escape route for {name}: head {direction} via "
                        f"{_ROAD_WORD.get(dom_hw, 'the road')} "
                        f"({length_m/1000:.1f} km to the map edge, "
                        f"clear of the fire)."),
            "_node_rc": [tuple(self._node_rc[i]) for i in path],
            "_is_new": True,
        }
        self._route_cache[key] = dict(route)
        out = dict(route)
        out.pop("_node_rc", None)
        return out


def build_advisor(geo_grid, cell_m: float,
                  osm_path: Optional[str]) -> Optional[ThreatAdvisor]:
    """Factory used by the server: fetch the places layer (cached) and build.
    Returns None when there is no OSM data at all (synthetic terrain)."""
    places_path = None
    try:
        from pipeline.auto_fetcher import fetch_osm_places
        places_path = fetch_osm_places(geo_grid.lon_min, geo_grid.lat_min,
                                       geo_grid.lon_max, geo_grid.lat_max)
    except Exception as exc:
        print(f"[Advisor] Places fetch failed: {exc}")

    if not osm_path and not places_path:
        return None
    adv = ThreatAdvisor(geo_grid, cell_m, osm_path=osm_path,
                        places_path=places_path)
    return adv if adv.settlements else None

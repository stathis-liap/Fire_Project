"""geometry.py
Pinhole Camera Model — maps a fire pixel (u, v) detected in vision.py
to an absolute GPS coordinate using live drone telemetry.

Coordinate conventions
----------------------
NED  : North-East-Down world frame (standard MAVLink / autopilot frame)
Body : Forward-Right-Down body frame attached to the drone
Cam  : OpenCV camera frame  X=right, Y=down, Z=into scene (optical axis)

Default assumption: camera mounted nadir (straight down).
Adjust cam_pitch_offset_deg in CameraIntrinsics for forward-tilted gimbals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Earth model (WGS-84 mean radius)
# ──────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    For a 1920×1080 sensor with ~90° diagonal FOV (typical drone camera):
        fx ≈ fy ≈ 980,  cx = 960,  cy = 540

    Obtain precise values via OpenCV calibrateCamera() for your specific lens.
    cam_pitch_offset_deg: 0° = pure nadir; positive = camera pitched forward.
    """
    fx: float = 980.0
    fy: float = 980.0
    cx: float = 960.0
    cy: float = 540.0
    cam_pitch_offset_deg: float = 0.0


@dataclass
class DroneTelemetry:
    """
    Drone state snapshot at the moment of frame capture.
    Angles in degrees; altitude in metres AGL; GPS in decimal degrees WGS-84.
    """
    lat: float        # Drone latitude  (decimal degrees, N positive)
    lon: float        # Drone longitude (decimal degrees, E positive)
    alt_agl: float    # Altitude above ground level (metres)
    roll: float       # Roll  (degrees, positive = right bank)
    pitch: float      # Pitch (degrees, positive = nose up)
    yaw: float        # Yaw / heading (degrees, 0 = North, clockwise positive)


@dataclass
class GeolocResult:
    lat: float
    lon: float
    confidence: float            # 0–1; reduced by large attitude angles / oblique rays
    ray_NED: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ground_range_m: float = 0.0  # horizontal distance from drone to fire (metres)


# ──────────────────────────────────────────────────────────────────────────────
# Elementary rotation matrices (angles in radians)
# ──────────────────────────────────────────────────────────────────────────────
def _Rx(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def _Ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def _Rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed nadir camera → body rotation
#   X_cam (image right)  →  Y_body (drone right)
#   Y_cam (image down)   →  X_body (drone forward)
#   Z_cam (optical axis) →  Z_body (drone down)
# ──────────────────────────────────────────────────────────────────────────────
_R_CAM2BODY_NADIR = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core geolocator
# ──────────────────────────────────────────────────────────────────────────────
class FireGeolocator:
    """
    Projects a fire pixel from vision.py to GPS coordinates on the ground.

    Usage
    -----
    geolocator = FireGeolocator(CameraIntrinsics(fx=980, fy=980, cx=960, cy=540))
    result = geolocator.pixel_to_gps(
        pixel_uv=(u, v),
        telemetry=DroneTelemetry(lat=..., lon=..., alt_agl=..., roll=..., pitch=..., yaw=...),
    )
    if result:
        print(f"Fire at {result.lat:.6f}, {result.lon:.6f}")
    """

    def __init__(self, intrinsics: Optional[CameraIntrinsics] = None):
        self.intrinsics = intrinsics or CameraIntrinsics()
        # Pre-compose camera pitch offset into cam→body rotation once at init.
        pitch_off_rad = math.radians(self.intrinsics.cam_pitch_offset_deg)
        self._R_cam2body: np.ndarray = _Ry(-pitch_off_rad) @ _R_CAM2BODY_NADIR

    def pixel_to_gps(
        self,
        pixel_uv: Tuple[float, float],
        telemetry: DroneTelemetry,
    ) -> Optional[GeolocResult]:
        """
        Map a single pixel to a GPS coordinate on the flat-earth ground plane.

        Returns None when the ray does not intersect the ground (e.g. camera
        points above the horizon due to extreme pitch).
        """
        K = self.intrinsics
        u, v = float(pixel_uv[0]), float(pixel_uv[1])

        # ── Step 1: pixel → normalised camera-frame direction vector ──────────
        x_n = (u - K.cx) / K.fx
        y_n = (v - K.cy) / K.fy
        d_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)   # not unit-length; OK

        # ── Step 2: camera frame → drone body frame ───────────────────────────
        d_body = self._R_cam2body @ d_cam

        # ── Step 3: body frame → NED world frame (ZYX Euler: yaw→pitch→roll) ──
        #   MAVLink sign convention:
        #     pitch-up = positive  → camera ray tilts forward; negate for NED
        #     roll-right = positive → camera tilts right; negate for NED
        psi   = math.radians(telemetry.yaw)
        theta = math.radians(telemetry.pitch)
        phi   = math.radians(telemetry.roll)
        R_body2NED = _Rz(psi) @ _Ry(-theta) @ _Rx(-phi)
        d_NED = R_body2NED @ d_body

        # ── Step 4: intersect ray with flat ground plane ──────────────────────
        #   In NED, Z is positive downward.
        #   Ray from drone origin: P(t) = t * d_NED
        #   Ground condition:      P_z(t) = alt_agl  →  t = alt_agl / d_NED[2]
        if d_NED[2] <= 1e-6:
            # Ray points up or horizontally — no valid ground intersection.
            return None

        t = telemetry.alt_agl / d_NED[2]
        north_m = t * d_NED[0]
        east_m  = t * d_NED[1]
        ground_range_m = math.hypot(north_m, east_m)

        # ── Step 5: metric offsets → GPS (equirectangular / small-angle) ──────
        lat_rad = math.radians(telemetry.lat)
        d_lat   = math.degrees(north_m / EARTH_RADIUS_M)
        d_lon   = math.degrees(east_m  / (EARTH_RADIUS_M * math.cos(lat_rad)))

        fire_lat = telemetry.lat + d_lat
        fire_lon = telemetry.lon + d_lon

        # ── Step 6: confidence heuristic ─────────────────────────────────────
        #   Degrades with large tilt (bad attitude) and highly oblique rays.
        max_tilt_deg = max(abs(telemetry.pitch), abs(telemetry.roll))
        obliquity    = ground_range_m / max(1.0, telemetry.alt_agl)
        confidence   = float(
            np.clip(1.0 - max_tilt_deg / 45.0 - obliquity / 10.0, 0.05, 1.0)
        )

        return GeolocResult(
            lat=fire_lat,
            lon=fire_lon,
            confidence=confidence,
            ray_NED=d_NED,
            ground_range_m=ground_range_m,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def centroid_of_polygon(polygon: np.ndarray) -> Tuple[float, float]:
    """
    Pixel centroid of a cv2 contour (shape N×1×2 or N×2).
    Returns (u, v) float tuple.
    """
    pts = polygon.reshape(-1, 2).astype(np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())

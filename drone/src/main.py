"""main.py — Fire GPS Traffic Controller
========================================
Orchestrates the full pipeline:
  drone  →  vision.py (YOLO + Bravonoid)  →  geometry.py (Pinhole)  →  CSV log

Supported sources
-----------------
  python main.py --source video   --video path/to/file.mp4          (offline test)
  python main.py --source mavlink --connect udp:127.0.0.1:14550     (ArduPilot/PX4)
  python main.py --source mavlink --connect /dev/ttyACM0            (serial)
  python main.py --source dji                                        (fill in SDK)

Output
------
  • Annotated live preview window (disable with --no-preview)
  • CSV log at  data/fire_gps_log.csv  (appended across runs)
  • Stdout per-detection line: frame, lat, lon, range, confidence
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from vision import HybridFireTracker
from geometry import (
    CameraIntrinsics,
    DroneTelemetry,
    FireGeolocator,
    GeolocResult,
    centroid_of_polygon,
)

# ── Optional: pymavlink (pip install pymavlink) ────────────────────────────────
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Paths & defaults
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = SRC_DIR.parent
WEIGHTS_PATH = str(BASE_DIR / "data" / "fire_yolov8n.pt")
LOG_PATH     = str(BASE_DIR / "data" / "fire_gps_log.csv")

# ── Camera intrinsics — replace with your calibrated values ───────────────────
DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=980.0,
    fy=980.0,
    cx=960.0,
    cy=540.0,
    cam_pitch_offset_deg=0.0,   # 0° = nadir; positive = camera pitched forward
)


# ──────────────────────────────────────────────────────────────────────────────
# Drone interface hierarchy
# ──────────────────────────────────────────────────────────────────────────────
class DroneInterface(ABC):
    """Any source of (frame, telemetry) pairs."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_frame_and_telemetry(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[DroneTelemetry]]: ...

    @abstractmethod
    def close(self) -> None: ...


# ── MAVLink (ArduPilot / PX4) ─────────────────────────────────────────────────
class MAVLinkInterface(DroneInterface):
    """
    Connects to any MAVLink autopilot.
    Telemetry comes from GLOBAL_POSITION_INT + ATTITUDE messages.
    Video comes from a separate source (USB cam, RTSP, GStreamer) passed via
    --video-src.  On a companion computer the camera is typically /dev/video0.
    """

    def __init__(self, connection_string: str, video_src: str = "0"):
        if not MAVLINK_AVAILABLE:
            raise RuntimeError(
                "pymavlink not installed.  Run:  pip install pymavlink"
            )
        self._conn_str  = connection_string
        self._video_src = video_src
        self._mav       = None
        self._cap: Optional[cv2.VideoCapture] = None
        # Safe initial telemetry (30 m AGL, stationary)
        self._telem = DroneTelemetry(lat=0.0, lon=0.0, alt_agl=30.0,
                                     roll=0.0, pitch=0.0, yaw=0.0)

    def connect(self) -> None:
        print(f"[MAVLink] Connecting to {self._conn_str} …")
        self._mav = mavutil.mavlink_connection(self._conn_str, autoreconnect=True)
        self._mav.wait_heartbeat()
        print(
            f"[MAVLink] Heartbeat — system {self._mav.target_system}, "
            f"component {self._mav.target_component}"
        )
        # Request high-rate telemetry streams from the autopilot.
        self._mav.mav.request_data_stream_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,   # 10 Hz
            1,    # start
        )
        video_src = (
            int(self._video_src)
            if self._video_src.isdigit()
            else self._video_src
        )
        self._cap = cv2.VideoCapture(video_src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._video_src}")
        print(f"[MAVLink] Video source opened: {self._video_src}")

    def _drain_mavlink(self) -> None:
        """Non-blocking: drain available messages and update cached telemetry."""
        while True:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "GLOBAL_POSITION_INT":
                self._telem.lat     = msg.lat / 1e7
                self._telem.lon     = msg.lon / 1e7
                self._telem.alt_agl = msg.relative_alt / 1000.0  # mm → m
            elif t == "ATTITUDE":
                self._telem.roll  = math.degrees(msg.roll)
                self._telem.pitch = math.degrees(msg.pitch)
                self._telem.yaw   = math.degrees(msg.yaw) % 360.0

    def get_frame_and_telemetry(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[DroneTelemetry]]:
        self._drain_mavlink()
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        # Return a shallow copy so the telemetry snapshot is frozen per frame.
        t = self._telem
        return frame, DroneTelemetry(
            lat=t.lat, lon=t.lon, alt_agl=t.alt_agl,
            roll=t.roll, pitch=t.pitch, yaw=t.yaw,
        )

    def close(self) -> None:
        if self._cap:
            self._cap.release()
        if self._mav:
            self._mav.close()


# ── DJI placeholder ────────────────────────────────────────────────────────────
class DJIInterface(DroneInterface):
    """
    Placeholder for DJI integration.

    Implement using one of:
      • DJI Onboard SDK (OSDK) / ROS wrapper
          https://github.com/dji-sdk/Onboard-SDK-ROS
      • DJI Payload SDK (PSDK) for embedded payloads
      • DJI Mobile SDK via a local bridge process
    """

    def connect(self) -> None:
        raise NotImplementedError(
            "DJI integration is hardware-specific.\n"
            "Implement get_frame_and_telemetry() using your DJI SDK."
        )

    def get_frame_and_telemetry(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[DroneTelemetry]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ── Offline video-file mode ────────────────────────────────────────────────────
class VideoFileInterface(DroneInterface):
    """
    Reads frames from a local video file with static mock telemetry.
    Ideal for developing and testing geometry.py against known footage.
    Override mock_telemetry with realistic values for your test video.
    """

    def __init__(
        self,
        video_path: str,
        mock_telemetry: Optional[DroneTelemetry] = None,
    ):
        self._path  = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._telem = mock_telemetry or DroneTelemetry(
            lat=37.9838, lon=23.7275,
            alt_agl=80.0,
            roll=0.0, pitch=0.0, yaw=0.0,
        )

    def connect(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")
        print(f"[VideoFile] Opened: {self._path}")

    def get_frame_and_telemetry(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[DroneTelemetry]]:
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        return frame, self._telem

    def close(self) -> None:
        if self._cap:
            self._cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# CSV event logger
# ──────────────────────────────────────────────────────────────────────────────
class FireEventLogger:
    """Appends one row per fire detection to a CSV file."""

    _FIELDS = [
        "timestamp_utc", "frame_idx",
        "drone_lat", "drone_lon", "drone_alt_agl",
        "drone_roll", "drone_pitch", "drone_yaw",
        "fire_lat", "fire_lon",
        "ground_range_m", "confidence",
        "pixel_u", "pixel_v",
    ]

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new_file = not os.path.isfile(path)
        self._fh  = open(path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._fh, fieldnames=self._FIELDS)
        if new_file:
            self._csv.writeheader()

    def log(
        self,
        frame_idx: int,
        pixel_uv: Tuple[float, float],
        telemetry: DroneTelemetry,
        result: GeolocResult,
    ) -> None:
        self._csv.writerow({
            "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
            "frame_idx":      frame_idx,
            "drone_lat":      round(telemetry.lat,    7),
            "drone_lon":      round(telemetry.lon,    7),
            "drone_alt_agl":  round(telemetry.alt_agl, 2),
            "drone_roll":     round(telemetry.roll,   2),
            "drone_pitch":    round(telemetry.pitch,  2),
            "drone_yaw":      round(telemetry.yaw,    2),
            "fire_lat":       round(result.lat, 7),
            "fire_lon":       round(result.lon, 7),
            "ground_range_m": round(result.ground_range_m, 1),
            "confidence":     round(result.confidence, 3),
            "pixel_u":        round(pixel_uv[0], 1),
            "pixel_v":        round(pixel_uv[1], 1),
        })
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Traffic controller
# ──────────────────────────────────────────────────────────────────────────────
def run(drone: DroneInterface, show_preview: bool = True) -> None:
    """
    Main loop:
      1. Grab frame + telemetry from drone.
      2. Run vision.py YOLO + Bravonoid → fire_polygon.
      3. Compute pixel centroid → geometry.py Pinhole → GPS.
      4. Log result to CSV and stdout.
      5. Overlay GPS text on preview frame.
    """
    tracker    = HybridFireTracker(WEIGHTS_PATH)
    geolocator = FireGeolocator(DEFAULT_INTRINSICS)
    logger     = FireEventLogger(LOG_PATH)

    drone.connect()
    print(f"[Main] GPS detections will be logged to: {LOG_PATH}")

    frame_idx  = 0
    prev_time  = 0.0

    try:
        while True:
            frame, telemetry = drone.get_frame_and_telemetry()
            if frame is None:
                print("[Main] Stream ended — no more frames.")
                break

            # ── Vision ────────────────────────────────────────────────────────
            annotated_frame, fire_polygon = tracker.process_frame(frame)
            frame_idx += 1

            # ── FPS overlay ───────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time > 0 else 0.0
            prev_time = now
            cv2.putText(
                annotated_frame, f"FPS: {int(fps)}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )

            # ── Geometry: pixel → GPS ─────────────────────────────────────────
            if fire_polygon is not None and telemetry is not None:
                pixel_uv = centroid_of_polygon(fire_polygon)
                result   = geolocator.pixel_to_gps(pixel_uv, telemetry)

                if result is not None:
                    logger.log(frame_idx, pixel_uv, telemetry, result)
                    print(
                        f"[{frame_idx:05d}] FIRE  "
                        f"lat={result.lat:.6f}  lon={result.lon:.6f}  "
                        f"range={result.ground_range_m:.1f}m  "
                        f"conf={result.confidence:.2f}"
                    )
                    # Overlay GPS coordinate on the video frame
                    gps_text = f"GPS {result.lat:.5f}, {result.lon:.5f}"
                    cv2.putText(
                        annotated_frame, gps_text,
                        (10, annotated_frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2,
                    )

            # ── Preview ───────────────────────────────────────────────────────
            if show_preview:
                cv2.imshow("Fire Tracker — GPS", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        drone.close()
        logger.close()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"[Main] Done — {frame_idx} frames processed.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fire GPS Traffic Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", choices=["mavlink", "video", "dji"], default="video",
        help="Data source",
    )
    p.add_argument(
        "--connect", default="udp:127.0.0.1:14550",
        help="MAVLink connection string (e.g. udp:127.0.0.1:14550, /dev/ttyACM0)",
    )
    p.add_argument(
        "--video", default=str(BASE_DIR / "data" / "test2.mp4"),
        help="Video file path (--source video)",
    )
    p.add_argument(
        "--video-src", default="0",
        help="Camera device index or RTSP URL when using --source mavlink",
    )
    # Mock telemetry for video-file mode
    p.add_argument("--lat",   type=float, default=37.9838,
                   help="Mock drone latitude  (video mode)")
    p.add_argument("--lon",   type=float, default=23.7275,
                   help="Mock drone longitude (video mode)")
    p.add_argument("--alt",   type=float, default=80.0,
                   help="Mock drone altitude AGL in metres (video mode)")
    p.add_argument("--roll",  type=float, default=0.0,
                   help="Mock drone roll  deg (video mode)")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="Mock drone pitch deg (video mode)")
    p.add_argument("--yaw",   type=float, default=0.0,
                   help="Mock drone yaw   deg (video mode)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable OpenCV preview window")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.source == "mavlink":
        drone: DroneInterface = MAVLinkInterface(
            connection_string=args.connect,
            video_src=args.video_src,
        )
    elif args.source == "dji":
        drone = DJIInterface()
    else:
        mock_telem = DroneTelemetry(
            lat=args.lat, lon=args.lon, alt_agl=args.alt,
            roll=args.roll, pitch=args.pitch, yaw=args.yaw,
        )
        drone = VideoFileInterface(args.video, mock_telemetry=mock_telem)

    run(drone, show_preview=not args.no_preview)

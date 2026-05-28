"""
main.py — Project WILSON single-command launcher
=================================================
Starts all three servers needed for the Sandbox UI:

    Port 8000  — static HTTP server  (sandbox.html)
    Port 8765  — WebSocket sim server (server.py)
    Port 5000  — Mesh API            (mesh_api.py / Flask)

Usage:
    python main.py

Then open:  http://localhost:8000/sandbox.html

Middle-click the map → yellow 3D pin + Delaunay mesh popup.
Right-click  the map → full-screen mesh modal (legacy).
"""

import subprocess
import sys
import os
import io
import time
import webbrowser
import signal
import threading

# Re-wrap stdout/stderr as UTF-8 on Windows so unicode chars in print()
# (arrows, Greek letters, box-drawing) don't crash with charmap errors.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable   # same interpreter that's running this file

# Force UTF-8 stdout/stderr in every child process so Windows charmap never
# causes UnicodeEncodeError on arrow/Greek characters in log output.
_UTF8_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _stream(proc, label, color_code):
    """Forward a subprocess's stdout to our console with a coloured prefix."""
    for line in iter(proc.stdout.readline, b""):
        print(f"\033[{color_code}m[{label}]\033[0m {line.decode(errors='replace').rstrip()}")


def launch():
    procs = []

    # ── 1. Static HTTP server (sandbox.html) ──────────────────────────────────
    http_proc = subprocess.Popen(
        [PY, "-m", "http.server", "8000"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_UTF8_ENV,
    )
    procs.append(http_proc)
    threading.Thread(target=_stream, args=(http_proc, "HTTP :8000", "94"), daemon=True).start()
    print("\033[94m[HTTP :8000]\033[0m  Static server started  ->  http://localhost:8000/sandbox.html")

    # ── 2. WebSocket simulation server ────────────────────────────────────────
    ws_proc = subprocess.Popen(
        [PY, "server.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_UTF8_ENV,
    )
    procs.append(ws_proc)
    threading.Thread(target=_stream, args=(ws_proc, "WS   :8765", "93"), daemon=True).start()
    print("\033[93m[WS   :8765]\033[0m  WebSocket sim server started")

    # ── 3. Mesh API (Flask) ───────────────────────────────────────────────────
    mesh_proc = subprocess.Popen(
        [PY, "mesh_api.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_UTF8_ENV,
    )
    procs.append(mesh_proc)
    threading.Thread(target=_stream, args=(mesh_proc, "MESH :5000", "92"), daemon=True).start()
    print("\033[92m[MESH :5000]\033[0m  Mesh API (Flask) started")

    # ── Brief pause then open browser ─────────────────────────────────────────
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000/sandbox.html")
    print()
    print("\033[1m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print("  \033[1;33m★  WILSON Sandbox is running\033[0m")
    print()
    print("  Browser  ->  http://localhost:8000/sandbox.html")
    print()
    print("  \033[96mCtrl+click\033[0m   the map  →  yellow 3D pin + interactive mesh popup")
    print("  \033[90mRight-click\033[0m  the map  →  full-screen mesh modal")
    print()
    print("  Press  \033[1mCtrl-C\033[0m  to stop all servers.")
    print("\033[1m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print()

    # ── Wait for Ctrl-C, then kill all children ────────────────────────────────
    def _shutdown(sig, frame):
        print("\n\033[90mShutting down…\033[0m")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep main thread alive
    for p in procs:
        p.wait()


if __name__ == "__main__":
    launch()

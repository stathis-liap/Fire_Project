"""
main.py -- Project WILSON single-command launcher
==================================================
Starts all three servers and opens the browser automatically.

    Port 8000  -- static HTTP  (sandbox.html)
    Port 8765  -- WebSocket simulation server (server.py)
    Port 5000  -- Mesh / DEM API  (mesh_api.py)

Usage:
    python main.py

Press Ctrl-C to stop everything cleanly.
"""

import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser

# ── UTF-8 console on Windows ──────────────────────────────────────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.abspath(__file__))
PY       = sys.executable
_ENV     = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
_PORTS   = [8000, 8765, 5000]
_LABELS  = {8000: ("HTTP :8000", "94"), 8765: ("WS   :8765", "93"), 5000: ("MESH :5000", "92")}


# ── Port helpers ──────────────────────────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    """Return True if something is already listening on port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.15)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_port(port: int) -> None:
    """Kill the process listening on port (Windows netstat / taskkill)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if "LISTEN" not in line.upper():
                continue
            parts = line.split()
            # LOCAL ADDRESS is typically 3rd token: 0.0.0.0:8765
            if not any(p.endswith(f":{port}") for p in parts):
                continue
            pid = parts[-1]
            if not pid.isdigit() or int(pid) <= 4:
                continue
            subprocess.run(["taskkill", "/F", "/PID", pid],
                           capture_output=True, timeout=3)
            _log(f"Freed :{port} (PID {pid})", "90")
            return
    except Exception:
        pass


def _wait_port(port: int, want_free: bool, timeout: float = 6.0) -> bool:
    """Block until port is free (want_free=True) or bound (want_free=False)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        in_use = _port_in_use(port)
        if want_free and not in_use:
            return True
        if not want_free and in_use:
            return True
        time.sleep(0.15)
    return False


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str, color: str = "0") -> None:
    print(f"\033[{color}m{msg}\033[0m", flush=True)


def _stream(proc: subprocess.Popen, label: str, color: str) -> None:
    """Pipe a subprocess stdout line-by-line to the console with a coloured tag."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            print(f"\033[{color}m[{label}]\033[0m {line}", flush=True)


# ── Launch ────────────────────────────────────────────────────────────────────

def launch() -> None:
    # ── Step 1: Release stale processes ──────────────────────────────────────
    _log("Checking ports ...", "90")
    for port in _PORTS:
        if _port_in_use(port):
            _kill_port(port)
            _wait_port(port, want_free=True, timeout=3.0)

    # ── Step 2: Start subprocesses ────────────────────────────────────────────
    cmds = {
        8000: [PY, "-m", "http.server", "8000"],
        8765: [PY, "server.py"],
        5000: [PY, "mesh_api.py"],
    }

    procs: list[subprocess.Popen] = []
    for port, cmd in cmds.items():
        label, color = _LABELS[port]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_ENV,
        )
        procs.append(proc)
        threading.Thread(target=_stream, args=(proc, label, color),
                         daemon=True).start()

    # ── Step 3: Wait until each server is actually listening ──────────────────
    _log("Waiting for servers to be ready ...", "90")
    all_ready = True
    for port in _PORTS:
        label, color = _LABELS[port]
        ready = _wait_port(port, want_free=False, timeout=12.0)
        if ready:
            _log(f"  [OK] {label}  ->  http://localhost:{port}", color)
        else:
            _log(f"  [!!] {label} did not start in time", "91")
            all_ready = False

    # ── Step 4: Open browser ──────────────────────────────────────────────────
    url = "http://localhost:8000/sandbox.html"
    webbrowser.open(url)

    print(flush=True)
    print("\033[1m" + "=" * 64 + "\033[0m", flush=True)
    if all_ready:
        _log("  WILSON Sandbox is running", "1;33")
    else:
        _log("  WILSON: some servers did not start — check output above", "91")
    print(flush=True)
    _log(f"  Browser  ->  {url}", "0")
    print(flush=True)
    _log("  Ctrl+click map  -> 3D terrain popup", "96")
    _log("  3D View tab     -> fire simulation on 3D terrain", "96")
    print(flush=True)
    _log("  Press Ctrl-C to stop all servers.", "0")
    print("\033[1m" + "=" * 64 + "\033[0m", flush=True)
    print(flush=True)

    # ── Step 5: Monitor & auto-restart crashed servers ────────────────────────
    restart_counts = {p.pid: 0 for p in procs}

    def _watchdog(idx: int, cmd: list[str], port: int) -> None:
        """Restart a server subprocess if it crashes unexpectedly."""
        label, color = _LABELS[port]
        while True:
            proc = procs[idx]
            proc.wait()                         # blocks until process exits
            if _shutdown_flag:
                return
            restarts = restart_counts.get(proc.pid, 0)
            if restarts >= 3:
                _log(f"[{label}] crashed 3 times — giving up", "91")
                return
            _log(f"[{label}] crashed — restarting (attempt {restarts+1})", "93")
            time.sleep(1.0)
            new_proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_ENV,
            )
            restart_counts[new_proc.pid] = restarts + 1
            procs[idx] = new_proc
            threading.Thread(target=_stream, args=(new_proc, label, color),
                             daemon=True).start()

    for idx, (port, cmd) in enumerate(cmds.items()):
        threading.Thread(target=_watchdog, args=(idx, cmd, port),
                         daemon=True).start()

    # ── Step 6: Ctrl-C handler ────────────────────────────────────────────────
    global _shutdown_flag
    _shutdown_flag = False

    def _shutdown(sig, frame):
        global _shutdown_flag
        _shutdown_flag = True
        _log("\nShutting down ...", "90")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        # Also make sure the ports are freed for next run
        time.sleep(0.3)
        for port in _PORTS:
            if _port_in_use(port):
                _kill_port(port)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _shutdown(None, None)


_shutdown_flag = False

if __name__ == "__main__":
    launch()

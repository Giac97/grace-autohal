"""`grace-al monitor`: follow a campaign live in a browser.

A small read-only web server (standard library only) on the login node. It reads the same
files as `grace-al status` - state.json, done markers, timing.json of walkers and aims tasks,
the job logs - which the running job writes on the shared filesystem, so the job needs no
hooks. It listens on 127.0.0.1: open it through an SSH tunnel.

  GET /             the page (polls /api/status)
  GET /api/status   JSON snapshot
  GET /plot/NAME    runs/report/NAME.png (regenerated when a stage finishes)
  GET /parity/K     runs/round_K/validate/parity.png
"""

from __future__ import annotations

import getpass
import json
import logging
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.state import load_state, round_dir, rounds_on_disk

if TYPE_CHECKING:
    from grace_al.config.schema import Config

log = logging.getLogger("grace_al.monitor")

LOG_LINES = 40


# ---------------------------------------------------------------- data
class PlotCache:
    """The report plots, regenerated only when a done marker changed."""

    def __init__(self, cfg: Config):
        self.cfg, self.key, self.lock = cfg, "", threading.Lock()

    def marker_key(self) -> str:
        ms = sorted(self.cfg.workdir.glob("round_*/.done/*.json"))
        return f"{len(ms)}-{max((m.stat().st_mtime_ns for m in ms), default=0)}"

    def refresh(self) -> str:
        key = self.marker_key()
        with self.lock:
            if key != self.key:
                from grace_al.report import write_report

                try:
                    write_report(self.cfg, html_page=False)
                except Exception as err:      # a half-written file: try again next poll
                    log.warning("report plots: %s", err)
                    return self.key
                self.key = key
        return self.key


def log_tail(cfg: Config, n: int = LOG_LINES) -> dict | None:
    """Last lines of the newest job log (logs/al_<id>.log)."""
    logs = sorted((cfg.home / "logs").glob("al_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    lines = logs[-1].read_text(errors="replace").splitlines()[-n:]
    return {"file": logs[-1].name, "lines": lines,
            "age_s": time.time() - logs[-1].stat().st_mtime}


def snapshot(cfg: Config, plots_key: str = "") -> dict:
    """Everything the page shows, as JSON-ready data."""
    from grace_al.hpc.slurm import sacct
    from grace_al.progress import stage_progress
    from grace_al.report import latest_tau_advice, round_rows

    st = load_state(cfg.workdir)
    progress = None
    if st.status in ("running", "waiting_for_labels", "interrupted") and st.stage:
        progress = stage_progress(cfg, st.round, st.stage)
    jobs = st.jobs[-8:]
    acct = {a["JobID"].split("_")[0]: a for a in sacct([str(j["id"]) for j in jobs])}
    report = cfg.workdir / "report"
    tau = latest_tau_advice(cfg)
    return {
        "campaign": cfg.home.name, "now": time.time(),
        "state": {"round": st.round, "stage": st.stage, "status": st.status,
                  "updated": st.updated, "message": st.message},
        "progress": progress,
        "rounds": round_rows(cfg),
        "tau": {"round": tau[0], **tau[1]} if tau else None,
        "jobs": [{**j, **{k.lower(): v for k, v in acct.get(str(j["id"]), {}).items()}}
                 for j in jobs],
        "log": log_tail(cfg),
        "plots": sorted(p.stem for p in report.glob("*.png")) if report.exists() else [],
        "parity": [k for k in rounds_on_disk(cfg.workdir)
                   if (round_dir(cfg.workdir, k) / "validate" / "parity.png").exists()],
        "plots_key": plots_key,
    }


# ---------------------------------------------------------------- server
def make_handler(cfg: Config, cache: PlotCache, refresh_s: int):
    from importlib import resources

    page = ((resources.files("grace_al") / "monitor.html").read_text()
            .replace("__REFRESH__", str(int(refresh_s))))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):            # quiet; errors still go to the log
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path):
            if path.is_file():
                self._send(200, path.read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_GET(self):                           # noqa: N802 (http.server API)
            url = self.path.split("?", 1)[0]
            if url == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url == "/api/status":
                body = json.dumps(snapshot(cfg, cache.refresh()), default=str)
                self._send(200, body.encode(), "application/json")
            elif m := re.fullmatch(r"/plot/([a-z_]+)\.png", url):
                self._file(cfg.workdir / "report" / f"{m.group(1)}.png")
            elif m := re.fullmatch(r"/parity/(\d+)\.png", url):
                self._file(round_dir(cfg.workdir, int(m.group(1))) / "validate" / "parity.png")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def make_server(cfg: Config, host: str = "127.0.0.1", port: int = 8765,
                refresh_s: int = 30) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(cfg, PlotCache(cfg), refresh_s))


def tunnel_hint(cfg: Config, port: int) -> str:
    user, here = getpass.getuser(), socket.gethostname()
    lines = [f"monitor of {cfg.home.name} on {here}:{port}. On your computer:",
             f"  ssh -L {port}:localhost:{port} {user}@{here}"]
    login = cfg.hpc.login_host if cfg.hpc else None
    if login and login != here:
        lines.append(f"  (if {here} is not reachable directly: ssh -J {user}@{login} "
                     f"-L {port}:localhost:{port} {user}@{here})")
    lines.append(f"then open http://localhost:{port}  (Ctrl-C here stops the monitor)")
    return "\n".join(lines)




def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765, refresh_s: int = 30) -> None:
    server = make_server(cfg, host, port, refresh_s)
    print(tunnel_hint(cfg, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

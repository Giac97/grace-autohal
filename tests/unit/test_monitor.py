import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from grace_al.config.loader import load_config
from grace_al.monitor import make_server, snapshot, tunnel_hint
from grace_al.state import update_state


@pytest.fixture
def server(fake_runs):
    cfg = load_config(fake_runs)
    srv = make_server(cfg, port=0, refresh_s=5)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield cfg, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_snapshot(fake_runs):
    cfg = load_config(fake_runs)
    s = snapshot(cfg)
    assert s["state"]["status"] == "finished" and len(s["rounds"]) == 2
    assert s["jobs"][0]["id"] == "1001" and s["progress"] is None
    json.dumps(s, default=str)


def test_snapshot_progress_and_log(fake_runs):
    cfg = load_config(fake_runs)
    (cfg.home / "logs").mkdir()
    (cfg.home / "logs" / "al_1001.log").write_text("\n".join(f"line {i}" for i in range(100)))
    d = cfg.workdir / "round_02" / "dft" / "0000"
    d.mkdir(parents=True)
    (d / "timing.json").write_text(json.dumps({"start": time.time() - 60, "end": None}))
    (d / "aims.out").write_text("  Invoking FHI-aims ...\n"
                                "  Begin self-consistency iteration #    3\n")
    update_state(cfg.workdir, round=2, stage="label", status="running", message="")
    s = snapshot(cfg)
    assert s["progress"]["running"] == 1 and s["progress"]["what"] == "DFT calculations"
    assert s["log"]["lines"][-1] == "line 99" and len(s["log"]["lines"]) == 40


def test_server_routes(server):
    cfg, base = server
    code, ctype, body = get(base + "/")
    assert code == 200 and "text/html" in ctype and b"__REFRESH__" not in body
    code, ctype, body = get(base + "/api/status")
    d = json.loads(body)
    assert code == 200 and d["campaign"] == cfg.home.name
    assert "exploration" in d["plots"] and d["plots_key"]          # plots made on demand
    code, ctype, body = get(base + "/plot/exploration.png")
    assert code == 200 and ctype == "image/png" and body[:4] == b"\x89PNG"


@pytest.mark.parametrize("path", ["/plot/../config.yaml", "/plot/..%2fconfig.yaml",
                                  "/parity/x.png", "/nope", "/plot/missing.png"])
def test_server_refuses_other_paths(server, path):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + path)
    assert e.value.code == 404


def test_parity_route(server):
    cfg, base = server
    p = cfg.workdir / "round_01" / "validate" / "parity.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG fake")
    assert get(base + "/parity/1.png")[2] == b"\x89PNG fake"
    assert json.loads(get(base + "/api/status")[2])["parity"] == [1]


def test_plots_regenerated_only_when_a_stage_finishes(server):
    cfg, base = server
    k1 = json.loads(get(base + "/api/status")[2])["plots_key"]
    png = cfg.workdir / "report" / "exploration.png"
    t1 = png.stat().st_mtime_ns
    assert json.loads(get(base + "/api/status")[2])["plots_key"] == k1
    assert png.stat().st_mtime_ns == t1
    m = cfg.workdir / "round_01" / ".done" / "select.json"
    m.write_text(m.read_text())
    assert json.loads(get(base + "/api/status")[2])["plots_key"] != k1


def test_tunnel_hint(fake_runs):
    cfg = load_config(fake_runs)
    h = tunnel_hint(cfg, 8765)
    assert "ssh -L 8765:localhost:8765" in h and "http://localhost:8765" in h

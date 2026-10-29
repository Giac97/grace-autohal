import json
import os
import time

import pytest
from typer.testing import CliRunner

from grace_al import progress
from grace_al.cli import app
from grace_al.config.loader import load_config
from grace_al.state import update_state

NICE = "...\n          | Total time                                  :  10.0 s   {w} s\n" \
       "          Have a nice day.\n"


def task(rdir, i, state, now, secs=None, scf=0):
    d = rdir / "dft" / f"{i:04d}"
    d.mkdir(parents=True)
    if state == "pending":
        return
    t = {"start": now - (secs or 600), "end": None}
    if state == "done":
        (d / "aims.out").write_text(NICE.format(w=secs))
        t["end"] = now
    else:
        body = "  Invoking FHI-aims ...\n" + "".join(
            f"  Begin self-consistency iteration #{n:5d}\n" for n in range(1, scf + 1))
        (d / "aims.out").write_text(body)
        if state == "failed":
            old = now - 2 * progress.STALE
            os.utime(d / "aims.out", (old, old))
    (d / "timing.json").write_text(json.dumps(t))


@pytest.fixture
def dft_round(fake_runs):
    cfg = load_config(fake_runs)
    rdir = cfg.workdir / "round_02"
    now = time.time()
    for i, (state, secs) in enumerate([("done", 1200), ("done", 1800), ("running", 900),
                                       ("running", 300), ("failed", None)] + [("pending", 0)] * 5):
        task(rdir, i, state, now, secs, scf=37 if state == "running" else 4)
    return cfg, rdir


def test_label_progress_counts_and_eta(dft_round):
    cfg, rdir = dft_round
    items = progress.label_items(rdir)
    s = progress.summarize(items, parallel=2)
    assert (s["done"], s["running"], s["failed"], s["pending"]) == (2, 2, 1, 5)
    assert s["mean_s"] == 1500 and s["min_s"] == 1200 and s["max_s"] == 1800
    # 5 pending / 2 at a time = 3 waves x 1500 s, + what the running ones still need
    assert s["eta_s"] == pytest.approx(3 * 1500 + ((1500 - 900) + (1500 - 300)) / 2, rel=0.01)
    assert {r["detail"] for r in s["running_items"]} == {"SCF iteration 37"}


def test_aims_wall_time_without_timing(tmp_path):
    assert progress._aims_wall(NICE.format(w=42.5)) == 42.5


def test_status_shows_dft_progress(dft_round):
    cfg, _ = dft_round
    update_state(cfg.workdir, round=2, stage="label", status="waiting_for_labels", message="")
    cfg_path = cfg.home / "config.yaml"
    r = CliRunner().invoke(app, ["status", str(cfg_path)])
    assert r.exit_code == 0, r.output
    assert "2 done, 2 running, 1 failed, 5 pending of 10" in r.output
    assert "25 min on average" in r.output and "ETA" in r.output
    assert "SCF iteration 37" in r.output


def test_fit_progress(tmp_path):
    fit = tmp_path / "fit"
    (fit / "seed" / "1").mkdir(parents=True)
    (fit / "input.yaml").write_text("x")
    old = time.time() - 100
    os.utime(fit / "input.yaml", (old, old))
    (fit / "seed" / "1" / "log.txt").write_text(
        "maxiter resolved from target_total_updates=200: 40 epochs\n")
    (fit / "seed" / "1" / "train_metrics.yaml").write_text("- {a: 1}\n- {a: 2}\n- {a: 3}\n"
                                                           "- {a: 4}\n")
    p = progress.fit_progress(fit)
    assert (p["epochs_done"], p["epochs_total"]) == (4, 40)
    assert p["eta_s"] == pytest.approx(36 * 25, rel=0.05)


def test_fmt_duration():
    assert progress.fmt_duration(42) == "42 s"
    assert progress.fmt_duration(1500) == "25 min"
    assert progress.fmt_duration(3 * 3600 + 600) == "3 h 10 min"

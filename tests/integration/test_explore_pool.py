"""Seeds + artifact + explore with the pool backend on the CPU: Pd(111)+O, 4 walkers,
2 at a time, 200 steps. Includes a SIGTERM in the middle of a real run (~10 min).

Run with:  pytest -m grace tests/integration/test_explore_pool.py
"""

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml

from grace_al import _tp, state, workflow
from grace_al.config.loader import load_config
from grace_al.data import read_frames
from grace_al.explore.plan import load_plan
from grace_al.stages.base import get_stage, run_stage

pytestmark = pytest.mark.grace

FM = "GRACE-2L-SMAX-OMAT-medium"
CACHE = os.environ.get("GRACE_AL_TEST_CACHE") or os.path.expanduser("~/.cache/grace")
CONFIG = {
    "schema_version": 1, "foundation_model": FM,
    "system": {"generator": "surface",
               "surface": {"element": "Pd", "facet": "fcc111", "size": [3, 3, 4],
                           "vacuum": 8.0},
               "adsorbate": {"species": "O"},
               "initial": {"strains": [1.0], "coverages": [2], "n_per_coverage": 2,
                           "relax_steps": 10}},
    "explore": {"ensembles": ["nvt"], "temperatures": [800], "tau": [0.25], "n_steps": 200,
                "walkers_per_setting": 4, "backend": "pool", "walkers_per_gpu": 2},
    "label": {"labeler": "teacher", "teacher_model": FM},
}


@pytest.fixture(autouse=True)
def _real_cache(monkeypatch):
    monkeypatch.setenv("GRACE_CACHE", CACHE)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    os.environ["GRACE_CACHE"] = CACHE
    if not (_tp.checkpoint_cached(FM) and _tp.savedmodel_cached(FM)):
        pytest.skip(f"{FM} not in {CACHE}")
    if not shutil.which("grace_uq"):
        pytest.skip("grace_uq not on PATH")
    home = tmp_path_factory.mktemp("explore_pool")
    (home / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    cfg = load_config(home / "config.yaml")
    workflow.run_seeds(cfg)
    ctx = workflow.context(cfg, 0, workflow.system_info(cfg))
    run_stage(get_stage("artifact"), ctx)
    ctx.driver = dataclasses.replace(ctx.driver,
                                     artifact=ctx.rdir / "UQ" / "gmm_artifacts.npz")
    return home, cfg, ctx


def summaries(ctx):
    return sorted((ctx.rdir / "hal").glob("w*/hal_summary.json"))


def test_sigterm_then_resume(campaign):
    home, cfg, ctx = campaign
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen([sys.executable, "-m", "grace_al", "run", str(home / "config.yaml"),
                             "--round", "0"], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    t0 = time.time()
    while not summaries(ctx) and time.time() - t0 < 900 and proc.poll() is None:
        time.sleep(1)
    assert summaries(ctx), "no walker finished"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(60) != 0
    done_before = {p: p.stat().st_mtime_ns for p in summaries(ctx)}
    assert 1 <= len(done_before) < 4
    assert state.load_state(cfg.workdir).status == "failed"

    res = run_stage(get_stage("explore"), ctx)
    assert res.status == "done"
    assert len(summaries(ctx)) == 4
    for p, t in done_before.items():                      # finished walkers not rerun
        assert p.stat().st_mtime_ns == t


def test_plan_and_outputs(campaign):
    _, _, ctx = campaign
    plan = load_plan(ctx.rdir / "hal" / "plan.json")
    starts = read_frames(ctx.rdir / "starts.extxyz")
    idx = [w.start_index for w in plan]
    assert len(set(idx)) == 4
    assert not all(starts[i].info["config"].startswith("clean") for i in idx)
    assert sum(starts[i].info["config"].startswith("clean") for i in idx) <= 0.2 * 4
    assert (ctx.rdir / "hal" / "walkers.txt").exists()
    for w in plan:
        assert (ctx.rdir / "hal" / w.dir / "walker.log").exists()
    m = state.read_marker(ctx.rdir, "explore")
    assert m["summary"]["n_walkers"] == 4


def test_rerun_is_noop(campaign):
    _, _, ctx = campaign
    stamps = {p: p.stat().st_mtime_ns for p in summaries(ctx)}
    assert run_stage(get_stage("explore"), ctx).status == "skipped"
    from grace_al.stages.explore import ExploreStage

    ExploreStage().run(ctx)                               # even without the marker
    assert {p: p.stat().st_mtime_ns for p in summaries(ctx)} == stamps
    json.loads((ctx.rdir / "hal" / "plan.json").read_text())

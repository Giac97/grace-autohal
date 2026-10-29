import json

import pytest
from ase.build import fcc111
from ase.io import write

from grace_al import state
from grace_al.config.schema import Config
from grace_al.explore.plan import load_plan
from grace_al.models import ModelRef
from grace_al.stages import explore as explore_mod
from grace_al.stages.base import RoundContext
from grace_al.stages.explore import ExploreStage
from grace_al.systems.base import SystemInfo


class FakeBackend:
    """Writes hal_summary.json (+ features) like a walker; outcome by walker id."""

    def __init__(self, outcome=None):
        self.outcome, self.calls = outcome or {}, []

    def run(self, walkers, starts_file, hal, model):
        failures = {}
        for w in walkers:
            out = hal / w.dir
            if (out / "hal_summary.json").exists():
                continue
            self.calls.append((hal.name, w.id, w.tau))
            o = self.outcome.get(w.id, {"n_collected": 3, "rejected": {"gamma": 1}})
            if o == "fail":
                failures[w.id] = {"exit_code": 1, "log_tail": "Traceback ..."}
                continue
            out.mkdir(parents=True, exist_ok=True)
            (out / "hal_summary.json").write_text(json.dumps({"stop_reason": "max_steps",
                                                              **o}))
            if o["n_collected"]:
                (out / "hal_features.npz").write_bytes(b"")
        return failures


@pytest.fixture
def make_ctx(tmp_path, monkeypatch):
    seeds = []
    for name in ["clean_strain1", "clean_strain1_rattle0", "O_fcc", "O_hcp", "Ox2_0",
                 "Ox2_1"]:
        at = fcc111("Pd", (2, 2, 3), vacuum=8, periodic=True)
        at.info["config"] = name
        seeds.append(at)
    write(tmp_path / "seeds.extxyz", seeds)

    def make(backend, k=0, **explore):
        monkeypatch.setattr(explore_mod, "make_backend", lambda cfg: backend)
        cfg = Config(home=tmp_path, label={"labeler": "manual"},
                     explore={"temperatures": [800], "walkers_per_setting": 4, **explore})
        rdir = state.round_dir(cfg.workdir, k)
        rdir.mkdir(parents=True, exist_ok=True)
        return RoundContext(cfg, k, cfg.workdir, rdir, SystemInfo("slab", ("O", "Pd"), 2),
                            ModelRef("fm", tmp_path / "m", tmp_path / "c", tmp_path / "a.npz"))
    return make


def test_summary_and_walkers_txt(make_ctx):
    b = FakeBackend({1: {"n_collected": 0, "rejected": {}}})
    ctx = make_ctx(b)
    res = ExploreStage().run(ctx)
    s = res.summary
    assert res.status == "done"
    assert s["n_walkers"] == 4 and s["collected"] == 9 and s["rejected"] == {"gamma": 3}
    assert s["zero_collected_fraction"] == 0.25
    assert s["reject_ratio"] == pytest.approx(3 / 12)
    assert (ctx.rdir / "hal" / "walkers.txt").read_text().split() == ["w000", "w002", "w003"]
    assert s["warnings"] == []


def test_plan_loaded_not_rebuilt(make_ctx):
    b = FakeBackend({2: "fail"})
    ctx = make_ctx(b)
    res = ExploreStage().run(ctx)
    assert res.summary["n_failed"] == 1 and res.status == "done"
    plan = load_plan(ctx.rdir / "hal" / "plan.json")
    # another config (more walkers) would build another plan; the saved one wins
    ctx2 = make_ctx(FakeBackend(), walkers_per_setting=6)
    ExploreStage().run(ctx2)
    assert load_plan(ctx.rdir / "hal" / "plan.json") == plan
    assert len(plan) == 4


def test_rerun_completes_only_missing(make_ctx):
    b = FakeBackend({2: "fail", 3: "fail"})
    ExploreStage().run(make_ctx(b))
    b2 = FakeBackend()
    ExploreStage().run(make_ctx(b2))
    assert [c[1] for c in b2.calls] == [2, 3]


def test_warnings(make_ctx):
    zero = {i: {"n_collected": 0, "rejected": {}} for i in range(4)}
    s = ExploreStage().run(make_ctx(FakeBackend(zero))).summary
    assert any("tau too small" in w for w in s["warnings"])


def test_too_many_rejected(make_ctx):
    many = {i: {"n_collected": 1, "rejected": {"short_bond": 5}} for i in range(4)}
    s = ExploreStage().run(make_ctx(FakeBackend(many))).summary
    assert any("tau too large" in w and "short_bond" in w for w in s["warnings"])


def test_all_failed_is_a_failure(make_ctx):
    res = ExploreStage().run(make_ctx(FakeBackend(dict.fromkeys(range(4), "fail"))))
    assert res.status == "failed" and "Traceback" in res.summary["error"]


def test_slab_npt_fails_cleanly(make_ctx):
    res = ExploreStage().run(make_ctx(FakeBackend(), ensembles=["npt"]))
    assert res.status == "failed" and "allow_cell_change_on_slab" in res.summary["error"]


def test_tau_calibration(make_ctx):
    # tau 0.5 rejects too much, 0.25 is the largest good one, 0.1 collects nothing
    class CalBackend(FakeBackend):
        def run(self, walkers, starts_file, hal, model):
            for w in walkers:
                self.outcome[w.id] = (
                    {"n_collected": 0, "rejected": {}} if w.tau == 0.1 else
                    {"n_collected": 2, "rejected": {"gamma": 9}} if w.tau == 0.5 else
                    {"n_collected": 4, "rejected": {"gamma": 1}})
            return super().run(walkers, starts_file, hal, model)

    b = CalBackend()
    ctx = make_ctx(b, calibrate=True, calibrate_taus=[0.1, 0.25, 0.5], calibrate_starts=2)
    ExploreStage().run(ctx)
    cal = json.loads((ctx.rdir / "hal" / "calibration.json").read_text())
    assert cal["tau"] == 0.25
    assert {c[2] for c in b.calls if c[0] == "calibrate"} == {0.1, 0.25, 0.5}
    assert {w.tau for w in load_plan(ctx.rdir / "hal" / "plan.json")} == {0.25}
    # round 1 reuses it without calibrating
    b1 = CalBackend()
    ExploreStage().run(make_ctx(b1, k=1, calibrate=True, calibrate_taus=[0.1, 0.25, 0.5]))
    assert all(c[0] == "hal" and c[2] == 0.25 for c in b1.calls)

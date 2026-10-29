import json
import os

import pytest

from grace_al import state
from grace_al.config.schema import Config
from grace_al.models import ModelRef
from grace_al.stages import base
from grace_al.stages.base import (
    FingerprintMismatch,
    RoundContext,
    StageFailed,
    StageResult,
    run_stage,
)
from grace_al.systems.base import SystemInfo


class FakeStage:
    def __init__(self, name="artifact", depends_on=("artifact",), result="done", applies=True):
        self.name, self.depends_on = name, depends_on
        self.result, self._applies, self.calls = result, applies, 0

    def applies(self, ctx):
        return self._applies

    def run(self, ctx):
        self.calls += 1
        return StageResult(self.result, {"n": self.calls})


def ctx_for(tmp_path, k=0, **cfg):
    c = Config(home=tmp_path, label={"labeler": "manual"}, **cfg)
    rdir = state.round_dir(c.workdir, k)
    rdir.mkdir(parents=True, exist_ok=True)
    return RoundContext(c, k, c.workdir, rdir, SystemInfo("slab", ("O", "Pd"), 2),
                        ModelRef("fm", tmp_path / "m.yaml", tmp_path / "ckpt"))


# ---------------------------------------------------------------- state.json
def test_state_round_trip_and_atomic_write(tmp_path, monkeypatch):
    st = state.save_state(tmp_path, state.State(round=2, stage="label",
                                                status="waiting_for_labels"))
    assert state.load_state(tmp_path) == st
    assert st.updated
    # a crash while writing leaves the old file intact and no temporary files behind
    def boom(*a):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        state.update_state(tmp_path, round=3)
    assert state.load_state(tmp_path).round == 2
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_missing_state_is_fresh(tmp_path):
    assert state.load_state(tmp_path) == state.State()


# ---------------------------------------------------------------- runner
def test_marker_written_then_skipped(tmp_path):
    ctx, stage = ctx_for(tmp_path), FakeStage()
    assert run_stage(stage, ctx).status == "done"
    m = state.read_marker(ctx.rdir, "artifact")
    assert m["fingerprint"] == base.fingerprint(stage, ctx)
    assert m["summary"] == {"n": 1} and m["result"] == "done"
    assert set(m["versions"]) >= {"grace_al", "tensorpotential"}
    res = run_stage(stage, ctx)
    assert (res.status, res.summary, stage.calls) == ("skipped", {"n": 1}, 1)


def test_fingerprint_mismatch_names_redo(tmp_path):
    run_stage(FakeStage(), ctx_for(tmp_path))
    ctx = ctx_for(tmp_path, artifact={"rp_dim": 64})
    with pytest.raises(FingerprintMismatch, match="grace-al redo --round 0 --from artifact"):
        run_stage(FakeStage(), ctx)


def test_unrelated_section_does_not_invalidate(tmp_path):
    run_stage(FakeStage(), ctx_for(tmp_path))
    stage = FakeStage()
    assert run_stage(stage, ctx_for(tmp_path, select={"n_random": 9})).status == "skipped"
    assert stage.calls == 0


def test_no_dependencies_never_mismatch(tmp_path):
    run_stage(FakeStage("validate", ()), ctx_for(tmp_path))
    assert run_stage(FakeStage("validate", ()), ctx_for(tmp_path, select={"n_random": 9})) \
        .status == "skipped"


def test_not_applicable_and_waiting_leave_no_marker(tmp_path):
    ctx = ctx_for(tmp_path)
    assert run_stage(FakeStage(applies=False), ctx).status == "skipped"
    assert run_stage(FakeStage("label", result="waiting"), ctx).status == "waiting"
    assert not (ctx.rdir / ".done").exists()


def test_converged_is_reported_again_when_skipped(tmp_path):
    ctx = ctx_for(tmp_path)
    run_stage(FakeStage("select", result="converged"), ctx)
    assert run_stage(FakeStage("select", result="done"), ctx).status == "converged"


def test_failed_raises(tmp_path):
    stage = FakeStage("label")
    stage.run = lambda ctx: StageResult("failed", {"error": "3/5 labels failed"})
    with pytest.raises(StageFailed, match="3/5 labels failed"):
        run_stage(stage, ctx_for(tmp_path))
    assert state.read_marker(ctx_for(tmp_path).rdir, "label") is None


def test_versions_from_summary_go_to_marker(tmp_path):
    stage, ctx = FakeStage("label"), ctx_for(tmp_path)
    stage.run = lambda ctx: StageResult("done", {"n": 1, "versions": {"aims": "240920"}})
    run_stage(stage, ctx)
    m = state.read_marker(ctx.rdir, "label")
    assert m["versions"]["aims"] == "240920" and "versions" not in m["summary"]


# ---------------------------------------------------------------- redo
def populate(root):
    for k in (0, 1, 2):
        r = state.round_dir(root, k)
        for s in state.STAGES:
            state.write_marker(r, s, {"stage": s})
        (r / "UQ").mkdir(parents=True)
        (r / "UQ" / "gmm_artifacts.npz").write_text("x")
        for f in ("selected.extxyz", "labeled.pkl.gz", "validation.json"):
            (r / f).write_text("x")
        (r / "fit").mkdir()
    (root / "FINISHED").write_text("finished\n")
    state.save_state(root, state.State(round=2, status="finished"))


def test_redo_removes_stage_and_later(tmp_path):
    populate(tmp_path)
    deleted = state.redo(tmp_path, 1, "label", yes=True)
    r0, r1 = state.round_dir(tmp_path, 0), state.round_dir(tmp_path, 1)
    assert r1 / "labeled.pkl.gz" in deleted
    # earlier stages of round 1 and all of round 0 stay
    assert (r1 / "selected.extxyz").exists() and (r1 / "UQ").exists()
    assert state.read_marker(r1, "select") is not None
    for s in ("label", "validate", "finetune"):
        assert state.read_marker(r1, s) is None
    assert not (r1 / "fit").exists() and not (r1 / "validation.json").exists()
    assert (r0 / "fit").exists() and state.read_marker(r0, "finetune") is not None
    # later rounds and FINISHED go
    assert not state.round_dir(tmp_path, 2).exists()
    assert not (tmp_path / "FINISHED").exists()
    st = state.load_state(tmp_path)
    assert (st.round, st.stage, st.status) == (1, "label", "running")


def test_redo_asks(tmp_path):
    populate(tmp_path)
    seen = []
    assert state.redo(tmp_path, 0, "finetune", confirm=lambda p: seen.extend(p) or False) == []
    assert seen and (state.round_dir(tmp_path, 0) / "fit").exists()


def test_redo_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        state.redo_targets(tmp_path, 0, "nope")


def test_every_stage_has_an_output_entry():
    assert set(state.STAGE_OUTPUTS) == set(state.STAGES)
    assert set(base.STAGE_CLASSES) == set(state.STAGES)


def test_marker_json_is_readable(tmp_path):
    ctx = ctx_for(tmp_path)
    run_stage(FakeStage(), ctx)
    json.loads(state.marker_path(ctx.rdir, "artifact").read_text())

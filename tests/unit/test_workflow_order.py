import importlib

import pytest
from ase.build import fcc111
from ase.io import write

from grace_al import state, workflow
from grace_al.config.schema import Config
from grace_al.stages import base
from grace_al.stages.base import StageFailed, StageResult


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Config with a seeds file; every stage keeps its real `applies` but runs a fake
    that records (round, stage, driver kind, driver artifact)."""
    write(tmp_path / "seeds.extxyz", [fcc111("Pd", (2, 2, 3), vacuum=8)])
    cfg = Config(home=tmp_path, label={"labeler": "manual"}, convergence={"max_rounds": 2})
    calls, outcome = [], {}

    def fake_get_stage(name):
        mod, cls = base.STAGE_CLASSES[name].split(":")
        real = getattr(importlib.import_module(mod), cls)

        def run(self, ctx):
            calls.append((ctx.k, name, ctx.driver.kind, ctx.driver.artifact))
            if name == "finetune":
                (ctx.rdir / "fit").mkdir(exist_ok=True)
                (ctx.rdir / "fit" / "FINISHED").touch()
            return StageResult(outcome.get((ctx.k, name), "done"))

        return type(cls, (real,), {"run": run})()

    monkeypatch.setattr(workflow, "get_stage", fake_get_stage)
    monkeypatch.setattr(workflow, "ensure_fm", lambda cfg: None)
    return cfg, calls, outcome


def test_stage_order_over_two_rounds(campaign):
    cfg, calls, _ = campaign
    assert workflow.run(cfg) == "finished"
    assert [(k, s) for k, s, *_ in calls] == [
        (0, "seeds"), (0, "artifact"), (0, "explore"), (0, "select"), (0, "label"),
        (0, "validate"), (0, "finetune"),
        (1, "artifact"), (1, "explore"), (1, "select"), (1, "label"), (1, "validate"),
        (1, "finetune")]
    assert (cfg.workdir / "FINISHED").read_text().startswith("finished")
    st = state.load_state(cfg.workdir)
    assert (st.round, st.status) == (1, "finished")


def test_driver_per_round_and_artifact_after_artifact_stage(campaign):
    cfg, calls, _ = campaign
    workflow.run(cfg)
    drivers = {(k, s): (kind, art) for k, s, kind, art in calls}
    assert drivers[(0, "artifact")] == ("fm", None)
    assert drivers[(0, "explore")] == ("fm", cfg.workdir / "round_00/UQ/gmm_artifacts.npz")
    assert drivers[(1, "artifact")][0] == "finetuned"
    assert drivers[(1, "validate")] == ("finetuned",
                                        cfg.workdir / "round_01/UQ/gmm_artifacts.npz")


def test_second_run_does_nothing(campaign):
    cfg, calls, _ = campaign
    workflow.run(cfg)
    n = len(calls)
    assert workflow.run(cfg) == "finished"
    assert len(calls) == n


def test_waiting_for_labels_then_resume(campaign):
    cfg, calls, outcome = campaign
    outcome[(1, "label")] = "waiting"
    assert workflow.run(cfg) == "waiting_for_labels"
    assert calls[-1][:2] == (1, "label")
    st = state.load_state(cfg.workdir)
    assert (st.round, st.stage, st.status) == (1, "label", "waiting_for_labels")
    assert workflow.EXIT_CODES[st.status] == 3
    assert not (cfg.workdir / "FINISHED").exists()
    del outcome[(1, "label")]
    assert workflow.run(cfg) == "finished"
    assert [(k, s) for k, s, *_ in calls[-3:]] == [(1, "label"), (1, "validate"),
                                                   (1, "finetune")]


def test_converged_stops_the_loop(campaign):
    cfg, calls, outcome = campaign
    cfg.convergence.max_rounds = 5
    outcome[(1, "select")] = "converged"
    assert workflow.run(cfg) == "converged"
    assert calls[-1][:2] == (1, "select")
    assert (cfg.workdir / "FINISHED").read_text().startswith("converged")
    assert workflow.run(cfg) == "converged"            # stays converged


def test_only_round(campaign):
    cfg, calls, _ = campaign
    assert workflow.run(cfg, only_round=0) == "running"
    assert {k for k, *_ in calls} == {0}
    assert not (cfg.workdir / "FINISHED").exists()


def test_round_without_previous_fit(campaign):
    cfg, _, _ = campaign
    with pytest.raises(Exception, match="has no FINISHED"):
        workflow.run(cfg, only_round=1)
    assert state.load_state(cfg.workdir).status == "failed"


def test_failure_recorded(campaign, monkeypatch):
    cfg, _, outcome = campaign
    outcome[(0, "explore")] = "failed"
    with pytest.raises(StageFailed):
        workflow.run(cfg)
    st = state.load_state(cfg.workdir)
    assert (st.round, st.stage, st.status) == (0, "explore", "failed")


def test_round0_validation_can_be_switched_off(campaign):
    cfg, calls, _ = campaign
    cfg.validation.include_round0 = False
    workflow.run(cfg)
    assert (0, "validate") not in [(k, s) for k, s, *_ in calls]
    assert (1, "validate") in [(k, s) for k, s, *_ in calls]

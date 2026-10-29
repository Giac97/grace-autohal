import json
import threading

import numpy as np
import pandas as pd
import pytest
from ase.build import fcc111
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from grace_al import state
from grace_al.config.schema import Config
from grace_al.labelers.base import LabelFailure, LabelTask, task_dirs
from grace_al.labelers.manual import ManualLabeler
from grace_al.models import ModelRef
from grace_al.stages import label as label_mod
from grace_al.stages.base import (
    FingerprintMismatch,
    RoundContext,
    StageFailed,
    fingerprint,
    run_stage,
)
from grace_al.stages.label import LabelStage
from grace_al.systems.base import SystemInfo


class FakeLabeler:
    """Energy = -index; tasks listed in `fail` fail with charge_integration."""
    name = "fake"

    def __init__(self, fail=()):
        self.fail, self.runs, self.lock = set(fail), [], threading.Lock()
        self.active = self.max_active = 0

    def prepare(self, frames, workdir, system):
        return [LabelTask(i, d, at) for i, (at, d) in
                enumerate(zip(frames, task_dirs(workdir, len(frames)), strict=True))]

    def run(self, task, slot=None):
        import time

        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
            self.runs.append((task.index, slot))
        task.workdir.mkdir(parents=True, exist_ok=True)
        (task.workdir / "done").write_text("")

    def started(self, task):
        return (task.workdir / "done").exists()

    def is_finished(self, task):
        return (task.workdir / "done").exists() and task.index not in self.fail

    def collect(self, task):
        if task.index in self.fail:
            return LabelFailure(task.index, "charge_integration", "Charge integration error")
        at = task.atoms.copy()
        at.calc = SinglePointCalculator(at, energy=-float(task.index),
                                        forces=np.zeros((len(at), 3)))
        return at


@pytest.fixture
def ctx_factory(tmp_path, monkeypatch):
    sel = [fcc111("Pd", (2, 2, 3), vacuum=8) for _ in range(4)]
    for i, at in enumerate(sel):
        at.rattle(0.01, seed=i)
        at.info["strategy"] = "calm-greedy" if i < 3 else "random-all"
        at.info["gamma"] = 5.0
    write(tmp_path / "seeds.extxyz", [fcc111("Pd", (2, 2, 3), vacuum=8)])

    def make(k=1, fake=None, **label):
        cfg = Config(home=tmp_path, label={"labeler": "manual", **label},
                     aims={"species_dir": "species", "kspacing": 0.2}, hpc={})
        rdir = state.round_dir(cfg.workdir, k)
        rdir.mkdir(parents=True, exist_ok=True)
        write(rdir / "selected.extxyz", sel)
        if fake is not None:
            monkeypatch.setattr(label_mod, "get_labeler", lambda c: fake)
        return RoundContext(cfg, k, cfg.workdir, rdir, SystemInfo("slab", ("Pd",), 2),
                            ModelRef("fm", tmp_path / "m", tmp_path / "c"))
    return make


def test_inline_order_and_output(ctx_factory):
    fake = FakeLabeler()
    ctx = ctx_factory(fake=fake)
    res = LabelStage().run(ctx)
    assert res.status == "done" and res.summary["n_labeled"] == 4
    assert [i for i, _ in fake.runs] == [0, 1, 2, 3] and fake.max_active == 1
    df = pd.read_pickle(ctx.rdir / "labeled.pkl.gz")
    assert list(df["strategy"]) == ["calm-greedy"] * 3 + ["random-all"]
    assert list(df["label_index"]) == [0, 1, 2, 3]
    assert list(df["energy"]) == [0.0, -1.0, -2.0, -3.0]
    assert json.loads((ctx.rdir / "failures.json").read_text())["failures"] == []


def test_round0_adds_seeds(ctx_factory):
    ctx = ctx_factory(k=0, fake=FakeLabeler())
    LabelStage().run(ctx)
    df = pd.read_pickle(ctx.rdir / "labeled.pkl.gz")
    assert list(df["strategy"]) == ["calm-greedy"] * 3 + ["random-all", "seed"]


def test_second_run_does_not_recompute(ctx_factory):
    fake = FakeLabeler()
    ctx = ctx_factory(fake=fake)
    LabelStage().run(ctx)
    LabelStage().run(ctx)
    assert len(fake.runs) == 4


def test_concurrent_slot_rank_split(ctx_factory, monkeypatch):
    monkeypatch.setenv("AIMS_NP", "8")
    fake = FakeLabeler()
    ctx = ctx_factory(fake=fake, mode="concurrent", concurrent_calcs=2)
    assert LabelStage().run(ctx).status == "done"
    assert fake.max_active == 2 and len(fake.runs) == 4
    slots = {s.env["I_MPI_PIN_PROCESSOR_LIST"]: s.ranks for _, s in fake.runs}
    assert slots == {"0-3": 4, "4-7": 4}


def test_array_waits_then_collects(ctx_factory):
    fake = FakeLabeler()
    ctx = ctx_factory(fake=fake, mode="array")
    res = LabelStage().run(ctx)
    assert res.status == "waiting" and res.summary["n_pending"] == 4
    listed = (ctx.rdir / "dft" / "tasks.txt").read_text().split()
    assert listed == [str(ctx.rdir / "dft" / f"{i:04d}") for i in range(4)]
    assert fake.runs == []
    for t in fake.prepare(label_mod.frames_to_label(ctx), ctx.rdir / "dft", None):
        fake.run(t)                                     # what the job array does
    assert LabelStage().run(ctx).status == "done"
    assert len(fake.runs) == 4                          # nothing run by the stage itself


def test_array_failed_tasks_are_collected_not_relisted(ctx_factory):
    fake = FakeLabeler(fail={2})
    ctx = ctx_factory(fake=fake, mode="array", max_failure_fraction=0.5)
    LabelStage().run(ctx)
    for t in fake.prepare(label_mod.frames_to_label(ctx), ctx.rdir / "dft", None):
        fake.run(t)
    res = LabelStage().run(ctx)
    assert res.status == "done" and res.summary["failures"] == {"charge_integration": 1}


def test_failure_threshold(ctx_factory):
    ctx = ctx_factory(fake=FakeLabeler(fail={0, 1}), max_failure_fraction=0.2)
    res = LabelStage().run(ctx)
    assert res.status == "failed"
    assert "2/4 labels failed" in res.summary["error"]
    assert "charge_integration" in res.summary["error"] and "pdgemm" in res.summary["error"]
    assert "grace-al redo --round 1 --from label" in res.summary["error"]
    assert not (ctx.rdir / "labeled.pkl.gz").exists()
    assert json.loads((ctx.rdir / "failures.json").read_text())["counts"] == \
        {"charge_integration": 2}
    with pytest.raises(StageFailed):
        run_stage(LabelStage(), ctx)


def test_fingerprint_includes_aims_not_mode(ctx_factory):
    stage = LabelStage()
    base = fingerprint(stage, ctx_factory())
    assert fingerprint(stage, ctx_factory(mode="array")) == base
    assert fingerprint(stage, ctx_factory(max_failure_fraction=0.5)) == base
    ctx = ctx_factory()
    ctx.cfg.aims.kspacing = 0.15
    assert fingerprint(stage, ctx) != base
    assert fingerprint(stage, ctx_factory(labeler="teacher", teacher_model="X")) != base


def test_changed_aims_refused_after_done(ctx_factory):
    ctx = ctx_factory(fake=FakeLabeler())
    run_stage(LabelStage(), ctx)
    ctx.cfg.aims.kspacing = 0.15
    with pytest.raises(FingerprintMismatch, match="--from label"):
        run_stage(LabelStage(), ctx)


# ---------------------------------------------------------------- manual labeler
def test_manual_waits_then_imports(ctx_factory):
    ctx = ctx_factory()                                 # labeler: manual (the real one)
    res = LabelStage().run(ctx)
    assert res.status == "waiting"
    assert "4 structures" in (ctx.rdir / "MANUAL_LABELING.txt").read_text()
    from grace_al.data import read_frames

    todo = read_frames(ctx.rdir / "to_label.extxyz")
    assert [at.info["label_index"] for at in todo] == [0, 1, 2, 3]
    done = []
    for at in todo[::-1][:3]:                           # out of order, one missing
        at.calc = SinglePointCalculator(at, energy=at.info["label_index"] * 1.0,
                                        forces=np.ones((len(at), 3)))
        done.append(at)
    write(ctx.rdir / "labeled.extxyz", done)
    ctx.cfg.label.max_failure_fraction = 0.3
    res = LabelStage().run(ctx)
    assert res.status == "done" and res.summary["n_labeled"] == 3
    df = pd.read_pickle(ctx.rdir / "labeled.pkl.gz")
    assert sorted(df["label_index"]) == [1, 2, 3]
    assert all(df["energy"] == df["label_index"])


def test_manual_positional_without_index(tmp_path):
    cfg = Config(home=tmp_path, label={"labeler": "manual"})
    rdir = tmp_path / "round_00"
    m = ManualLabeler(cfg)
    frames = [fcc111("Pd", (1, 1, 2), vacuum=5) for _ in range(2)]
    tasks = m.prepare(frames, rdir / "dft")
    assert not m.is_finished(tasks[0])
    rows = []
    for i, at in enumerate(frames):
        rows.append({"ase_atoms": at, "energy": -i, "forces": np.zeros((2, 3))})
    pd.DataFrame(rows).to_pickle(rdir / "labeled.pkl.gz")
    m = ManualLabeler(cfg)
    assert m.collect(tasks[1]).get_potential_energy() == -1

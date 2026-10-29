"""SIGTERM stops whole process groups and records `interrupted` (real processes)."""

import os
import signal
import stat
import threading
import time

import pytest
import yaml

from grace_al import utils


def sigterm_after(seconds):
    threading.Timer(seconds, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie (exited, not yet reaped by its parent) counts as gone
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().split(")")[-1].split()[0] != "Z"


def test_sigterm_stops_the_whole_process_group(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    # the child starts a grandchild (like mpirun starting aims ranks) and waits for it
    cmd = ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"]
    sigterm_after(0.5)
    t0 = time.time()
    with pytest.raises(utils.Interrupted), utils.on_sigterm():
        utils.run(cmd)
    assert time.time() - t0 < 5
    time.sleep(0.2)
    assert not alive(int(pidfile.read_text()))
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_timeout_stops_the_group(tmp_path):
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        utils.wait_process(["sleep", "30"], timeout=0.3)


def fake_mpirun(tmp_path):
    p = tmp_path / "bin" / "mpirun"
    p.parent.mkdir()
    p.write_text("#!/bin/sh\necho '  Invoking FHI-aims ...'\n"
                 "echo '  Begin self-consistency iteration #    1'\nsleep 60\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def aims_cfg(tmp_path, mpirun, **label):
    from grace_al.config.schema import Config

    (tmp_path / "species").mkdir(exist_ok=True)
    return Config(home=tmp_path, label={"labeler": "aims", **label},
                  aims={"species_dir": "species", "bin": "aims.x"},
                  hpc={"mpirun": str(mpirun)})


def test_interrupted_aims_run_is_marked_and_rerun(tmp_path):
    from grace_al.labelers import aims_parse, launcher
    from grace_al.labelers.aims import AimsLabeler
    from grace_al.labelers.base import LabelTask

    cfg = aims_cfg(tmp_path, fake_mpirun(tmp_path))
    d = tmp_path / "dft" / "0000"
    d.mkdir(parents=True)
    sigterm_after(0.5)
    with pytest.raises(utils.Interrupted), utils.on_sigterm():
        launcher.run_aims(d, cfg, 2)
    assert launcher.interrupted(d)
    assert "grace-al: interrupted" in (d / "aims.err").read_text()
    task = LabelTask(0, d, None)
    lab = AimsLabeler(cfg)
    assert not lab.is_finished(task) and not lab.started(task)     # array mode: relisted
    cat, msg = aims_parse.categorize((d / "aims.out").read_text(), (d / "aims.err").read_text())
    assert cat == "crashed" and "runs again on restart" in msg


def test_concurrent_label_stops_all_calculations(tmp_path):
    from grace_al.labelers.base import LabelTask
    from grace_al.stages.label import run_concurrent, slots

    started = []

    class SleepLabeler:
        name = "sleep"

        def run(self, task, slot=None):
            started.append(task.index)
            utils.wait_process(["sleep", "60"])

    cfg = aims_cfg(tmp_path, "mpirun", mode="concurrent", concurrent_calcs=2)
    tasks = [LabelTask(i, tmp_path / f"{i}", None) for i in range(4)]
    sigterm_after(0.8)
    t0 = time.time()
    with pytest.raises(utils.Interrupted), utils.on_sigterm():
        run_concurrent(SleepLabeler(), tasks, slots(cfg, 2))
    assert time.time() - t0 < 15
    assert sorted(started) == [0, 1]                      # the queued two never started


def test_workflow_records_interrupted(tmp_path, monkeypatch):
    from grace_al import state, workflow
    from grace_al.config.schema import Config

    cfg = Config(home=tmp_path, label={"labeler": "manual"})

    def boom(cfg):
        raise utils.Interrupted("SIGTERM")

    monkeypatch.setattr(workflow, "run_seeds", boom)
    with pytest.raises(utils.Interrupted):
        workflow.run(cfg)
    st = state.load_state(cfg.workdir)
    assert st.status == "interrupted" and "the next run resumes" in st.message
    assert workflow.EXIT_CODES["interrupted"] == 4


def test_cli_run_exit_code_4(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from grace_al import workflow
    from grace_al.cli import app

    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "label": {"labeler": "manual"}}))

    def interrupted(cfg, k=None):
        raise utils.Interrupted("SIGTERM")

    monkeypatch.setattr(workflow, "run", interrupted)
    r = CliRunner().invoke(app, ["run", str(p)])
    assert r.exit_code == 4 and "interrupted" in r.output

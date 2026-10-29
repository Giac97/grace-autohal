"""Submission logic with fake sbatch / squeue on PATH."""

import stat

import pytest
import yaml
from typer.testing import CliRunner

from grace_al.cli import app
from grace_al.config.loader import load_config
from grace_al.hpc import slurm
from grace_al.state import State, load_state, save_state

SBATCH = """#!/bin/sh
# fake sbatch: ids 1001, 1002, ...; records the arguments
d=$(dirname "$0")
n=$(($(cat "$d/next" 2>/dev/null || echo 1000) + 1)); echo $n > "$d/next"
echo "$n $*" >> "$d/sbatch.log"
echo "$n;cluster"
"""
SQUEUE = """#!/bin/sh
# fake squeue: prints the lines of squeue.out
cat "$(dirname "$0")/squeue.out" 2>/dev/null
"""


@pytest.fixture
def fakebin(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE)):
        (b / name).write_text(text)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{b}:/usr/bin:/bin")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return b


def calls(b):
    p = b / "sbatch.log"
    return p.read_text().splitlines() if p.exists() else []


@pytest.fixture
def camp(tmp_path):
    def make(mode="inline", **hpc):
        p = tmp_path / "camp" / "config.yaml"
        p.parent.mkdir(exist_ok=True)
        p.write_text(yaml.safe_dump({
            "schema_version": 1, "site": "bwunicluster3",
            "label": {"labeler": "aims", "mode": mode},
            "aims": {"species_dir": "species"}, "hpc": {"max_resubmit": 3, **hpc}}))
        return load_config(p), p
    return make


def test_submit_job_records_id(fakebin, camp):
    cfg, p = camp()
    jid, script = slurm.submit_job(cfg, p)
    assert jid == "1001" and script.exists() and script.parent == cfg.workdir / "jobs"
    assert (cfg.home / "logs").is_dir()
    job = load_state(cfg.workdir).jobs[0]
    assert (job["id"], job["kind"], job["dependency"]) == ("1001", "gpu", None)
    assert calls(fakebin)[0].startswith("1001 --parsable ")


def write_tasks(cfg, k, n_done, n_todo):
    dft = cfg.workdir / f"round_{k:02d}" / "dft"
    dirs = []
    for i in range(n_done + n_todo):
        d = dft / f"{i:04d}"
        d.mkdir(parents=True)
        if i < n_done:
            (d / "aims.out").write_text("...\n          Have a nice day.\n")
        dirs.append(str(d))
    (dft / "tasks.txt").write_text("".join(f"{d}\n" for d in dirs))
    return dirs


def test_array_chain(fakebin, camp):
    cfg, p = camp("array", array_max_parallel=5)
    save_state(cfg.workdir, State(round=2, stage="label", status="waiting_for_labels"))
    dirs = write_tasks(cfg, 2, n_done=1, n_todo=3)
    ids = slurm.submit_labels(cfg, p)
    assert ids == ["1001", "1002"]
    c = calls(fakebin)
    assert "--dependency" not in c[0] and "dft_round02" in c[0]
    assert "--dependency=afterany:1001" in c[1]
    listed = (cfg.workdir / "jobs" / "round_02_tasks.txt").read_text().split()
    assert listed == dirs[1:]                       # only the unfinished tasks
    script = next((cfg.workdir / "jobs").glob("*dft_round02.sbatch")).read_text()
    assert "#SBATCH --array=0-2%5" in script
    jobs = load_state(cfg.workdir).jobs
    assert [(j["id"], j["kind"]) for j in jobs] == [("1001", "array"), ("1002", "gpu")]
    assert jobs[1]["dependency"] == "afterany:1001"


def test_array_all_done_resubmits_gpu_only(fakebin, camp):
    cfg, p = camp("array")
    save_state(cfg.workdir, State(round=0, status="waiting_for_labels"))
    write_tasks(cfg, 0, n_done=2, n_todo=0)
    assert slurm.submit_labels(cfg, p) == ["1001"]


def test_labels_needs_waiting_state(fakebin, camp):
    cfg, p = camp("array")
    with pytest.raises(slurm.SubmitError, match="waiting_for_labels"):
        slurm.submit_labels(cfg, p)


def test_resubmit_limits(fakebin, camp):
    cfg, p = camp()
    slurm.check_can_submit(cfg, resubmit=3)
    with pytest.raises(slurm.SubmitError, match="max_resubmit"):
        slurm.check_can_submit(cfg, resubmit=4)
    cfg.workdir.mkdir(exist_ok=True)
    (cfg.workdir / "FINISHED").write_text("converged\n")
    with pytest.raises(slurm.SubmitError, match="ended"):
        slurm.check_can_submit(cfg, resubmit=1)


def test_active_job_guard(fakebin, camp, monkeypatch):
    cfg, p = camp()
    slurm.submit_job(cfg, p)
    (fakebin / "squeue.out").write_text("1001 RUNNING\n")
    with pytest.raises(slurm.SubmitError, match="still queued or running"):
        slurm.check_can_submit(cfg)
    slurm.check_can_submit(cfg, force=True)
    monkeypatch.setenv("SLURM_JOB_ID", "1001")       # the job's own trap may resubmit
    slurm.check_can_submit(cfg, resubmit=1)


def test_cli_submit_and_dry_run(fakebin, camp):
    cfg, p = camp()
    r = CliRunner().invoke(app, ["submit", str(p), "--dry-run"])
    assert r.exit_code == 0 and "#SBATCH --signal=B:USR1@900" in r.output
    assert calls(fakebin) == []
    r = CliRunner().invoke(app, ["submit", str(p)])
    assert r.exit_code == 0 and "submitted: 1001" in r.output
    (fakebin / "squeue.out").write_text("1001 PENDING\n")
    r = CliRunner().invoke(app, ["submit", str(p)])
    assert r.exit_code == 1 and "still queued" in r.output


def test_cli_waiting_array_campaign_submits_chain(fakebin, camp):
    cfg, p = camp("array")
    save_state(cfg.workdir, State(round=0, status="waiting_for_labels"))
    write_tasks(cfg, 0, n_done=0, n_todo=2)
    r = CliRunner().invoke(app, ["submit", str(p)])
    assert r.exit_code == 0 and "submitted: 1001, 1002" in r.output

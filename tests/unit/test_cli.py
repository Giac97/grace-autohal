from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from grace_al import _tp
from grace_al.cli import app
from grace_al.config.loader import load_config

runner = CliRunner()


def test_help_lists_all_commands():
    out = runner.invoke(app, ["--help"]).output
    for cmd in ("init", "check", "seeds", "run", "prepare", "submit", "status", "redo",
                "report"):
        assert cmd in out


@pytest.mark.parametrize("template", ["surface", "custom_seeds"])
def test_init_creates_loadable_config(tmp_path, template):
    res = runner.invoke(app, ["init", template, str(tmp_path / "c"), "--site", "bwunicluster3"])
    assert res.exit_code == 0, res.output
    cfg = load_config(tmp_path / "c" / "config.yaml")
    assert cfg.site == "bwunicluster3"
    assert cfg.aims.parameters["elsi_build_dm_method"] == "pdgemm"


def test_init_refuses_overwrite(tmp_path):
    assert runner.invoke(app, ["init", "surface", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["init", "surface", str(tmp_path)]).exit_code == 1
    assert runner.invoke(app, ["init", "surface", str(tmp_path), "--force"]).exit_code == 0


def test_init_unknown_template(tmp_path):
    assert runner.invoke(app, ["init", "nope", str(tmp_path)]).exit_code == 2


def _teacher_config(tmp_path: Path) -> Path:
    from ase.build import fcc111
    from ase.io import write

    slab = fcc111("Pd", (2, 2, 3), vacuum=10.0)
    slab.pbc = True
    write(tmp_path / "seeds.extxyz", [slab])
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "label": {
        "labeler": "teacher", "teacher_model": "GRACE-2L-OMAT-large-ft-E"}}))
    return p


def test_check_fails_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp, "list_gpus", lambda: 1)
    res = runner.invoke(app, ["check", str(_teacher_config(tmp_path))])
    assert res.exit_code == 1
    assert "[FAIL] checkpoint in" in res.output
    assert "grace_models checkpoint GRACE-2L-SMAX-OMAT-medium" in res.output


def test_check_passes_with_fake_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp, "list_gpus", lambda: 0)
    fm = _tp.fm_checkpoint_dir("GRACE-2L-SMAX-OMAT-medium")
    fm.mkdir(parents=True)
    (fm / "model.yaml").touch()
    (fm / "checkpoint.index").touch()
    teacher = _tp.fm_savedmodel_dir("GRACE-2L-OMAT-large-ft-E")
    teacher.mkdir(parents=True)
    (teacher / "saved_model.pb").touch()
    monkeypatch.setenv("PATH", str(_fake_tools(tmp_path)) + ":" + __import__("os").environ["PATH"])
    res = runner.invoke(app, ["check", str(_teacher_config(tmp_path))])
    assert res.exit_code == 0, res.output
    assert "[warn] 0 GPU(s) visible" in res.output


def _fake_tools(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for tool in ("gracemaker", "grace_uq", "grace_models"):
        (b / tool).write_text("#!/bin/sh\n")
        (b / tool).chmod(0o755)
    return b


def test_bad_config_exit_code(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("schema_version: 1\nexplore: {tua: [0.2]}\n")
    res = runner.invoke(app, ["check", str(p), "--no-gpu"])
    assert res.exit_code == 2
    assert "explore.tua" in res.output


def test_status_of_a_fresh_campaign(tmp_path):
    p = _teacher_config(tmp_path)
    res = runner.invoke(app, ["status", str(p)])
    assert res.exit_code == 0, res.output
    assert "round 0" in res.output and "running" in res.output

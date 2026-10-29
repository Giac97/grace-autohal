import json

import pytest
from typer.testing import CliRunner

from grace_al.cli import app
from grace_al.config.loader import load_config
from grace_al.export import ExportError, export


def with_models(fake_runs):
    """Add the model files gracemaker / grace_uq write to the fake fits."""
    cfg = load_config(fake_runs)
    for k in (0, 1):
        fit = cfg.workdir / f"round_{k:02d}" / "fit"
        seed = fit / "seed" / "1"
        (seed / "final_model" / "variables").mkdir(parents=True)
        (seed / "final_model" / "saved_model.pb").write_text("pb")
        (seed / "model.yaml").write_text("model")
        (seed / "checkpoints").mkdir()
        for ext in ("index", "data-00000-of-00001"):
            (seed / "checkpoints" / f"checkpoint.best_test_loss.{ext}").write_text("x")
        (seed / "checkpoints" / "checkpoint.index").write_text("latest, not exported")
        (seed / "log.txt").write_text("reference_energy=auto: computed single-atom energies "
                                      "(eV): {'O': -2047.1, 'Pd': -136000.5}\n")
        (fit / "UQ" / "saved_model").mkdir(parents=True)
        (fit / "UQ" / "gmm_artifacts.npz").write_text("npz")
        (fit / "input.yaml").write_text("seed: 1\n")
    return cfg


def test_export_last_round(fake_runs):
    cfg = with_models(fake_runs)
    out = export(cfg)
    assert out == cfg.home / "export" / "round_01"
    for p in ("saved_model/saved_model.pb", "uq/saved_model", "uq/gmm_artifacts.npz",
              "checkpoint/model.yaml", "checkpoint/checkpoint.best_test_loss.index",
              "data/input.yaml", "data/test_metrics.yaml", "MODEL_CARD.md"):
        assert (out / p).exists(), p
    assert not (out / "checkpoint" / "checkpoint.index").exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["round"] == 1 and prov["reference_energies"]["Pd"] == -136000.5
    assert prov["out_of_sample"] is None and prov["versions"]["aims"] == "261007"
    card = (out / "MODEL_CARD.md").read_text()
    assert "Pd -136000.500000 eV" in card and "Out of sample: not measured" in card


def test_export_round0_has_out_of_sample(fake_runs):
    cfg = with_models(fake_runs)
    out = export(cfg, 0)
    assert (out / "validation" / "validation.json").exists()
    assert "force RMSE 0.12 eV/Å (random picks: 0.05 eV/Å)" in (out / "MODEL_CARD.md").read_text()


def test_export_refuses_overwrite_and_missing(fake_runs):
    cfg = with_models(fake_runs)
    export(cfg, 1)
    with pytest.raises(ExportError, match="exists"):
        export(cfg, 1)
    export(cfg, 1, force=True)
    with pytest.raises(ExportError, match="no finished fit"):
        export(cfg, 5)


def test_export_cli(fake_runs):
    with_models(fake_runs)
    r = CliRunner().invoke(app, ["export", str(fake_runs), "--round", "0"])
    assert r.exit_code == 0 and "MODEL_CARD.md" in r.output

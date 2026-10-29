from typer.testing import CliRunner

from grace_al.cli import app
from grace_al.config.loader import load_config
from grace_al.report import round_rows


def test_round_rows(fake_runs):
    rows = round_rows(load_config(fake_runs), gammas=True)
    assert [r["round"] for r in rows] == [0, 1]
    r0, r1 = rows
    assert (r0["collected"], r0["rejected"], r0["selected"], r0["natural_set"]) == (40, 3, 12, 10)
    assert (r1["labeled"], r1["failed"], r1["failures"]) == (6, 1, {"scf_not_converged": 1})
    assert r0["val_force_rmse"] is None and r1["val_force_rmse"] == 0.12
    assert r1["val_by_strategy"] == {"calm-greedy": 0.15, "random-all": 0.05}
    assert r1["fit_test"]["force_rmse"] == 0.06 and r1["fit_done"]
    assert r0["label_seconds"] == 7200 and r0["aims_version"] == "261007"
    assert r1["gammas"] == [4.5, 3.0, 2.4]


def test_status_table(fake_runs):
    r = CliRunner().invoke(app, ["status", str(fake_runs)])
    assert r.exit_code == 0, r.output
    assert "finished" in r.output and "max_rounds (2) reached" in r.output
    assert "40 / 3" in r.output and "0.12" in r.output and "(0.05)" in r.output
    assert "1001" in r.output


def test_status_shows_failure(fake_runs):
    from grace_al.state import update_state

    cfg = load_config(fake_runs)
    update_state(cfg.workdir, status="failed", stage="label",
                 message="StageFailed: 5/8 labels failed ... Fix: set pdgemm")
    r = CliRunner().invoke(app, ["status", str(fake_runs)])
    assert "failed" in r.output and "Fix: set pdgemm" in r.output

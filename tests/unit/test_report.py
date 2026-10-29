import json
import subprocess
import sys

from typer.testing import CliRunner

from grace_al.cli import app


def test_report_plots_and_html(fake_runs):
    r = CliRunner().invoke(app, ["report", str(fake_runs), "--html"])
    assert r.exit_code == 0, r.output
    out = fake_runs.parent / "runs" / "report"
    for name in ("exploration", "natural_set", "validation", "fit", "gamma", "failures",
                 "time"):
        assert (out / f"{name}.png").stat().st_size > 1000, name
    page = (out / "index.html").read_text()
    assert page.count("data:image/png;base64,") == 7
    assert "<td>0.12</td>" in page and "<title>" in page
    rows = json.loads((out / "rounds.json").read_text())
    assert len(rows) == 2


def test_report_without_tensorflow(fake_runs):
    code = ("import sys; sys.modules['tensorflow'] = None; sys.modules['tensorpotential'] = None\n"
            "from grace_al.config.loader import load_config\n"
            "from grace_al.report import write_report\n"
            f"write_report(load_config({str(fake_runs)!r}), True)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_redo_cli(fake_runs):
    r = CliRunner().invoke(app, ["redo", str(fake_runs), "--round", "1", "--from", "label"],
                           input="n\n")
    assert r.exit_code == 1 and "labeled" not in r.output or "will delete" in r.output
    runs = fake_runs.parent / "runs"
    assert (runs / "round_01" / ".done" / "label.json").exists()
    r = CliRunner().invoke(app, ["redo", str(fake_runs), "--round", "1", "--from", "label",
                                 "--yes"])
    assert r.exit_code == 0
    assert not (runs / "round_01" / ".done" / "label.json").exists()
    assert (runs / "round_01" / ".done" / "select.json").exists()
    assert not (runs / "FINISHED").exists()

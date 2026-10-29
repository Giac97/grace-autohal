import shutil
from pathlib import Path

import pytest
from ase.build import bulk, fcc111

from grace_al.config.schema import Config
from grace_al.labelers import aims_parse
from grace_al.labelers.aims import AimsLabeler
from grace_al.labelers.base import LabelFailure, LabelTask

DATA = Path(__file__).parent / "data" / "aims"


def text(name, ext="out"):
    return aims_parse.read_text(DATA / f"{name}.{ext}")


@pytest.mark.parametrize("name, category, words", [
    ("charge_integration", "charge_integration", "Charge integration error is too large"),
    ("scf_fail", "scf_not_converged", "SCF cycle not converged"),
    ("setup_fail", "setup", "No species in control.in"),
    ("stacksize", "stacksize", "stacksize too low"),
    ("mpi", "mpi", "prterun"),
    ("interrupted", "crashed", "33 SCF iterations"),
])
def test_categorize(name, category, words):
    cat, msg = aims_parse.categorize(text(name), text(name, "err"))
    assert cat == category
    assert words in msg


def test_hint_for_every_category():
    assert {"stacksize", "mpi", "charge_integration", "scf_not_converged", "setup",
            "crashed", "parse"} <= set(aims_parse.HINTS)
    assert "pdgemm" in aims_parse.HINTS["charge_integration"]
    assert "I_MPI_COLL_INTRANODE" in aims_parse.HINTS["charge_integration"]


def test_ok_outputs():
    for name in ("ok_slab", "ok_bulk"):
        assert aims_parse.finished(text(name))
    for name in ("scf_fail", "setup_fail", "charge_integration", "interrupted", "stacksize"):
        assert not aims_parse.finished(text(name))
    assert aims_parse.version(text("ok_slab")) == "261007"
    assert aims_parse.version(text("ok_bulk")) == "260331"
    assert aims_parse.scf_iterations(text("scf_fail")) == 3
    assert aims_parse.scf_iterations(text("ok_slab")) == 67      # trimmed: last number
    assert aims_parse.scf_iterations(text("setup_fail")) == 0


def test_read_results_slab():
    at = aims_parse.read_results(DATA / "ok_slab.out", pbc=[True] * 3)
    r = at.calc.results
    assert len(at) == 45 and r["forces"].shape == (45, 3)
    assert r["energy"] == pytest.approx(-6249682.935728, abs=1e-5)   # the job log's value
    assert "stress" in r


# ---------------------------------------------------------------- AimsLabeler.collect
@pytest.fixture
def labeler(tmp_path):
    (tmp_path / "species").mkdir()
    cfg = Config(home=tmp_path, label={"labeler": "aims"}, aims={"species_dir": "species"})
    return AimsLabeler(cfg)


def task_with(tmp_path, name, i=0, atoms=None):
    d = tmp_path / "dft" / f"{i:04d}"
    d.mkdir(parents=True)
    for ext in ("out", "err"):
        if (DATA / f"{name}.{ext}").exists():
            shutil.copy(DATA / f"{name}.{ext}", d / f"aims.{ext}")
    return LabelTask(i, d, atoms or fcc111("Pd", (3, 3, 5), vacuum=8))


def test_collect_finished_and_failed(labeler, tmp_path):
    ok = task_with(tmp_path, "ok_bulk", 0, bulk("Al"))
    bad = task_with(tmp_path, "charge_integration", 1)
    never = LabelTask(2, tmp_path / "dft" / "0002", bulk("Al"))
    assert labeler.is_finished(ok) and not labeler.is_finished(bad)
    assert labeler.started(bad) and not labeler.started(never)
    at = labeler.collect(ok)
    assert at.pbc.all() and at.get_forces().shape == (1, 3)
    f = labeler.collect(bad)
    assert isinstance(f, LabelFailure) and (f.index, f.category) == (1, "charge_integration")
    assert labeler.collect(never).message.startswith("not run")
    s = labeler.summary([ok, bad, never])
    assert s["versions"] == {"aims": "260331"} and s["mean_scf_iterations"] == 10


def test_prepare_leaves_finished_folder_alone(labeler, tmp_path):
    t = task_with(tmp_path, "ok_bulk", 0, bulk("Al"))
    before = {p.name: p.stat().st_mtime_ns for p in t.workdir.iterdir()}
    tasks = labeler.prepare([bulk("Al")], tmp_path / "dft")
    assert labeler.is_finished(tasks[0])
    assert {p.name: p.stat().st_mtime_ns for p in t.workdir.iterdir()} == before

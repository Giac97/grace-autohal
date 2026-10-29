"""One FHI-aims calculation through the label stage: Pd(111) 2x2x3 + O (13 atoms), light
species, 6 ranks, workstation site profile (system OpenMPI). ~5-15 min on a workstation.

Run with:  pytest -m aims tests/integration/test_aims_slab.py
Needs AIMS_BIN (default /usr/local/bin/aims.x) and AIMS_SPECIES_DIR (light species).
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml
from ase.build import add_adsorbate, fcc111
from ase.io import write

from grace_al import state
from grace_al.config.loader import load_config
from grace_al.models import ModelRef
from grace_al.stages.base import RoundContext, run_stage
from grace_al.stages.label import LabelStage
from grace_al.systems.base import SystemInfo

pytestmark = pytest.mark.aims

AIMS_BIN = os.environ.get("AIMS_BIN", "/usr/local/bin/aims.x")
SPECIES = os.environ.get(
    "AIMS_SPECIES_DIR",
    os.path.expanduser("~/Software/Scientific/fhi-aims.260331/species_defaults/"
                       "defaults_2020/light"))


@pytest.fixture(scope="module")
def labeled(tmp_path_factory):
    if not (shutil.which(AIMS_BIN) or Path(AIMS_BIN).is_file()):
        pytest.skip(f"no aims binary {AIMS_BIN}")
    if not Path(SPECIES, "46_Pd_default").exists():
        pytest.skip(f"no light species in {SPECIES}")
    home = tmp_path_factory.mktemp("aims_slab")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "site": "workstation",
        "label": {"labeler": "aims", "mode": "inline"},
        "aims": {"bin": AIMS_BIN, "species_dir": SPECIES, "np": 6, "kspacing": 0.3,
                 "parameters": {"sc_iter_limit": 150}},
    }))
    cfg = load_config(home / "config.yaml")
    slab = fcc111("Pd", (2, 2, 3), vacuum=7.0, periodic=True)
    add_adsorbate(slab, "O", 1.2, "fcc")
    slab.info["strategy"] = "calm-greedy"
    rdir = state.round_dir(cfg.workdir, 1)
    rdir.mkdir(parents=True)
    write(rdir / "selected.extxyz", [slab])
    ctx = RoundContext(cfg, 1, cfg.workdir, rdir, SystemInfo("slab", ("O", "Pd"), 2),
                       ModelRef("fm", home / "m", home / "c"))
    os.environ.pop("AIMS_NP", None)
    os.environ.pop("SLURM_NTASKS", None)
    res = run_stage(LabelStage(), ctx)
    return ctx, res


def test_converged_with_forces_and_stress(labeled):
    import pandas as pd

    ctx, res = labeled
    assert res.status == "done", (ctx.rdir / "failures.json").read_text()
    assert res.summary["n_labeled"] == 1 and res.summary["failures"] == {}
    df = pd.read_pickle(ctx.rdir / "labeled.pkl.gz")
    assert df["forces"].iloc[0].shape == (13, 3)
    assert df["stress"].iloc[0].shape == (3, 3)
    assert df["energy"].iloc[0] < -1e5                   # all-electron total, eV
    m = state.read_marker(ctx.rdir, "label")
    assert m["versions"]["aims"] and m["summary"]["mean_scf_iterations"] > 0


def test_inputs_slab_settings(labeled):
    ctx, _ = labeled
    words = " ".join((ctx.rdir / "dft" / "0000" / "control.in").read_text().split())
    assert "use_dipole_correction" in words
    k = words.split("k_grid ")[1].split()[:3]
    assert k[2] == "1" and k[0] == k[1] != "1"


def test_second_run_recomputes_nothing(labeled):
    ctx, _ = labeled
    out = ctx.rdir / "dft" / "0000" / "aims.out"
    before = out.stat().st_mtime_ns
    (ctx.rdir / ".done" / "label.json").unlink()        # force the stage to run again
    assert LabelStage().run(ctx).status == "done"
    assert out.stat().st_mtime_ns == before

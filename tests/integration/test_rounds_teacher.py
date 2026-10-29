"""Two real rounds on a Pd(111)+O toy with the teacher labeler (CPU: ~15-30 min).

Run with:  pytest -m grace tests/integration/test_rounds_teacher.py
Needs tensorpotential and, in the GRACE cache, the GRACE-2L-SMAX-OMAT-medium checkpoint +
SavedModel and the GRACE-2L-OMAT-large-ft-E SavedModel (teacher).
"""

import json
import shutil

import pytest
import yaml

from grace_al import _tp, state, workflow
from grace_al.config.loader import load_config
from grace_al.stages.base import FingerprintMismatch

pytestmark = pytest.mark.grace

FM, TEACHER = "GRACE-2L-SMAX-OMAT-medium", "GRACE-2L-OMAT-large-ft-E"

CONFIG = {
    "schema_version": 1,
    "foundation_model": FM,
    "system": {"generator": "surface",
               "surface": {"element": "Pd", "facet": "fcc111", "size": [3, 3, 5],
                           "vacuum": 8.0},
               "adsorbate": {"species": "O"},
               "initial": {"strains": [1.0], "coverages": [2], "n_per_coverage": 2,
                           "relax_steps": 20}},
    "explore": {"ensembles": ["nvt"], "temperatures": [800], "tau": [0.25], "n_steps": 300,
                "walkers_per_setting": 1, "backend": "inproc"},
    "select": {"max_per_round": 10, "n_random": 2},
    "label": {"labeler": "teacher", "teacher_model": TEACHER},
    # never used by the teacher, but part of the label stage's fingerprint
    "aims": {"species_dir": "species", "kspacing": 0.2},
    "finetune": {"target_total_updates": 20, "batch_size": 4},
    "convergence": {"max_rounds": 2, "max_natural_set": -1},   # never converge early
}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    import os

    # the conftest points GRACE_CACHE to an empty folder; this test needs the real one
    cache = os.environ.get("GRACE_AL_TEST_CACHE") or os.path.expanduser("~/.cache/grace")
    os.environ["GRACE_CACHE"] = cache
    if not (_tp.checkpoint_cached(FM) and _tp.savedmodel_cached(FM)
            and _tp.savedmodel_cached(TEACHER)):
        pytest.skip(f"models not in {cache}")
    if not shutil.which("gracemaker"):
        pytest.skip("gracemaker not on PATH")
    home = tmp_path_factory.mktemp("pd111_o")
    (home / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    cfg = load_config(home / "config.yaml")
    status = workflow.run(cfg)
    return home, cfg, status


@pytest.fixture(autouse=True)
def _real_cache(monkeypatch):
    import os

    monkeypatch.setenv("GRACE_CACHE", os.environ.get("GRACE_AL_TEST_CACHE")
                       or os.path.expanduser("~/.cache/grace"))


def test_two_rounds_finish(campaign):
    _, cfg, status = campaign
    assert status == "finished"
    assert (cfg.workdir / "FINISHED").read_text().startswith("finished")
    assert state.load_state(cfg.workdir).status == "finished"


def test_markers(campaign):
    _, cfg, _ = campaign
    r0, r1 = (state.round_dir(cfg.workdir, k) for k in (0, 1))
    for s in ("seeds", "artifact", "explore", "select", "label", "validate", "finetune"):
        assert state.read_marker(r0, s) is not None, s
    assert (r0 / "validate" / "parity.png").exists()            # the FM vs the first labels
    assert (r0 / "validate" / "parity.npz").exists()
    for s in ("artifact", "explore", "select", "label", "validate", "finetune"):
        assert state.read_marker(r1, s) is not None, s
    assert (r1 / "fit" / "FINISHED").exists()
    assert (r1 / "fit" / "UQ" / "gmm_artifacts.npz").exists()


def test_round1_driven_by_finetuned_model(campaign):
    _, cfg, _ = campaign
    r1 = state.round_dir(cfg.workdir, 1)
    v = json.loads((r1 / "validation.json").read_text())
    assert v["driver"] == "finetuned" and v["n"] > 0
    assert v["force_rmse"] >= 0 and "by_strategy" in v
    plan = json.loads((r1 / "hal" / "plan.json").read_text())
    assert len(plan["walkers"]) == 1


def test_second_run_does_nothing(campaign):
    _, cfg, _ = campaign
    stamps = {p: p.stat().st_mtime_ns for p in cfg.workdir.rglob(".done/*.json")}
    assert workflow.run(cfg) == "finished"
    assert {p: p.stat().st_mtime_ns for p in cfg.workdir.rglob(".done/*.json")} == stamps


def test_changed_aims_setting_is_refused(campaign):
    home, _, _ = campaign
    d = yaml.safe_load((home / "config.yaml").read_text())
    d["aims"]["kspacing"] = 0.15
    (home / "changed.yaml").write_text(yaml.safe_dump(d))
    cfg = load_config(home / "changed.yaml")
    with pytest.raises(FingerprintMismatch, match="--round 0 --from label"):
        workflow.run(cfg)

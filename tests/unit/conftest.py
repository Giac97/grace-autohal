import json

import pytest
import yaml
from ase.build import fcc111
from ase.io import write

from grace_al.state import State, save_state, write_marker


def marker(stage, k, summary, started="2026-10-29T10:00:00+00:00",
           finished="2026-10-29T10:30:00+00:00", **extra):
    return {"stage": stage, "round": k, "fingerprint": "x", "result": "done",
            "versions": {"grace_al": "0.1", "tensorpotential": "0.6.2.1", **extra},
            "started": started, "finished": finished, "summary": summary}


@pytest.fixture
def fake_runs(tmp_path):
    """A campaign with two finished rounds (no TF, no real outputs)."""
    p = tmp_path / "camp" / "config.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump({"schema_version": 1, "label": {"labeler": "manual"}}))
    root = p.parent / "runs"
    for k in (0, 1):
        r = root / f"round_{k:02d}"
        write_marker(r, "explore", marker("explore", k, {
            "n_walkers": 8, "collected": 40 - 15 * k, "rejected": {"gamma": 3, "short_bond": k}}))
        write_marker(r, "select", marker("select", k, {
            "n_selected": 12 - 5 * k, "natural_set": 10 - 5 * k,
            "picks": {"calm-greedy": 10 - 5 * k, "random-all": 2}}))
        write_marker(r, "label", marker("label", k, {
            "n_labeled": 11 - 5 * k, "n_failed": 1, "mode": "inline",
            "failures": {"scf_not_converged": 1}},
            started="2026-10-29T10:30:00+00:00", finished="2026-10-29T12:30:00+00:00",
            aims="261007"))
        (r / "failures.json").write_text(json.dumps(
            {"counts": {"scf_not_converged": 1}, "failures": []}))
        write_marker(r, "finetune", marker("finetune", k, {"n_train": 50}))
        (r / "fit" / "seed" / "1").mkdir(parents=True)
        (r / "fit" / "FINISHED").touch()
        (r / "fit" / "seed" / "1" / "test_metrics.yaml").write_text(yaml.safe_dump([
            {"rmse/f_comp": 0.2, "rmse/depa": 0.01, "epoch": 0},
            {"rmse/f_comp": 0.08 - 0.02 * k, "rmse/depa": 0.004, "epoch": 9}]))
        frames = []
        for g in (3.5, 2.0, 1.4):
            at = fcc111("Pd", (1, 1, 2), vacuum=5, periodic=True)
            at.info["gamma_max_pristine"] = g + k
            frames.append(at)
        write(r / "selected.extxyz", frames)
    write_marker(root / "round_01", "validate", marker("validate", 1, {}))
    (root / "round_01" / "validation.json").write_text(json.dumps({
        "force_rmse": 0.12, "energy_rmse_per_atom": 0.006,
        "by_strategy": {"calm-greedy": {"force_rmse": 0.15},
                        "random-all": {"force_rmse": 0.05}}}))
    save_state(root, State(round=1, stage="finetune", status="finished",
                           message="max_rounds (2) reached",
                           jobs=[{"id": "1001", "kind": "gpu", "submitted": "x"}]))
    return p

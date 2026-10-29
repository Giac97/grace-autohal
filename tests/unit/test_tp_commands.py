from pathlib import Path

import pytest

from grace_al import _tp
from grace_al.models import ModelRef

M = ModelRef("fm", Path("/m/model.yaml"), Path("/m/checkpoint"), Path("/r/UQ/gmm.npz"))


def test_uq_build_command():
    cmd = _tp.uq_build_command(M, [Path("a.pkl.gz"), Path("b.pkl.gz")], Path("UQ/g.npz"),
                               rp_dim=28, n_clusters=[1, 2], export=False)
    assert cmd == ["grace_uq", "build", "--model-yaml", "/m/model.yaml",
                   "--checkpoint", "/m/checkpoint", "--train-data", "a.pkl.gz", "b.pkl.gz",
                   "--rp-dim", "28", "--n-clusters", "1", "2", "--n-workers", "1",
                   "--artifact-path", "UQ/g.npz", "--no-export"]


def test_hal_cli_command():
    cmd = _tp.hal_cli_command(M, Path("starts.extxyz"), 3, Path("hal/w7"), ensemble="nvt",
                              temperature=800, tau=0.25, n_steps=2000, seed=1007,
                              walker_id=7, overwrite=True)
    assert cmd == ["grace_uq", "hal", "--model-yaml", "/m/model.yaml",
                   "--checkpoint", "/m/checkpoint", "--artifact-path", "/r/UQ/gmm.npz",
                   "--structure", "starts.extxyz", "--structure-index", "3",
                   "--ensemble", "nvt", "-T", "800", "--tau", "0.25", "--n-steps", "2000",
                   "--seed", "1007", "--walker-id", "7", "--output-dir", "hal/w7",
                   "--progress-every", "0", "--overwrite"]


def test_hal_needs_artifact():
    with pytest.raises(ValueError, match="UQ artifact"):
        _tp.hal_cli_command(ModelRef("fm", Path("m"), Path("c")), Path("s"), 0, Path("o"),
                            ensemble="nvt", temperature=1, tau=1, n_steps=1, seed=0,
                            walker_id=0)


def test_uq_select_command():
    cmd = _tp.uq_select_command([Path("hal/w0"), Path("hal/w1")], Path("UQ/g.npz"),
                                ["calm-greedy:100", "random-all:5"], Path("selected.extxyz"))
    assert cmd == ["grace_uq", "select", "--hal-dirs", "hal/w0", "hal/w1",
                   "--artifact", "UQ/g.npz", "--strategy", "calm-greedy:100", "random-all:5",
                   "--output", "selected.extxyz"]


def test_uq_info_and_gracemaker_commands():
    assert _tp.uq_info_command(Path("g.npz")) == ["grace_uq", "info", "g.npz"]
    assert _tp.gracemaker_command() == ["gracemaker", "input.yaml"]
    assert _tp.gracemaker_command(restart_latest=True) == ["gracemaker", "input.yaml", "-rl"]


@pytest.mark.parametrize("name, expected", [
    ("GRACE-1L-OMAT", ["rho/reducing_"]),
    ("GRACE-2L-SMAX-OMAT-medium", ["I2/reducing_", "rho/reducing_", "I1/reducing_"]),
    ("GRACE-3L-X", ["rho1/reducing_", "rho2/reducing_", "rho3/reducing_", "eq1/reducing_",
                    "eq2/reducing_"]),
])
def test_trainable_variables(name, expected):
    assert _tp.finetune_trainable_variables(name) == expected


def test_cache_paths_follow_grace_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GRACE_CACHE", str(tmp_path))
    assert _tp.fm_checkpoint_dir("X") == tmp_path / "checkpoints" / "X"
    assert _tp.fm_savedmodel_dir("X") == tmp_path / "X"
    assert not _tp.checkpoint_cached("X")
    ref = _tp.fm_ref("X")
    assert ref.checkpoint == tmp_path / "checkpoints" / "X" / "checkpoint"


def test_version_range():
    assert _tp.version_supported("0.6.2.1")
    assert not _tp.version_supported("0.7.0")
    assert not _tp.version_supported("0.6.1")
    assert not _tp.version_supported(None)


def test_no_tensorflow_at_import():
    import subprocess
    import sys

    code = ("import sys; sys.modules['tensorflow'] = None; sys.modules['tensorpotential'] = None\n"
            "import grace_al.cli, grace_al._tp, grace_al.preflight, grace_al.explore.backends, "
            "grace_al.workflow, grace_al.state, grace_al.data, grace_al.models, "
            "grace_al.systems.surface, grace_al.systems.bulk, grace_al.systems.from_file\n"
            "from grace_al.stages.base import STAGE_CLASSES, get_stage\n"
            "[get_stage(n) for n in STAGE_CLASSES]")
    subprocess.run([sys.executable, "-c", code], check=True)

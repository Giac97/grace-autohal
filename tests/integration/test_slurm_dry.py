"""The rendered job scripts pass `sbatch --test-only` (only where sbatch exists).

Run on a login node:  GRACE_AL_TEST_CONFIG=path/to/config.yaml pytest -m slurm tests/integration
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grace_al.config.loader import load_config
from grace_al.hpc import slurm

pytestmark = pytest.mark.slurm


@pytest.fixture
def cfg_path():
    if not shutil.which("sbatch"):
        pytest.skip("no sbatch")
    p = os.environ.get("GRACE_AL_TEST_CONFIG")
    if not p:
        pytest.skip("set GRACE_AL_TEST_CONFIG to a campaign config with an hpc section")
    return Path(p)


def test_job_script_accepted(cfg_path, tmp_path):
    cfg = load_config(cfg_path)
    script = tmp_path / "job.sbatch"
    script.write_text(slurm.render_job(cfg, cfg_path))
    slurm.logs_dir(cfg).mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["sbatch", "--test-only", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_array_script_accepted(cfg_path, tmp_path):
    cfg = load_config(cfg_path)
    script = tmp_path / "array.sbatch"
    script.write_text(slurm.render_array(cfg, cfg_path, tmp_path / "tasks.txt", 4, 0))
    out = subprocess.run(["sbatch", "--test-only", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

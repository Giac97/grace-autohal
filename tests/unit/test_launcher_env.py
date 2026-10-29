import os
import stat

import pytest

from grace_al.config.schema import Config
from grace_al.labelers import launcher

CONDA = {"PATH": "/opt/conda/envs/grace/bin:/usr/bin", "CONDA_PREFIX": "/opt/conda/envs/grace",
         "LD_LIBRARY_PATH": "/opt/conda/envs/grace/lib:/usr/lib64/openmpi/lib",
         "I_MPI_ROOT": "/opt/conda/envs/grace", "FI_PROVIDER_PATH": "/opt/conda/x",
         "LIBRARY_PATH": "/opt/conda/envs/grace/lib", "HOME": "/home/u"}


def cfg(tmp_path, hpc=None, **aims):
    return Config(home=tmp_path, label={"labeler": "manual"},
                  aims={"species_dir": "species", **aims}, hpc=hpc)


def test_without_conda_removes_mpi_vars_and_paths():
    env = launcher.without_conda(CONDA)
    assert env["PATH"] == "/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/usr/lib64/openmpi/lib"
    for k in ("I_MPI_ROOT", "FI_PROVIDER_PATH", "LIBRARY_PATH", "CONDA_PREFIX"):
        assert k not in env
    assert env["HOME"] == "/home/u"


def test_saved_env_wins(tmp_path):
    c = cfg(tmp_path)
    saved = launcher.env_file(c)
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"PATH=/opt/intel/mpi/bin:/usr/bin\0I_MPI_ROOT=/opt/intel/mpi\0"
                      b"MULTI=line1\nline2\0CONDA_PREFIX=/x\0")
    loaded = launcher.load_saved_env(saved)
    assert loaded["MULTI"] == "line1\nline2"
    env = launcher.aims_env(c, environ=CONDA)
    assert env["PATH"] == "/opt/intel/mpi/bin:/usr/bin"
    assert env["I_MPI_ROOT"] == "/opt/intel/mpi"          # the module's MPI, kept
    assert "CONDA_PREFIX" not in env and "HOME" not in env
    assert env["OMP_NUM_THREADS"] == "1"


def test_site_env_and_extra_on_top(tmp_path):
    c = cfg(tmp_path, hpc={"env": {"I_MPI_COLL_INTRANODE": "pt2pt", "PATH": "/clean"}})
    env = launcher.aims_env(c, {"I_MPI_PIN_PROCESSOR_LIST": "4-7"}, environ=CONDA)
    assert env["I_MPI_COLL_INTRANODE"] == "pt2pt" and env["PATH"] == "/clean"
    assert env["I_MPI_PIN_PROCESSOR_LIST"] == "4-7"
    assert "I_MPI_ROOT" not in env


def test_save_module_env_from_current(tmp_path, monkeypatch):
    for k, v in CONDA.items():
        monkeypatch.setenv(k, v)
    p = launcher.save_module_env(tmp_path / ".aims_env")
    env = launcher.load_saved_env(p)
    assert "I_MPI_ROOT" not in env and env["PATH"] == "/usr/bin"
    assert launcher.save_env_snippet("runs/.aims_env") == 'env -0 > "runs/.aims_env"'


def test_build_command(tmp_path, monkeypatch):
    monkeypatch.delenv("AIMS_BIN", raising=False)
    assert launcher.build_command(cfg(tmp_path), 8) == ["mpirun", "-n", "8", "aims.x"]
    c = cfg(tmp_path, hpc={"mpirun": "srun --mpi=pmi2"}, bin="/opt/aims/aims.x")
    assert launcher.build_command(c, 32) == ["srun", "--mpi=pmi2", "-n", "32",
                                             "/opt/aims/aims.x"]
    monkeypatch.setenv("AIMS_BIN", "/env/aims.x")
    assert launcher.build_command(cfg(tmp_path), 2)[-1] == "/env/aims.x"


def test_total_ranks(tmp_path, monkeypatch):
    monkeypatch.delenv("AIMS_NP", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    assert launcher.total_ranks(cfg(tmp_path, np=6)) == 6
    monkeypatch.setenv("SLURM_NTASKS", "32")
    assert launcher.total_ranks(cfg(tmp_path, np=6)) == 32
    monkeypatch.setenv("AIMS_NP", "16")
    assert launcher.total_ranks(cfg(tmp_path, np=6)) == 16


@pytest.fixture
def fake_mpirun(tmp_path):
    """An 'mpirun' that records its arguments, env and stack limit as aims.out."""
    p = tmp_path / "bin" / "mpirun"
    p.parent.mkdir()
    p.write_text("#!/bin/sh\necho \"args: $*\"\necho \"OMP=$OMP_NUM_THREADS "
                 "PIN=$I_MPI_PIN_PROCESSOR_LIST ROOT=${I_MPI_ROOT:-none}\"\n"
                 "echo \"stack: $(ulimit -s)\"\necho '          Have a nice day.'\n"
                 "echo oops >&2\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_run_aims(tmp_path, fake_mpirun, monkeypatch):
    monkeypatch.setenv("I_MPI_ROOT", "/conda/impi")
    c = cfg(tmp_path, hpc={"mpirun": str(fake_mpirun)}, bin="aims.x")
    d = tmp_path / "task"
    d.mkdir()
    assert launcher.run_aims(d, c, 4, {"I_MPI_PIN_PROCESSOR_LIST": "0-3"}) == 0
    out = (d / "aims.out").read_text()
    assert "args: -n 4 aims.x" in out
    assert "OMP=1 PIN=0-3 ROOT=none" in out
    if launcher.stack_unlimited_possible():
        assert "stack: unlimited" in out
    assert (d / "aims.err").read_text() == "oops\n"
    assert launcher.run_task(c, d) == 0                 # finished before: not rerun


def test_missing_mpirun_is_an_mpi_failure(tmp_path):
    from grace_al.labelers import aims_parse

    c = cfg(tmp_path, hpc={"mpirun": str(tmp_path / "nope")})
    d = tmp_path / "task"
    d.mkdir()
    assert launcher.run_aims(d, c, 2) == 127
    cat, _ = aims_parse.categorize((d / "aims.out").read_text(), (d / "aims.err").read_text())
    assert cat == "mpi"
    assert os.path.exists(d / "aims.out")

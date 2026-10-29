import subprocess

import pytest
import yaml

from grace_al.config.loader import load_config
from grace_al.hpc import slurm


def campaign(tmp_path, site="bwunicluster3", **over):
    d = {"schema_version": 1, "site": site,
         "label": {"labeler": "aims", "mode": over.pop("mode", "inline")},
         "aims": {"species_dir": "species", "np": 16},
         "hpc": {"partition_gpu": "gpu_h100", "partition_cpu": "cpu", "account": "ka_scc",
                 **over}}
    p = tmp_path / "camp" / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d))
    return load_config(p), p


def lines(text):
    return text.splitlines()


def test_job_order_and_paths(tmp_path):
    cfg, p = campaign(tmp_path)
    text = slurm.render_job(cfg, p)
    L = lines(text)
    load = next(i for i, x in enumerate(L) if x.startswith("module load"))
    save = next(i for i, x in enumerate(L) if x.startswith("env -0 >"))
    act = next(i for i, x in enumerate(L) if "micromamba activate grace" in x)
    assert load < save < act
    assert "compiler/intel/2025.0" in L[load] and "mpi/impi" in L[load]
    assert f'env -0 > "{cfg.workdir / ".aims_env"}"' in L[save]
    assert f"#SBATCH --output={cfg.home}/logs/al_%j.log" in L
    assert "#SBATCH --partition=gpu_h100" in L and "#SBATCH --account=ka_scc" in L
    assert "#SBATCH --signal=B:USR1@900" in L
    assert "trap resubmit USR1" in text and "--resubmit 1" in text
    assert "export I_MPI_COLL_INTRANODE=pt2pt" in L
    assert "check \"$CONFIG\" --in-job || {" in text
    assert "--labels" not in text                      # inline mode


def test_resubmission_counter(tmp_path):
    cfg, p = campaign(tmp_path)
    assert "--resubmit 4" in slurm.render_job(cfg, p, resubmit=3)


@pytest.mark.parametrize("gres", ["gpu:1", "gpu:h100:1", "gpu:a100:2"])
def test_gres_syntax(tmp_path, gres):
    cfg, p = campaign(tmp_path, gres=gres)
    assert f"#SBATCH --gres={gres}" in lines(slurm.render_job(cfg, p))


def test_sbatch_extra_and_no_partition(tmp_path):
    cfg, p = campaign(tmp_path, site="workstation", partition_gpu=None,
                      sbatch_extra=["--constraint=h100", "--mail-type=END"])
    L = lines(slurm.render_job(cfg, p))
    assert not any(x.startswith("#SBATCH --partition") for x in L)
    assert "#SBATCH --constraint=h100" in L and "#SBATCH --mail-type=END" in L


def test_array_mode_job_calls_labels(tmp_path):
    cfg, p = campaign(tmp_path, mode="array")
    text = slurm.render_job(cfg, p)
    assert "if ((rc == 3)); then" in text and 'submit "$CONFIG" --labels' in text


def test_array_template(tmp_path):
    cfg, p = campaign(tmp_path, mode="array", array_max_parallel=7, label_time="02:00:00")
    text = slurm.render_array(cfg, p, cfg.workdir / "jobs" / "t.txt", 12, 3)
    L = lines(text)
    assert "#SBATCH --array=0-11%7" in L and "#SBATCH --partition=cpu" in L
    assert "#SBATCH --ntasks=16" in L and "#SBATCH --time=02:00:00" in L
    assert "micromamba" not in text                    # modules only
    assert "aims-run" in text and str(cfg.workdir / "jobs" / "t.txt") in text


def test_scripts_are_valid_bash(tmp_path):
    cfg, p = campaign(tmp_path, mode="array")
    for text in (slurm.render_job(cfg, p),
                 slurm.render_array(cfg, p, tmp_path / "t.txt", 3, 0)):
        f = tmp_path / "s.sh"
        f.write_text(text)
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_no_hpc_section(tmp_path):
    from grace_al.config.schema import Config

    cfg = Config(home=tmp_path, label={"labeler": "manual"})
    with pytest.raises(slurm.SubmitError, match="hpc"):
        slurm.render_job(cfg, tmp_path / "config.yaml")

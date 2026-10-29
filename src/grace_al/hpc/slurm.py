"""SLURM: job scripts from the config + site profile, submission, the DFT array chain.

GPU job (job.sbatch.j2)       modules -> saved aims env -> Python env -> check -> run;
                              USR1 15 min before the limit -> `grace-al submit --resubmit N+1`
DFT chain (label.mode array)  `grace-al run` exits 3 (waiting for labels) -> `grace-al submit
                              --labels`: a CPU job array over the round's unfinished tasks,
                              then the GPU job again with --dependency=afterany:<array>

Every submitted job is recorded in runs/state.json (`jobs`). Scripts go to runs/jobs/, logs
to <config dir>/logs/ (created here: a missing --output folder makes SLURM drop the job
without a log). Other schedulers can follow the same functions (render, submit, active_jobs).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from grace_al.labelers.launcher import env_file, save_env_snippet
from grace_al.state import load_state, now, round_dir, save_state

if TYPE_CHECKING:
    from grace_al.config.schema import Config

log = logging.getLogger("grace_al.hpc")


class SubmitError(RuntimeError):
    pass


# ---------------------------------------------------------------- rendering
def _env() -> jinja2.Environment:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True,
                             loader=jinja2.FunctionLoader(
                                 lambda name: (resources.files("grace_al.hpc.templates")
                                               / name).read_text()))
    env.filters["shell_quote"] = lambda v: shlex.quote(str(v))
    return env


def logs_dir(cfg: Config) -> Path:
    return cfg.home / "logs"


def jobs_dir(cfg: Config) -> Path:
    return cfg.workdir / "jobs"


def job_name(cfg: Config) -> str:
    return f"al-{cfg.home.name}"[:40]


def _common(cfg: Config, config_path: Path) -> dict:
    if cfg.hpc is None:
        raise SubmitError("no `hpc:` section (or site profile) in the config")
    return {"hpc": cfg.hpc, "home": str(cfg.home), "config": str(config_path.resolve()),
            "python": sys.executable, "logs": str(logs_dir(cfg)), "job_name": job_name(cfg)}


def render_job(cfg: Config, config_path: Path, resubmit: int = 0) -> str:
    """The GPU job script."""
    common = _common(cfg, config_path)
    assert cfg.hpc is not None
    threads = max(1, min(8, cfg.hpc.ntasks * cfg.hpc.cpus_per_task))
    return _env().get_template("job.sbatch.j2").render(
        **common, resubmit=resubmit, label_mode=cfg.label.mode,
        save_env=save_env_snippet(env_file(cfg)), threads=threads,
        job_kind="resubmission" if resubmit else "first submission")


def render_array(cfg: Config, config_path: Path, tasks_file: Path, n_tasks: int,
                 k: int) -> str:
    """The CPU job array over the lines of tasks_file."""
    return _env().get_template("aims_array.sbatch.j2").render(
        **_common(cfg, config_path), tasks_file=str(tasks_file), n_tasks=n_tasks, round=k,
        ranks=cfg.aims.np if cfg.aims else 1)


# ---------------------------------------------------------------- scheduler calls
def _sbatch() -> str:
    exe = shutil.which("sbatch")
    if not exe:
        raise SubmitError("sbatch not found: submit from a login node of the cluster")
    return exe


def sbatch(script: Path, dependency: str | None = None) -> str:
    """Submit; returns the job id."""
    cmd = [_sbatch(), "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    out = subprocess.run([*cmd, str(script)], capture_output=True, text=True)
    if out.returncode:
        raise SubmitError(f"sbatch {script.name} failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def active_jobs(ids: list[str]) -> dict[str, str]:
    """{id: state} of the jobs among `ids` that are pending or running (squeue)."""
    if not ids or not shutil.which("squeue"):
        return {}
    out = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"],
                         capture_output=True, text=True)
    res = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            res[parts[0].split("_")[0]] = parts[1]
    return res


def sacct(ids: list[str]) -> list[dict]:
    """JobID, State, Elapsed, ExitCode, Start, End of the jobs (empty without sacct)."""
    if not ids or not shutil.which("sacct"):
        return []
    fields = ["JobID", "State", "Elapsed", "ExitCode", "Start", "End"]
    out = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o",
                          ",".join(fields)], capture_output=True, text=True)
    return [dict(zip(fields, line.split("|"), strict=False))
            for line in out.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------- submission
def _write(cfg: Config, name: str, text: str) -> Path:
    d = jobs_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.sbatch")))
    p = d / f"{n:03d}_{name}.sbatch"
    p.write_text(text)
    return p


def _record(cfg: Config, **job) -> None:
    st = load_state(cfg.workdir)
    st.jobs.append({"submitted": now(), **job})
    save_state(cfg.workdir, st)


def check_can_submit(cfg: Config, resubmit: int = 0, force: bool = False) -> None:
    """Refuse when the campaign is finished, the resubmit limit is reached, or one of its
    jobs is still queued or running (other than the one calling, $SLURM_JOB_ID)."""
    assert cfg.hpc is not None
    if (cfg.workdir / "FINISHED").exists():
        raise SubmitError(f"{cfg.workdir / 'FINISHED'} exists: the campaign has ended "
                          "(grace-al redo, or a larger convergence.max_rounds, to continue)")
    if resubmit > cfg.hpc.max_resubmit:
        raise SubmitError(f"resubmission {resubmit} > hpc.max_resubmit "
                          f"{cfg.hpc.max_resubmit}: not resubmitting")
    if force:
        return
    me = os.environ.get("SLURM_JOB_ID", "")
    ids = [str(j["id"]) for j in load_state(cfg.workdir).jobs if str(j["id"]) != me]
    busy = active_jobs(ids)
    if busy:
        raise SubmitError(f"jobs of this campaign are still queued or running: {busy} "
                          "(scancel them, or --force)")


def submit_job(cfg: Config, config_path: Path, resubmit: int = 0,
               dependency: str | None = None, dry_run: bool = False) -> tuple[str, Path | str]:
    """The GPU job; returns (job id, script path) or ("dry-run", script text)."""
    text = render_job(cfg, config_path, resubmit)
    if dry_run:
        return "dry-run", text
    logs_dir(cfg).mkdir(parents=True, exist_ok=True)
    script = _write(cfg, f"gpu_r{resubmit}", text)
    jid = sbatch(script, dependency)
    _record(cfg, id=jid, kind="gpu", script=str(script), resubmit=resubmit,
            dependency=dependency, round=load_state(cfg.workdir).round)
    log.info("submitted GPU job %s (%s)", jid, script.name)
    return jid, script


def unfinished_tasks(cfg: Config, k: int) -> list[str]:
    """Task folders of round k's dft/tasks.txt whose aims.out has not finished."""
    from grace_al.labelers import aims_parse

    listing = round_dir(cfg.workdir, k) / "dft" / "tasks.txt"
    if not listing.exists():
        return []
    return [d for d in listing.read_text().split()
            if not aims_parse.finished(aims_parse.read_text(Path(d) / "aims.out"))]


def submit_labels(cfg: Config, config_path: Path, dry_run: bool = False) -> list[str]:
    """Array mode, state waiting_for_labels: the CPU array over the unfinished tasks of the
    round, then the GPU job with afterany on it. Returns the job ids (dry run: the
    scripts)."""
    st = load_state(cfg.workdir)
    if st.status != "waiting_for_labels" or cfg.label.mode != "array":
        raise SubmitError(f"--labels needs label.mode array and status waiting_for_labels "
                          f"(now: mode {cfg.label.mode}, status {st.status})")
    k = st.round
    tasks = unfinished_tasks(cfg, k)
    if not tasks:
        log.info("round %d: every listed task has finished; resubmitting the GPU job", k)
        jid, script = submit_job(cfg, config_path, dry_run=dry_run)
        return [str(script)] if dry_run else [jid]
    tasks_file = jobs_dir(cfg) / f"round_{k:02d}_tasks.txt"
    text = render_array(cfg, config_path, tasks_file, len(tasks), k)
    if dry_run:
        return [text, render_job(cfg, config_path)]
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text("".join(f"{t}\n" for t in tasks))
    logs_dir(cfg).mkdir(parents=True, exist_ok=True)
    script = _write(cfg, f"dft_round{k:02d}", text)
    array_id = sbatch(script)
    _record(cfg, id=array_id, kind="array", script=str(script), round=k, n_tasks=len(tasks))
    log.info("submitted DFT array %s: %d tasks, at most %d at a time", array_id, len(tasks),
             cfg.hpc.array_max_parallel if cfg.hpc else 0)
    gpu_id, _ = submit_job(cfg, config_path, dependency=f"afterany:{array_id}")
    return [array_id, gpu_id]


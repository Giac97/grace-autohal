"""grace-al command line."""

from __future__ import annotations

import logging
import re
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from grace_al import __version__
from grace_al.config.loader import ConfigError, available_sites, load_config

app = typer.Typer(help="Foundation-model-driven active learning for GRACE fine-tuning.",
                  no_args_is_help=True, add_completion=False)
console = Console(soft_wrap=True)  # long paths stay on one line
log = logging.getLogger("grace_al")

# exit codes for job scripts
EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_WAITING = 0, 1, 2, 3

ConfigArg = Annotated[Path, typer.Argument(help="config.yaml", exists=True, dir_okay=False)]
SiteOpt = Annotated[Optional[str], typer.Option(  # noqa: UP045 (typer needs Optional on 3.10)
    "--site", help=f"site profile ({', '.join(available_sites())}) or a YAML file")]


def templates() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("grace_al.config.templates").iterdir()
                  if p.name.endswith(".yaml"))


def _load(config: Path, site: str | None):
    try:
        return load_config(config, site=site)
    except ConfigError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(EXIT_CONFIG) from None


def _version(value: bool):
    if value:
        console.print(f"grace-al {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="debug output")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="warnings and errors only")] = False,
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="print the version")] = False,
):
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    # third-party libraries (numexpr, TF, ...) only from WARNING on
    logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", force=True,
                        handlers=[RichHandler(console=console, show_path=False,
                                              markup=False)])
    log.setLevel(level)


# ---------------------------------------------------------------- implemented
@app.command()
def init(
    template: Annotated[str, typer.Argument(help=f"one of: {', '.join(templates())}")],
    directory: Annotated[Path, typer.Argument(help="campaign folder (created)")],
    site: SiteOpt = None,
    force: Annotated[bool, typer.Option(help="overwrite an existing config.yaml")] = False,
):
    """Start a campaign: copy a template config.yaml into DIRECTORY."""
    if template not in templates():
        console.print(f"[red]unknown template {template!r}; available: "
                      f"{', '.join(templates())}[/red]")
        raise typer.Exit(EXIT_CONFIG)
    if site and site not in available_sites():
        console.print(f"[red]unknown site {site!r}; available: "
                      f"{', '.join(available_sites())}[/red]")
        raise typer.Exit(EXIT_CONFIG)
    out = directory / "config.yaml"
    if out.exists() and not force:
        console.print(f"[red]{out} exists; use --force to overwrite[/red]")
        raise typer.Exit(EXIT_ERROR)
    text = (resources.files("grace_al.config.templates") / f"{template}.yaml").read_text()
    if site:
        text = re.sub(r"^# site: .*$", f"site: {site}", text, count=1, flags=re.M)
    directory.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    console.print(f"wrote {out}; edit it, then: grace-al check {out}")


@app.command()
def check(
    config: ConfigArg,
    site: SiteOpt = None,
    gpu: Annotated[bool, typer.Option(help="probe the GPU (starts TensorFlow)")] = True,
    in_job: Annotated[bool, typer.Option("--in-job", help="inside the GPU job: no visible "
                                         "GPU is an error")] = False,
):
    """Preflight: tools, models, seeds, DFT setup, GPU. Exit code 1 if anything is missing."""
    from grace_al.preflight import run_checks

    cfg = _load(config, site)
    failed = _print_checks(run_checks(cfg, gpu=gpu, where="job" if in_job else "any"))
    raise typer.Exit(EXIT_ERROR if failed else EXIT_OK)


def _print_checks(checks) -> bool:
    """Print grouped check results; True if any error failed."""
    group = None
    for c in checks:
        if c.group != group:
            group = c.group
            console.print(f"[bold]{group}[/bold]")
        tag = ("[green]ok[/green]" if c.ok else
               "[yellow]warn[/yellow]" if c.severity == "warning" else "[red]FAIL[/red]")
        msg = f" ({c.message})" if c.message and not c.ok else ""
        console.print(f"  [{tag}] {c.name}{msg}", highlight=False)
        if not c.ok and c.fix:
            console.print(f"         -> {c.fix}", highlight=False)
    failed = [c for c in checks if c.failed]
    console.print("\n[red]some checks FAILED[/red]" if failed else "\nall checks passed")
    return bool(failed)


@app.command()
def seeds(
    config: ConfigArg,
    site: SiteOpt = None,
    force: Annotated[bool, typer.Option(help="overwrite an existing seeds file")] = False,
):
    """Build the initial structures from `system:` (FM pre-relaxation: a few minutes on CPU)."""
    from ase.io import write

    from grace_al.stages.seeds import build, summary

    cfg = _load(config, site)
    if cfg.system is None:
        console.print("[red]no `system:` section in the config[/red]")
        raise typer.Exit(EXIT_CONFIG)
    if cfg.seeds.exists() and not force:
        console.print(f"[red]{cfg.seeds} exists; use --force to overwrite[/red]")
        raise typer.Exit(EXIT_ERROR)
    frames = build(cfg)
    write(cfg.seeds, frames, format="extxyz")
    s = summary(frames, cfg.artifact.rp_dim_round0, cfg.system.generator)
    console.print(f"{s['n_seeds']} structures {s['configs']}, atoms per element "
                  f"{s['elements']}", highlight=False)
    if "warning" in s:
        console.print(f"[yellow]{s['warning']}[/yellow]", highlight=False)
    console.print(f"written to {cfg.seeds}")


@app.command()
def run(
    config: ConfigArg,
    site: SiteOpt = None,
    round_: Annotated[Optional[int], typer.Option(  # noqa: UP045
        "--round", help="run/resume only this round")] = None,
):
    """Run the active-learning loop (resumes where it stopped).

    Exit codes: 0 finished or converged, 3 waiting for DFT labels, 1 error.
    """
    from grace_al.models import MissingModel
    from grace_al.stages.base import FingerprintMismatch, StageFailed
    from grace_al.workflow import EXIT_CODES
    from grace_al.workflow import run as run_workflow

    cfg = _load(config, site)
    try:
        status = run_workflow(cfg, round_)
    except (FingerprintMismatch, StageFailed, MissingModel) as err:
        console.print(f"[red]{err}[/red]", highlight=False)
        raise typer.Exit(EXIT_ERROR) from None
    console.print(f"status: {status}")
    raise typer.Exit(EXIT_CODES[status])


# ---------------------------------------------------------------- cluster
@app.command()
def prepare(
    config: ConfigArg,
    site: SiteOpt = None,
    seeds_: Annotated[bool, typer.Option("--seeds", help="also build the seeds here "
                                         "(FM pre-relaxation on the CPU)")] = False,
):
    """Login node: download the models, run the checks, list the partitions open to you."""
    from grace_al import _tp
    from grace_al.hpc.partitions import partitions
    from grace_al.preflight import run_checks

    cfg = _load(config, site)
    models = [cfg.foundation_model]
    if cfg.label.labeler == "teacher" and cfg.label.teacher_model:
        models.append(cfg.label.teacher_model)
    for m in models:
        console.print(f"model {m}: checkpoint + SavedModel -> {_tp.grace_cache()}")
        _tp.download_fm(m, checkpoint=True, savedmodel=True)
    if seeds_ and not cfg.seeds.exists() and cfg.system is not None:
        from ase.io import write

        from grace_al.stages.seeds import build

        write(cfg.seeds, build(cfg), format="extxyz")
        console.print(f"seeds written to {cfg.seeds}")
    failed = _print_checks(run_checks(cfg, gpu=False, where="login"))

    parts = partitions()
    if parts is None:
        console.print("\n(no scontrol here: partitions not checked)")
    else:
        from rich.table import Table

        t = Table("partition", "open to you", "AllowGroups", "AllowAccounts", "MaxTime",
                  "TRES", title="SLURM partitions")
        for p in parts:
            t.add_row(p.name, "[green]yes[/green]" if p.open else "[red]no[/red]",
                      ",".join(p.allow_groups), ",".join(p.allow_accounts), p.max_time,
                      p.tres)
        console.print(t)
        names = {p.name: p for p in parts}
        for key in ("partition_gpu", "partition_cpu"):
            want = getattr(cfg.hpc, key, None) if cfg.hpc else None
            if want and (want not in names or not names[want].open):
                failed = True
                console.print(f"[red]hpc.{key} = {want} is not a partition open to you "
                              "(sbatch would fail with \"User's group not permitted to use "
                              "this partition\"); pick one marked yes[/red]")
    raise typer.Exit(EXIT_ERROR if failed else EXIT_OK)


@app.command()
def submit(
    config: ConfigArg,
    site: SiteOpt = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the script(s), submit "
                                          "nothing")] = False,
    resubmit: Annotated[int, typer.Option(help="submission number (the job's time-limit "
                                          "trap)")] = 0,
    labels: Annotated[bool, typer.Option("--labels", help="array mode: the DFT job array "
                                         "+ the GPU job after it")] = False,
    force: Annotated[bool, typer.Option(help="submit even if a job of this campaign is "
                                        "queued or running")] = False,
):
    """Write the job script from `hpc:` + the site profile and submit it.

    In array mode, while the campaign waits for labels, this submits the DFT job array and
    the GPU job that continues after it.
    """
    from grace_al.hpc import slurm
    from grace_al.state import load_state

    cfg = _load(config, site)
    try:
        waiting = (cfg.label.mode == "array"
                   and load_state(cfg.workdir).status == "waiting_for_labels")
        if dry_run:
            texts = (slurm.submit_labels(cfg, config, dry_run=True) if labels or waiting
                     else [str(slurm.submit_job(cfg, config, resubmit, dry_run=True)[1])])
            for t in texts:
                console.print(t, highlight=False, markup=False)
            raise typer.Exit(EXIT_OK)
        slurm.check_can_submit(cfg, resubmit, force)
        if labels or waiting:
            ids = slurm.submit_labels(cfg, config)
        else:
            ids = [slurm.submit_job(cfg, config, resubmit)[0]]
    except slurm.SubmitError as err:
        console.print(f"[red]{err}[/red]", highlight=False)
        raise typer.Exit(EXIT_ERROR) from None
    console.print(f"submitted: {', '.join(ids)}  (logs in {slurm.logs_dir(cfg)})")


@app.command()
def status(config: ConfigArg, site: SiteOpt = None):
    """Round history and the state of the jobs."""
    from rich.table import Table

    from grace_al.hpc import slurm
    from grace_al.report import fmt, round_rows
    from grace_al.state import load_state

    cfg = _load(config, site)
    st = load_state(cfg.workdir)
    color = {"failed": "red", "converged": "green", "finished": "green",
             "waiting_for_labels": "yellow"}.get(st.status, "cyan")
    console.print(f"round {st.round}, stage {st.stage or '-'}: [{color}]{st.status}[/{color}]"
                  f"  ({st.updated})", highlight=False)
    if st.message:
        console.print(f"  {st.message}", style="red" if st.status == "failed" else None,
                      highlight=False)
    # walkers; HAL frames collected / rejected; selected; natural set; labeled / failed;
    # fit done; validation force RMSE, all (random-all picks)
    t = Table("round", "stages", "walk.", "coll/rej", "sel.", "nat.", "lab/fail", "fit",
              "val F (rand)", caption="F in eV/Å")
    for r in round_rows(cfg):
        rnd = r["val_by_strategy"].get("random-all")
        t.add_row(str(r["round"]), str(len(r["done"])), fmt(r["walkers"]),
                  f"{fmt(r['collected'])} / {fmt(r['rejected'])}", fmt(r["selected"]),
                  fmt(r["natural_set"]), f"{fmt(r['labeled'])} / {fmt(r['failed'])}",
                  fmt(r["fit_done"]),
                  f"{fmt(r['val_force_rmse'])} ({fmt(rnd)})" if r["val_force_rmse"] else "–")
    console.print(t)
    _print_progress(cfg, st)
    _print_tau_advice(cfg)
    ids = [str(j["id"]) for j in st.jobs[-8:]]
    acct = {a["JobID"].split("_")[0]: a for a in slurm.sacct(ids)}
    if st.jobs:
        jt = Table("job", "kind", "submitted", "state", "elapsed", "exit")
        for j in st.jobs[-8:]:
            a = acct.get(str(j["id"]), {})
            jt.add_row(str(j["id"]), j.get("kind", ""), j.get("submitted", ""),
                       a.get("State", "?"), a.get("Elapsed", ""), a.get("ExitCode", ""))
        console.print(jt)


def _print_progress(cfg, st) -> None:
    """Live progress of the running (or waiting) stage."""
    from datetime import datetime, timezone

    from grace_al.progress import fmt_duration, stage_progress

    if st.status not in ("running", "waiting_for_labels") or not st.stage:
        return
    p = stage_progress(cfg, st.round, st.stage)
    since = ""
    if st.status == "running" and st.updated:
        el = (datetime.now(timezone.utc) - datetime.fromisoformat(st.updated)).total_seconds()
        since = f", running for {fmt_duration(el)}"
    console.print(f"\n[bold]round {st.round}, {st.stage}{since}[/bold]", highlight=False)
    if p is None:
        return
    if "epochs_done" in p:
        console.print(f"  {p['what']}: {p['epochs_done']} / {p['epochs_total'] or '?'}, "
                      f"ETA ~{fmt_duration(p['eta_s'])}", highlight=False)
        return
    console.print(f"  {p['what']}: {p['done']} done, {p['running']} running, {p['failed']} "
                  f"failed, {p['pending']} pending of {p['total']} ({p['parallel']} at a time)",
                  highlight=False)
    if p["mean_s"] is not None:
        console.print(f"  time per item: {fmt_duration(p['mean_s'])} on average "
                      f"({fmt_duration(p['min_s'])} - {fmt_duration(p['max_s'])}); "
                      f"ETA ~{fmt_duration(p['eta_s'])}", highlight=False)
    for r in p["running_items"][:8]:
        console.print(f"    {r['name']}  {r.get('detail', '')}  "
                      f"{fmt_duration(r.get('elapsed'))}", highlight=False)
    if len(p["running_items"]) > 8:
        console.print(f"    ... {len(p['running_items']) - 8} more", highlight=False)


def _print_tau_advice(cfg) -> None:
    from grace_al.state import read_marker, round_dir, rounds_on_disk

    for k in reversed(rounds_on_disk(cfg.workdir)):
        m = read_marker(round_dir(cfg.workdir, k), "explore")
        advice = (m or {}).get("summary", {}).get("tau_advice")
        if advice:
            style = "yellow" if advice["verdict"] in ("increase", "decrease") else None
            console.print(f"\ntau (round {k}): {advice['message']}", style=style,
                          highlight=False)
            return


@app.command()
def redo(
    config: ConfigArg,
    round_: Annotated[int, typer.Option("--round", help="round to redo")],
    from_stage: Annotated[str, typer.Option("--from", help="first stage to recompute")],
    site: SiteOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="don't ask")] = False,
):
    """Delete a round's outputs from a stage on, and every later round, to recompute them."""
    from grace_al.state import STAGES, redo_targets
    from grace_al.state import redo as do_redo

    cfg = _load(config, site)
    if from_stage not in STAGES:
        console.print(f"[red]unknown stage {from_stage!r}; stages: {', '.join(STAGES)}[/red]")
        raise typer.Exit(EXIT_CONFIG)
    paths = redo_targets(cfg.workdir, round_, from_stage)
    if not paths:
        console.print("nothing to delete")
        raise typer.Exit(EXIT_OK)
    console.print("will delete:")
    for p in paths:
        console.print(f"  {p}", highlight=False)
    if not yes and not typer.confirm("proceed?", default=False):
        raise typer.Exit(EXIT_ERROR)
    do_redo(cfg.workdir, round_, from_stage, yes=True)
    console.print(f"done; `grace-al run` (or submit) recomputes round {round_} from "
                  f"{from_stage}")


@app.command()
def report(
    config: ConfigArg,
    site: SiteOpt = None,
    html: Annotated[bool, typer.Option("--html", help="also a self-contained index.html")]
    = False,
):
    """Plots and tables of the campaign (runs/report/)."""
    from grace_al.report import write_report

    cfg = _load(config, site)
    out, rows = write_report(cfg, html)
    console.print(f"{len(rows)} round(s): plots in {out}"
                  + (f", {out / 'index.html'}" if html else ""))


@app.command()
def validate(
    config: ConfigArg,
    prop: Annotated[Optional[list[str]], typer.Option(  # noqa: UP045
        "--property", "-p", help="energy, relative, adsorption, cohesive (repeatable; "
        "default: energy, relative, + adsorption for surfaces)")] = None,
    model: Annotated[str, typer.Option(help="fm | last | round:K (the model fine-tuned in "
                                       "round K)")] = "last",
    data: Annotated[Optional[Path], typer.Option(  # noqa: UP045
        help="DFT-labeled structures (extxyz/pkl) instead of all labeled rounds")] = None,
    references: Annotated[Optional[Path], typer.Option(  # noqa: UP045
        help="DFT-labeled reference structures (isolated atoms, gas molecules) for "
             "adsorption / cohesive energies")] = None,
    substrate: Annotated[Optional[list[str]], typer.Option(  # noqa: UP045
        help="substrate elements for adsorption (default: system.surface.element)")] = None,
    site: SiteOpt = None,
):
    """Parity plots (+ .npz to replot) of a model vs DFT for derived properties."""
    from grace_al.properties import PropertyError, run

    cfg = _load(config, site)
    props = prop or ["energy", "relative"] + (
        ["adsorption"] if cfg.system is not None and cfg.system.surface is not None else [])
    try:
        res = run(cfg, model, props, data, references, substrate)
    except PropertyError as err:
        console.print(f"[red]{err}[/red]", highlight=False)
        raise typer.Exit(EXIT_ERROR) from None
    from grace_al.report import fmt

    console.print(f"model {res['model']}, {res['n_frames']} structures")
    for p in props:
        r = res.get(p, {})
        if p == "energy":
            console.print(f"  energy  RMSE {fmt(r['energy']['rmse'])} eV/atom, forces RMSE "
                          f"{fmt(r['forces']['rmse'])} eV/Å  -> {r['plot']}", highlight=False)
        elif r.get("n"):
            console.print(f"  {p:<10} RMSE {fmt(r['rmse'])}  MAE {fmt(r['mae'])}  "
                          f"(n={r['n']})  -> {r['plot']}", highlight=False)
        else:
            console.print(f"  {p:<10} no structures to compare", highlight=False)


@app.command("export")
def export_cmd(
    config: ConfigArg,
    round_: Annotated[Optional[int], typer.Option(  # noqa: UP045
        "--round", help="round whose fitted model to export (default: the last one)")] = None,
    out: Annotated[Optional[Path], typer.Option(  # noqa: UP045
        help="output folder (default: <config dir>/export/round_XX)")] = None,
    force: Annotated[bool, typer.Option(help="replace an existing export")] = False,
    site: SiteOpt = None,
):
    """The fine-tuned model of a round as a self-contained folder with a model card."""
    from grace_al.export import ExportError, export

    cfg = _load(config, site)
    try:
        path = export(cfg, round_, out, force)
    except ExportError as err:
        console.print(f"[red]{err}[/red]", highlight=False)
        raise typer.Exit(EXIT_ERROR) from None
    console.print(f"exported to {path} (see MODEL_CARD.md)", highlight=False)


@app.command("aims-run")
def aims_run(
    config: ConfigArg,
    task_dir: Annotated[Path, typer.Argument(help="a prepared aims task folder "
                                             "(runs/round_XX/dft/NNNN)")],
    site: SiteOpt = None,
):
    """Run one prepared FHI-aims calculation (DFT job array); exit code 0 if it finished."""
    from grace_al.labelers.launcher import run_task

    cfg = _load(config, site)
    raise typer.Exit(0 if run_task(cfg, task_dir.resolve()) == 0 else EXIT_ERROR)

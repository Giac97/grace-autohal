"""The only module that touches tensorpotential (Python API or CLIs).

tensorpotential (and with it TensorFlow) is imported inside the functions, never at
module level, so the rest of grace_al and the unit tests run without TF. The
`*_command` functions only build argument lists; the others run them.

Verified against tensorpotential 0.6.2.1.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from grace_al.models import ModelRef
from grace_al.utils import run

SUPPORTED = ">=0.6.2,<0.7"

# copy of tensorpotential.calculator.foundation_models.MODELS_ALIASES_DICT (importing that
# module loads TF): short or old name -> full name
FM_ALIASES = {
    "GRACE-1L-OAM_2Feb25": "GRACE-1L-OAM",
    "GRACE_2L_OAM_28Jan25": "GRACE-2L-OAM",
    "GRACE-2L-OMAT-L": "GRACE-2L-OMAT-large-ft-E",
    "GRACE-2L-OAM-L": "GRACE-2L-OMAT-large-ft-AM",
    "GRACE-2L-OMAT-M": "GRACE-2L-OMAT-medium-ft-E",
    "GRACE-2L-OAM-M": "GRACE-2L-OMAT-medium-ft-AM",
    "GRACE-1L-OMAT-L": "GRACE-1L-OMAT-large-ft-E",
    "GRACE-1L-OAM-L": "GRACE-1L-OMAT-large-ft-AM",
    "GRACE-1L-OMAT-M": "GRACE-1L-OMAT-medium-ft-E",
    "GRACE-1L-OAM-M": "GRACE-1L-OMAT-medium-ft-AM",
}

# weights trained in a frozen fine-tune (GRACE docs, "Fine-tuning foundation models")
TRAINABLE_REDUCING = {
    "1L": ["rho/reducing_"],
    "2L": ["I2/reducing_", "rho/reducing_", "I1/reducing_"],
    "3L": ["rho1/reducing_", "rho2/reducing_", "rho3/reducing_", "eq1/reducing_",
           "eq2/reducing_"],
}


# ---------------------------------------------------------------- versions, cache
def tp_version() -> str | None:
    """Installed tensorpotential version (from package metadata, no import)."""
    try:
        return metadata.version("tensorpotential")
    except metadata.PackageNotFoundError:
        return None


def version_supported(version: str | None) -> bool:
    from packaging.specifiers import SpecifierSet

    return version is not None and version in SpecifierSet(SUPPORTED)


def grace_cache() -> Path:
    """Same rule as tensorpotential: $GRACE_CACHE, else ~/.cache/grace (read at call time)."""
    return Path(os.environ.get("GRACE_CACHE") or Path.home() / ".cache" / "grace")


def fm_checkpoint_dir(name: str) -> Path:
    """Checkpoint of a foundation model (model.yaml + checkpoint.*), for HAL, UQ and fits."""
    return grace_cache() / "checkpoints" / name


def fm_savedmodel_dir(name: str) -> Path:
    """SavedModel of a foundation model (grace_fm calculator: seed relaxation, teacher)."""
    return grace_cache() / name


def fm_ref(name: str) -> ModelRef:
    d = fm_checkpoint_dir(name)
    return ModelRef("fm", d / "model.yaml", d / "checkpoint")


def checkpoint_cached(name: str) -> bool:
    d = fm_checkpoint_dir(name)
    return (d / "model.yaml").exists() and (d / "checkpoint.index").exists()


def savedmodel_cached(name: str) -> bool:
    return (fm_savedmodel_dir(name) / "saved_model.pb").exists()


def download_fm(name: str, checkpoint: bool = True, savedmodel: bool = True) -> None:
    """Download into the cache (needs internet: login node)."""
    if checkpoint and not checkpoint_cached(name):
        run(["grace_models", "checkpoint", name])
    if savedmodel and not savedmodel_cached(name):
        run(["grace_models", "download", name])


def fm_calculator(name: str):
    """ASE calculator of a foundation model (loads TF)."""
    from tensorpotential.calculator.foundation_models import grace_fm

    return grace_fm(name)


def list_gpus(timeout: float = 120) -> int:
    """Number of GPUs TensorFlow sees, probed in a subprocess (keeps TF out of this process)."""
    code = ("import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
            "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=timeout)
    if out.returncode:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip()
                           else f"exit code {out.returncode}")
    return int(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- grace_uq
def uq_build_command(model: ModelRef, train_data: Sequence[Path], artifact: Path, rp_dim: int,
                     n_clusters: Sequence[int], export: bool = True,
                     n_workers: int = 1) -> list[str]:
    cmd = ["grace_uq", "build",
           "--model-yaml", model.model_yaml, "--checkpoint", model.checkpoint,
           "--train-data", *train_data,
           "--rp-dim", rp_dim, "--n-clusters", *n_clusters,
           "--n-workers", n_workers,
           "--artifact-path", artifact]
    if not export:
        cmd.append("--no-export")
    return [str(c) for c in cmd]


def uq_build(model: ModelRef, train_data: Sequence[Path], artifact: Path, rp_dim: int,
             n_clusters: Sequence[int], export: bool = True, n_workers: int = 1,
             log_file: Path | None = None) -> Path:
    run(uq_build_command(model, train_data, artifact, rp_dim, n_clusters, export, n_workers),
        log_file=log_file)
    return artifact


def uq_info_command(artifact: Path) -> list[str]:
    return ["grace_uq", "info", str(artifact)]


def uq_info(artifact: Path) -> tuple[str, dict[str, dict]]:
    """`grace_uq info` text and its per-element table: {el: {K_eff, n_atoms, min_rank}}."""
    out = subprocess.run(uq_info_command(artifact), capture_output=True, text=True, check=True)
    return out.stdout, parse_uq_info(out.stdout)


def parse_uq_info(text: str) -> dict[str, dict]:
    """Rows `  Pd(0)  K_eff  n_atoms  [eff_count]  a..b  max_cond  r/D ...` of the
    per-element table (min_rank as an int; None where the table shows '-')."""
    out = {}
    for line in text.splitlines():
        tok = line.split()
        if len(tok) < 6 or not (tok[1].isdigit() and tok[2].isdigit()):
            continue
        el = re.sub(r"\(\d+\)$", "", tok[0])
        ranks = [t for t in tok[3:] if re.fullmatch(r"\d+/\d+", t)]
        out[el] = {"K_eff": int(tok[1]), "n_atoms": int(tok[2]),
                   "min_rank": int(ranks[0].split("/")[0]) if ranks else None}
    return out


def hal_cli_command(model: ModelRef, structure: Path, index: int, outdir: Path, *,
                    ensemble: str, temperature: float, tau: float, n_steps: int, seed: int,
                    walker_id: int, progress_every: int = 0, overwrite: bool = False,
                    extra: Sequence[Any] = ()) -> list[str]:
    """One HAL walker started from frame `index` of `structure`."""
    if model.artifact is None:
        raise ValueError("HAL needs a model with a UQ artifact")
    cmd = ["grace_uq", "hal",
           "--model-yaml", model.model_yaml, "--checkpoint", model.checkpoint,
           "--artifact-path", model.artifact,
           "--structure", structure, "--structure-index", index,
           "--ensemble", ensemble, "-T", temperature, "--tau", tau,
           "--n-steps", n_steps, "--seed", seed, "--walker-id", walker_id,
           "--output-dir", outdir, "--progress-every", progress_every]
    if overwrite:
        cmd.append("--overwrite")
    return [str(c) for c in [*cmd, *extra]]


def hal_run_inproc(atoms, model: ModelRef, outdir: Path, **kw) -> dict:
    """run_hal in this process (loads TF); returns the summary dict
    (stop_reason, n_collected, rejected, ...)."""
    from tensorpotential.uq.hal import run_hal

    if model.artifact is None:
        raise ValueError("HAL needs a model with a UQ artifact")
    kw.setdefault("overwrite", True)
    res = run_hal(atoms, model_yaml=str(model.model_yaml), checkpoint=str(model.checkpoint),
                  uq_artifact=str(model.artifact), output_dir=str(outdir), **kw)
    return res.summary


def uq_select_command(walker_dirs: Sequence[Path], artifact: Path, strategy: Sequence[str],
                      output: Path) -> list[str]:
    return [str(c) for c in ["grace_uq", "select", "--hal-dirs", *walker_dirs,
                             "--artifact", artifact, "--strategy", *strategy,
                             "--output", output]]


def uq_select(walker_dirs: Sequence[Path], artifact: Path, strategy: Sequence[str],
              output: Path, log_file: Path | None = None) -> None:
    run(uq_select_command(walker_dirs, artifact, strategy, output), log_file=log_file)


def parse_select_log(text: str) -> dict:
    """From `grace_uq select` stdout: whether calm-greedy stopped at its natural set
    (False: the -n cap was reached first) and the natural set size if printed."""
    m = re.search(r"natural set size:? (\d+)", text)
    capped = "reached before the natural set" in text
    return {"natural": not capped and m is not None,
            "natural_set": int(m.group(1)) if m else None}


def predict_command(model: ModelRef, dataset: Path, output: Path) -> list[str]:
    """E/F/S (+ gamma) of every row of `dataset` -> `output` (.pkl.gz: energy_predicted,
    forces_predicted, stress_predicted, gamma; rows in input order)."""
    if model.artifact is None:
        raise ValueError("grace_uq predict with a model.yaml needs a UQ artifact")
    return [str(c) for c in ["grace_uq", "predict", "--model", model.model_yaml,
                             "--checkpoint", model.checkpoint, "--artifact", model.artifact,
                             "--dataset", dataset, "--output", output, "--n-workers", 1,
                             "--no-ase-atoms"]]


def predict(model: ModelRef, dataset: Path, output: Path, log_file: Path | None = None):
    """grace_uq predict in a subprocess (TF stays out of this process); the DataFrame."""
    import pandas as pd

    run(predict_command(model, dataset, output), log_file=log_file)
    return pd.read_pickle(output)


# ---------------------------------------------------------------- gracemaker
def gracemaker_command(restart_latest: bool = False) -> list[str]:
    return ["gracemaker", "input.yaml"] + (["-rl"] if restart_latest else [])


def gracemaker(fit_dir: Path, restart_latest: bool = False) -> None:
    """Run gracemaker on fit_dir/input.yaml (-rl: continue from the latest checkpoint).
    Its log goes to fit_dir/log.txt (gracemaker's own -l default)."""
    run(gracemaker_command(restart_latest), cwd=fit_dir, log_file=fit_dir / "gracemaker.out")


def fit_reference_energies(fit_dir: Path) -> dict[str, float] | None:
    """Per-element E0 that gracemaker computed for `reference_energy: auto` (from its
    log.txt in seed/1/; the last occurrence, i.e. that of the latest restart), or None."""
    logs = [p for p in (fit_dir / "seed" / "1" / "log.txt", fit_dir / "log.txt") if p.exists()]
    if not logs:
        return None
    found = re.findall(r"computed single-atom energies \(eV\): (\{[^}]*\})",
                       logs[0].read_text(errors="ignore"))
    return {k: float(v) for k, v in ast.literal_eval(found[-1]).items()} if found else None


def n_layers(fm_name: str) -> str:
    return "3L" if "3L" in fm_name else "2L" if "2L" in fm_name else "1L"


def finetune_trainable_variables(fm_name: str) -> list[str]:
    """Weights trained in a frozen fine-tune of `fm_name` (by its number of layers)."""
    return list(TRAINABLE_REDUCING[n_layers(fm_name)])

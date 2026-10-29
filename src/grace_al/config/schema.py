"""Configuration schema (pydantic v2). Defaults = the values of the original config.yaml.

Every section forbids unknown keys, so a typo (`aims.paramters`) is an error. Paths are
made absolute against `home` (the folder of config.yaml) once, at load time.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grace_al import _tp
from grace_al.utils import sha256_json

SCHEMA_VERSION = 1

FACETS = ("fcc100", "fcc110", "fcc111", "fcc211", "bcc100", "bcc110", "bcc111",
          "hcp0001", "hcp10m10", "diamond100", "diamond111")
Ensemble = Literal["nvt", "nvt_ramp", "npt", "npt_stress", "cluster"]
BUILTIN_LABELERS = ("aims", "teacher", "manual")

DEFAULT_AIMS_PARAMETERS: dict[str, Any] = {
    "xc": "pbe",
    "relativistic": "atomic_zora scalar",
    "occupation_type": "methfessel-paxton 0.1 1",
    "spin": "none",
    "use_dipole_correction": True,   # slabs only (the labeler drops it otherwise)
    "sc_accuracy_rho": 1.0e-6,
    "sc_iter_limit": 200,
}


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- system
class SurfaceConfig(Section):
    element: str
    facet: Literal[FACETS] = "fcc111"  # type: ignore[valid-type]
    size: list[int] = Field(default=[3, 3, 5], min_length=3, max_length=3)
    vacuum: float = 15.0              # Å on each side
    a: float | None = None            # lattice constant; None = ASE reference value
    c: float | None = None
    orthogonal: bool | None = None
    fix_layers: int = 2               # bottom layers fixed in the FM pre-relaxation


class AdsorbateConfig(Section):
    species: str                      # element or ase.build.molecule name
    sites: list[str] | None = None    # None = all sites of the facet
    height: float = 1.5
    min_distance: float = 2.5
    bind_index: int | None = None     # default: least electronegative heavy atom


class InitialConfig(Section):
    strains: list[float] = [0.98, 1.0, 1.02]
    coverages: list[int] = [2, 3]
    n_per_coverage: int = 2
    fm_relax: bool = True
    fmax: float = 0.05
    relax_steps: int = 100
    n_rattle: int = 1
    rattle_std: float = 0.05
    seed: int = 0
    mix: bool = False                 # several adsorbate species: coverages mix them


class BulkConfig(Section):
    element: str                      # or a compound for ase.build.bulk (e.g. NaCl, rocksalt)
    crystalstructure: str | None = None   # fcc, bcc, hcp, diamond, rocksalt, ...; None: ASE's
    a: float | None = None
    c: float | None = None
    cubic: bool = True
    size: list[int] = Field(default=[2, 2, 2], min_length=3, max_length=3)
    vacancies: list[int] = [0, 1]     # vacancies per supercell (random sites)
    substitutions: dict[str, float] = {}   # element -> fraction of host sites (alloys)


class SystemConfig(Section):
    generator: str = "surface"        # entry point group grace_al.systems
    surface: SurfaceConfig | None = None
    bulk: BulkConfig | None = None
    adsorbate: AdsorbateConfig | list[AdsorbateConfig] | None = None
    initial: InitialConfig = Field(default_factory=InitialConfig)
    path: Path | None = None          # generator: file

    @model_validator(mode="after")
    def _generator_inputs(self):
        if self.generator == "surface" and self.surface is None:
            raise ValueError("generator 'surface' needs a `surface:` section")
        if self.generator == "bulk" and self.bulk is None:
            raise ValueError("generator 'bulk' needs a `bulk:` section")
        if self.generator == "file" and self.path is None:
            raise ValueError("generator 'file' needs `path:`")
        return self


# ---------------------------------------------------------------- stages
class ArtifactConfig(Section):
    rp_dim_round0: int = 32           # lowered automatically when an element has few atoms
    rp_dim: int = 128
    n_clusters: list[int] = [1, 2, 4, 8, 16]
    n_workers: int = 1


class ExploreConfig(Section):
    ensembles: list[Ensemble] = Field(default=["nvt"], min_length=1)
    temperatures: list[int | float] = Field(default=[800, 1200], min_length=1)
    tau: list[float] = Field(default=[0.25], min_length=1)
    n_steps: int = 2000
    walkers_per_setting: int = 4      # walkers per (ensemble, T, tau)
    backend: Literal["inproc", "pool"] = "pool"
    walkers_per_gpu: int = 8
    run_hal_kwargs: dict[str, Any] = {}
    # max fraction of walkers starting from frames whose info["config"] starts with the key
    start_weights: dict[str, float] = {"clean": 0.2}
    allow_cell_change_on_slab: bool = False   # nvt_ramp/npt rescale the vacuum too
    # tau calibration (round 0): short walkers over calibrate_taus, keep the largest tau
    # whose rejected/(collected+rejected) <= max_reject_ratio and that collects something
    calibrate: bool = False
    calibrate_taus: list[float] = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5]
    calibrate_steps: int = 500
    calibrate_starts: int = 4
    max_reject_ratio: float = 0.2
    recalibrate_each_round: bool = False
    est_steps_per_s: float | None = None     # preflight time estimate (default: 25 on CPU)


class SelectConfig(Section):
    max_per_round: int = 100          # cap on calm-greedy's natural set
    n_random: int = 5


class ValidationConfig(Section):
    include_round0: bool = True       # round 0 too: the foundation model vs the first DFT data
    parity: bool = True               # validate/parity.png + parity.npz every round


class LabelConfig(Section):
    labeler: str = "aims"             # aims | teacher | manual | module:function
    teacher_model: str | None = None  # labeler: teacher
    mode: Literal["inline", "concurrent", "array"] = "inline"
    concurrent_calcs: int = 1
    max_failure_fraction: float = 0.2

    @model_validator(mode="before")
    @classmethod
    def _fm_shortcut(cls, data):
        # `labeler: fm:<MODEL>` (original scripts) = teacher labeler
        if isinstance(data, dict) and str(data.get("labeler", "")).startswith("fm:"):
            data = {**data, "labeler": "teacher", "teacher_model": data["labeler"][3:]}
        return data

    @model_validator(mode="after")
    def _known(self):
        if self.labeler not in BUILTIN_LABELERS and ":" not in self.labeler:
            raise ValueError(f"labeler {self.labeler!r}: use one of {BUILTIN_LABELERS} "
                             "or module:function")
        if self.labeler == "teacher" and not self.teacher_model:
            raise ValueError("labeler 'teacher' needs `teacher_model`")
        return self


class AimsConfig(Section):
    command: Path | None = None       # legacy launcher script (run_aims.sh)
    bin: str | None = None            # aims.x; default: $AIMS_BIN
    np: int = 8                       # MPI ranks when AIMS_NP / SLURM_NTASKS are not set
    species_dir: Path | None = None
    kspacing: float = Field(default=0.2, gt=0)   # 1/Å incl. 2π (VASP KSPACING)
    slab_min_vacuum: float = 6.0      # Å; larger gap along c -> slab (k_z = 1, dipole)
    timeout: float | None = None      # s per calculation
    parameters: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_AIMS_PARAMETERS))

    @field_validator("parameters", mode="before")
    @classmethod
    def _defaults(cls, v):
        # on top of the defaults; `key: null` removes a default
        merged = {**DEFAULT_AIMS_PARAMETERS, **(v or {})}
        return {k: val for k, val in merged.items() if val is not None}

    @field_validator("bin")
    @classmethod
    def _expand(cls, v):
        return os.path.expanduser(os.path.expandvars(v)) if v else v


class FinetuneConfig(Section):
    frozen: bool = True               # train only the reducing_ weights
    reference_energy: Literal["auto"] | float | dict[str, float] = "auto"
    learning_rate: float = 1.0e-3
    target_total_updates: int = 10000
    batch_size: int = 16
    test_size: float = 0.1
    energy_weight: float = 16
    force_weight: float = 32
    stress_weight: float = 128.0
    start_from: Literal["fm", "previous"] = "fm"
    extra: dict[str, Any] = {}        # deep-merged into input.yaml


class ConvergenceConfig(Section):
    max_natural_set: int = 0          # stop when calm-greedy finds at most this many frames
    force_rmse: float | None = None   # eV/Å on the random picks
    max_rounds: int = 5


class HpcConfig(Section):
    scheduler: Literal["slurm"] = "slurm"
    partition_gpu: str | None = None
    partition_cpu: str | None = None
    account: str | None = None
    gres: str = "gpu:1"
    ntasks: int = 32
    cpus_per_task: int = 1
    mem: str = "128G"
    time: str = "48:00:00"
    label_time: str = "08:00:00"      # per aims array task
    array_max_parallel: int = 20
    max_resubmit: int = 10
    modules: list[str] = []
    env: dict[str, str] = {}          # exported for aims (site quirks)
    activate: str | None = None       # shell line that activates the Python env
    mpirun: str | None = None
    sbatch_extra: list[str] = []

    @field_validator("env", mode="before")
    @classmethod
    def _str_values(cls, v):
        return {k: str(val) for k, val in (v or {}).items()}


# ---------------------------------------------------------------- root
class Config(Section):
    schema_version: int = SCHEMA_VERSION
    site: str | None = None
    home: Path = Field(default_factory=Path.cwd, exclude=True)   # folder of config.yaml
    workdir: Path = Path("runs")
    foundation_model: str = "GRACE-2L-SMAX-OMAT-medium"
    seeds: Path = Path("seeds.extxyz")
    fm_rounds: int = 1
    system: SystemConfig | None = None
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)
    explore: ExploreConfig = Field(default_factory=ExploreConfig)
    select: SelectConfig = Field(default_factory=SelectConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    aims: AimsConfig | None = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    hpc: HpcConfig | None = None

    @field_validator("foundation_model")
    @classmethod
    def _full_name(cls, v):
        if v in _tp.FM_ALIASES:
            raise ValueError(f"{v!r} is an alias; use the full name {_tp.FM_ALIASES[v]!r}")
        return v

    @model_validator(mode="after")
    def _finish(self):
        self.home = Path(self.home).expanduser().resolve()
        # $VARS and ~ expanded, then relative to the config's folder
        resolve = lambda p: p if p is None else (
            self.home / Path(os.path.expandvars(str(p))).expanduser())
        self.workdir = resolve(self.workdir)
        self.seeds = resolve(self.seeds)
        if self.system is not None:
            self.system.path = resolve(self.system.path)
        if self.aims is not None:
            self.aims.species_dir = resolve(self.aims.species_dir)
            self.aims.command = resolve(self.aims.command)
        if self.label.labeler == "aims":
            if self.aims is None:
                raise ValueError("label.labeler is 'aims' but there is no `aims:` section")
            if self.aims.species_dir is None:
                raise ValueError("label.labeler is 'aims': set aims.species_dir")
        if self.label.mode == "array" and self.hpc is None:
            raise ValueError("label.mode 'array' needs an `hpc:` section (or a site profile)")
        return self

    # ---- serialization ----
    def to_dict(self) -> dict:
        """Plain dict with paths relative to `home` where possible (written back as YAML,
        hashed for fingerprints)."""
        return _relativize(self.model_dump(mode="json"), str(self.home))

    def section_fingerprint(self, *sections: str) -> str:
        """sha256 of these sections: stable under key order and under moving `home`."""
        d = self.to_dict()
        for s in sections:
            if s not in type(self).model_fields:
                raise KeyError(f"no config section {s!r}")
        return sha256_json({s: d.get(s) for s in sections})


def _relativize(obj: Any, home: str) -> Any:
    if isinstance(obj, dict):
        return {k: _relativize(v, home) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v, home) for v in obj]
    if isinstance(obj, str) and obj.startswith(home + os.sep):
        return obj[len(home) + 1:]
    return obj


# ---------------------------------------------------------------- legacy configs
def migrate_legacy(d: dict) -> tuple[dict, list[str]]:
    """Config of the original al_workflow.py (no schema_version) -> schema 1.
    Returns the new dict and a note per changed key."""
    d = copy.deepcopy(d)
    notes = []
    if "hal" in d:
        hal = d.pop("hal") or {}
        # the old default was the in-process run_hal loop
        hal["backend"] = "pool" if hal.pop("parallel_cli", False) else "inproc"
        if "repeats" in hal:
            hal["walkers_per_setting"] = hal.pop("repeats")
        d["explore"] = hal
        notes.append("hal -> explore (parallel_cli -> backend, repeats -> walkers_per_setting)")
    spec = d.pop("labeler", None)          # the old default: none = compute DFT yourself
    if spec is None:
        d["label"] = {"labeler": "manual"}
    elif str(spec).startswith("fm:"):
        d["label"] = {"labeler": "teacher", "teacher_model": spec[3:]}
    else:
        d["label"] = {"labeler": spec}
    notes.append(f"labeler: {spec} -> label.labeler: {d['label']['labeler']}")
    if "max_rounds" in d:
        d.setdefault("convergence", {})["max_rounds"] = d.pop("max_rounds")
        notes.append("max_rounds -> convergence.max_rounds")
    d["schema_version"] = SCHEMA_VERSION
    return d, notes

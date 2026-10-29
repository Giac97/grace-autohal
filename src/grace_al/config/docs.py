"""Config reference (docs/config.md) generated from the schema: one table per section with
every key, its type, default and the comment next to it in schema.py.

    python -m grace_al.config.docs > docs/config.md
"""

from __future__ import annotations

import inspect
import re
import typing

from pydantic import BaseModel

from grace_al.config import schema

INTRO = """# Configuration reference

Generated from `grace_al/config/schema.py` (`python -m grace_al.config.docs`); do not edit.

Paths are relative to the folder of `config.yaml`. Unknown keys are errors. Layers, later wins:
package defaults → site profile (`site:`) → `config.yaml` → environment variables
`GRACE_AL__SECTION__KEY=value` (e.g. `GRACE_AL__AIMS__NP=32`).
"""

SECTIONS: list[tuple[str, type[BaseModel]]] = [  # (key in config.yaml, model)
    ("(top level)", schema.Config),
    ("system", schema.SystemConfig),
    ("system.surface", schema.SurfaceConfig),
    ("system.bulk", schema.BulkConfig),
    ("system.adsorbate", schema.AdsorbateConfig),
    ("system.initial", schema.InitialConfig),
    ("artifact", schema.ArtifactConfig),
    ("explore", schema.ExploreConfig),
    ("select", schema.SelectConfig),
    ("label", schema.LabelConfig),
    ("aims", schema.AimsConfig),
    ("validation", schema.ValidationConfig),
    ("finetune", schema.FinetuneConfig),
    ("convergence", schema.ConvergenceConfig),
    ("hpc", schema.HpcConfig),
]

# keys without an inline comment in schema.py ("section.key"; top level: "key")
DESCRIPTIONS = {
    "schema_version": "config format version; configs without it are read as al_workflow.py "
                      "configs and migrated",
    "site": "site profile (bwunicluster3, workstation, or a YAML path) merged under this file",
    "workdir": "campaign output folder (state.json, round_XX/)",
    "foundation_model": "GRACE foundation model (full name): drives the first rounds, is "
                        "fine-tuned every round",
    "seeds": "initial structures; built from `system:` if missing",
    "fm_rounds": "rounds driven by the foundation model; later rounds use the last fine-tuned "
                 "model",
    "system": "what to build the seeds from (omit to use your own `seeds` file)",
    "artifact": "CALM artifact (uncertainty model) of each round",
    "explore": "HAL exploration",
    "select": "structure selection (grace_uq select)",
    "label": "labeling (DFT)",
    "aims": "FHI-aims settings; keep them fixed for a campaign (refused if changed)",
    "finetune": "gracemaker fine-tuning",
    "validation": "the driver's errors on each round's new DFT data (validate stage)",
    "convergence": "stop criteria",
    "hpc": "batch jobs (usually from the site profile)",
    "system.surface": "slab (generator surface)",
    "system.bulk": "crystal (generator bulk)",
    "system.adsorbate": "one adsorbate, or a list of them",
    "system.initial": "which configurations to build, relaxation, rattling",
    "system.surface.element": "element of the slab",
    "system.surface.size": "lateral repeats x, y and number of layers",
    "system.surface.c": "c lattice constant (hcp); null = ASE reference value",
    "system.surface.orthogonal": "orthogonal cell (ASE builder option); null = its default",
    "system.bulk.a": "lattice constant in Å; null = ASE reference value",
    "system.bulk.c": "c lattice constant (hcp); null = ASE reference value",
    "system.bulk.cubic": "cubic cell instead of the primitive one",
    "system.bulk.size": "supercell repeats",
    "system.adsorbate.height": "Å above the top layer, before the pre-relaxation",
    "system.adsorbate.min_distance": "Å between adsorbates in coverage configs (lateral, "
                                     "minimum image)",
    "system.initial.strains": "lateral (surface) or isotropic (bulk) strain factors",
    "system.initial.coverages": "numbers of adsorbates of the random coverage configs",
    "system.initial.n_per_coverage": "random configs per coverage (bulk: per vacancy count "
                                     "and strain)",
    "system.initial.fm_relax": "pre-relax every config with the foundation model",
    "system.initial.fmax": "eV/Å, pre-relaxation force threshold",
    "system.initial.relax_steps": "maximum BFGS steps of the pre-relaxation",
    "system.initial.n_rattle": "rattled copies of every config",
    "system.initial.rattle_std": "Å, standard deviation of the rattle displacements",
    "system.initial.seed": "random seed (coverage sites, rattling)",
    "artifact.rp_dim": "feature dimension of later rounds' artifacts (capped by the atom count "
                       "of the scarcest element)",
    "artifact.n_clusters": "GMM cluster counts tried per element (round 0: [1])",
    "artifact.n_workers": "grace_uq build worker processes",
    "explore.ensembles": "HAL ensembles: nvt, nvt_ramp, npt, npt_stress, cluster (slabs: nvt)",
    "explore.temperatures": "K",
    "explore.tau": "bias strength(s): too small collects nothing, too large gets rejected",
    "explore.n_steps": "MD steps per walker",
    "explore.backend": "pool: walkers as subprocesses sharing the GPU; inproc: one by one in "
                       "the main process",
    "explore.walkers_per_gpu": "pool: walkers at a time per GPU (or on the CPU without one)",
    "explore.run_hal_kwargs": "extra run_hal options (pool: passed as --kebab-case flags)",
    "explore.calibrate_taus": "tau values tried by the calibration",
    "explore.calibrate_steps": "MD steps of a calibration walker",
    "explore.calibrate_starts": "calibration walkers per tau",
    "explore.max_reject_ratio": "calibration: highest acceptable rejected / (collected + "
                                "rejected)",
    "explore.recalibrate_each_round": "calibrate again with every fine-tuned driver",
    "select.n_random": "random picks per round (out-of-sample validation, convergence)",
    "label.mode": "inline: in the GPU job; concurrent: several at once; array: CPU job array",
    "label.concurrent_calcs": "concurrent mode: calculations side by side",
    "label.max_failure_fraction": "more failed labels than this stops the round",
    "aims.species_dir": "species defaults folder (light for tests, intermediate for production)",
    "aims.parameters": "control.in keywords on top of the defaults (pbe, atomic_zora, "
                       "methfessel-paxton 0.1, dipole correction for slabs, ...); "
                       "`key: null` removes a default",
    "finetune.reference_energy": "auto: per-element E0 by least squares (aims totals are "
                                 "all-electron); or a number / {element: E0}",
    "finetune.learning_rate": "start of the cosine schedule (minimum: lr / 12)",
    "finetune.target_total_updates": "optimizer steps of a fit",
    "finetune.batch_size": "structures per batch",
    "finetune.test_size": "fraction of the data held out as test set",
    "finetune.energy_weight": "loss weight",
    "finetune.force_weight": "loss weight",
    "finetune.stress_weight": "loss weight (only when every structure has stress)",
    "finetune.start_from": "fm: fine-tune the foundation model every round; previous: "
                           "continue from the last fine-tuned model",
    "convergence.max_rounds": "stop after this many rounds",
    "hpc.scheduler": "batch system",
    "hpc.partition_gpu": "partition of the GPU job (must be open to your group: grace-al "
                         "prepare)",
    "hpc.partition_cpu": "partition of the DFT job array",
    "hpc.account": "SLURM account (--account)",
    "hpc.gres": "GPU request, e.g. gpu:1 or gpu:h100:1",
    "hpc.ntasks": "tasks of the GPU job = aims ranks in inline mode",
    "hpc.cpus_per_task": "CPUs per task of the GPU job",
    "hpc.mem": "memory of the GPU job",
    "hpc.time": "time limit of the GPU job (resubmits itself 15 min before)",
    "hpc.array_max_parallel": "array tasks running at once",
    "hpc.max_resubmit": "resubmissions at most",
    "hpc.modules": "modules FHI-aims was built with (loaded before the Python env)",
    "hpc.mpirun": "MPI launcher of aims, may include options (srun --mpi=pmi2)",
    "hpc.sbatch_extra": "extra #SBATCH lines, e.g. --constraint=...",
}


def comments(model: type[BaseModel]) -> dict[str, str]:
    """The `# comment` at the end of each `name: type = default` line of the class body."""
    out: dict[str, str] = {}
    for line in inspect.getsource(model).splitlines():
        m = re.match(r"\s{4}(\w+)\s*:.*?#\s*(.*)$", line)
        if m and m.group(1) in model.model_fields:
            out[m.group(1)] = m.group(2).strip()
    return out


def above(model: type[BaseModel]) -> dict[str, str]:
    """Comment lines directly above a field."""
    out, buf = {}, []
    for line in inspect.getsource(model).splitlines():
        c = re.match(r"\s{4}#\s*(.*)", line)
        if c:
            buf.append(c.group(1).strip())
            continue
        m = re.match(r"\s{4}(\w+)\s*:", line)
        if m and buf:
            out[m.group(1)] = " ".join(buf)
        buf = []
    return out


def type_name(ann) -> str:
    s = str(ann).replace("typing.", "").replace("pathlib.", "")
    s = re.sub(r"<class '([\w.]+)'>", lambda m: m.group(1).split(".")[-1], s)
    s = re.sub(r"grace_al\.config\.schema\.(\w+)", r"\1", s)
    s = re.sub(r"Literal\[(.*?)\]", lambda m: " | ".join(m.group(1).replace("'", "").split(", ")),
               s)
    return s.replace("NoneType", "null").replace("|", r"\|")


def default(field) -> str:
    if field.is_required():
        return "required"
    v = field.get_default(call_default_factory=True)
    if isinstance(v, BaseModel):
        return "(section)"
    if isinstance(v, dict) and len(str(v)) > 60:
        return "(see schema.py)"
    return "`" + ("null" if v is None else str(v).replace("|", r"\|")) + "`"


def key_path(section: str, key: str) -> str:
    return key if section == "(top level)" else f"{section}.{key}"


def description(section: str, model: type[BaseModel], key: str) -> str:
    parts = (above(model).get(key, ""), comments(model).get(key, ""),
             DESCRIPTIONS.get(key_path(section, key), ""))
    return " ".join(x for x in parts if x).replace("|", "/")


def table(name: str, model: type[BaseModel]) -> str:
    rows = ["| key | type | default | meaning |", "|---|---|---|---|"]
    for key, f in model.model_fields.items():
        if f.exclude:
            continue
        ann = typing.get_type_hints(model).get(key, f.annotation)
        rows.append(f"| `{key}` | {type_name(ann)} | {default(f)} | "
                    f"{description(name, model, key)} |")
    doc = inspect.cleandoc(model.__dict__.get("__doc__") or "")
    head = f"## `{name}`\n\n" + (doc + "\n\n" if doc and model is not schema.Config else "")
    return head + "\n".join(rows) + "\n"


def render() -> str:
    return INTRO + "\n" + "\n".join(table(n, m) for n, m in SECTIONS)


if __name__ == "__main__":
    print(render(), end="")

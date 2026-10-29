"""Preflight checks (`grace-al check`): everything a run needs, before it starts.

Each check function takes the Config and yields Check results. Modules of other parts
register theirs with `@register_check` (preflight_aims, preflight_explore, ...).
"""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from grace_al import _tp
from grace_al.config.schema import Config


@dataclass
class Check:
    name: str
    ok: bool
    message: str = ""
    fix: str = ""
    severity: Literal["error", "warning"] = "error"
    group: str = "general"

    @property
    def failed(self) -> bool:
        return not self.ok and self.severity == "error"


CheckFn = Callable[[Config], Iterable[Check]]
_REGISTRY: list[CheckFn] = []
# modules that register checks when imported
EXTENSIONS = ("grace_al.preflight_explore", "grace_al.preflight_aims")


def register_check(fn: CheckFn) -> CheckFn:
    if fn not in _REGISTRY:
        _REGISTRY.append(fn)
    return fn


def run_checks(cfg: Config, gpu: bool = True,
               where: Literal["any", "login", "job"] = "any") -> list[Check]:
    """where: login (`prepare`: no GPU, maybe no aims binary -> warnings only), job (the
    GPU job: no visible GPU is an error), any (`check`)."""
    for mod in EXTENSIONS:
        importlib.import_module(mod)
    out: list[Check] = []
    for fn in (check_tools, check_models, check_seeds, *_REGISTRY):
        out.extend(fn(cfg))
    if gpu:
        out.extend(check_gpu(cfg))
    for c in out:
        if where == "login" and (c.group == "GPU" or c.name.startswith("aims binary")):
            c.severity = "warning"
        if where == "job" and c.group == "GPU" and c.name.endswith("visible"):
            c.severity = "error"
    return out


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=8)
def _elements_from_file(path: str, mtime: float) -> tuple[str, ...]:
    from ase.io import read

    return tuple(sorted({s for at in read(path, ":") for s in at.get_chemical_symbols()}))


def system_elements(cfg: Config) -> list[str]:
    """Elements of the campaign: from the seeds file, else from `system:`."""
    if cfg.seeds.exists():
        return list(_elements_from_file(str(cfg.seeds), cfg.seeds.stat().st_mtime))
    if cfg.system is None:
        return []
    from grace_al.systems.base import get_generator

    return list(get_generator(cfg.system.generator).info(cfg).elements)


# ---------------------------------------------------------------- checks
def check_tools(cfg: Config) -> Iterable[Check]:
    from importlib import metadata

    g = "python / tools"
    pd_version = metadata.version("pandas")
    yield Check(f"pandas {pd_version} < 3", int(pd_version.split(".")[0]) < 3,
                fix='mamba install "pandas<3" (pandas 3 breaks gracemaker batching)', group=g)
    v = _tp.tp_version()
    yield Check(f"tensorpotential {v} ({_tp.SUPPORTED})", _tp.version_supported(v),
                fix=f'pip install "tensorpotential{_tp.SUPPORTED}"', group=g)
    for tool in ("gracemaker", "grace_uq", "grace_models"):
        yield Check(f"{tool} on PATH", bool(shutil.which(tool)), fix="activate the grace env",
                    group=g)


def check_models(cfg: Config) -> Iterable[Check]:
    fm = cfg.foundation_model
    g = f"foundation model {fm}  (compute nodes often have no internet)"
    yield Check(f"checkpoint in {_tp.fm_checkpoint_dir(fm)}", _tp.checkpoint_cached(fm),
                fix=f"on a login node: grace_models checkpoint {fm}", group=g)
    if cfg.system is not None and cfg.system.initial.fm_relax and not cfg.seeds.exists():
        yield Check(f"SavedModel in {_tp.fm_savedmodel_dir(fm)} (seed pre-relaxation)",
                    _tp.savedmodel_cached(fm), fix=f"on a login node: grace_models download {fm}",
                    group=g)
    if cfg.label.labeler == "teacher":
        t = cfg.label.teacher_model
        assert t is not None                     # validated for labeler 'teacher'
        yield Check(f"teacher SavedModel in {_tp.fm_savedmodel_dir(t)}", _tp.savedmodel_cached(t),
                    fix=f"on a login node: grace_models download {t}", group=g)


def check_seeds(cfg: Config) -> Iterable[Check]:
    g = "seeds"
    if cfg.seeds.exists():
        from ase.io import read

        seeds = read(cfg.seeds, ":")
        yield Check(f"{cfg.seeds.name}: {len(seeds)} structures, periodic in 3D",
                    all(at.pbc.all() for at in seeds),
                    fix="set atoms.pbc = True (vacuum along z)", group=g)
    elif cfg.system is not None:
        elements = system_elements(cfg)
        yield Check(f"{cfg.seeds.name} will be built from `system:` ({', '.join(elements)})",
                    True, group=g)
    else:
        yield Check(f"{cfg.seeds}", False, fix="provide the seeds file or a `system:` section",
                    group=g)


def check_gpu(cfg: Config) -> Iterable[Check]:
    try:
        n = _tp.list_gpus()
    except Exception as err:
        yield Check("tensorflow import", False, str(err), group="GPU")
        return
    yield Check(f"{n} GPU(s) visible", n > 0,
                "fine on a login node; the job must see one", severity="warning", group="GPU")

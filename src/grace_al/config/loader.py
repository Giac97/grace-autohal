"""Load a config: site profile -> user YAML -> GRACE_AL__SECTION__KEY environment overrides.

Dicts merge, everything else (lists included) is replaced by the later layer. The schema's
defaults fill whatever no layer sets.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from grace_al.config.schema import Config, migrate_legacy
from grace_al.utils import deep_merge

log = logging.getLogger("grace_al.config")
ENV_PREFIX = "GRACE_AL__"


class ConfigError(ValueError):
    pass


def available_sites() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("grace_al.sites").iterdir()
                  if p.name.endswith(".yaml"))


def site_profile(name: str) -> dict:
    """A packaged site profile (grace_al/sites/<name>.yaml) or a path to a YAML file."""
    path = Path(name).expanduser()
    if path.suffix in (".yaml", ".yml") and path.exists():
        return yaml.safe_load(path.read_text()) or {}
    res = resources.files("grace_al.sites") / f"{name}.yaml"
    if not res.is_file():
        raise ConfigError(f"unknown site {name!r}; available: {', '.join(available_sites())}")
    return yaml.safe_load(res.read_text()) or {}


def env_overrides(environ: Mapping[str, str] | None = None) -> dict:
    """GRACE_AL__AIMS__NP=32 -> {"aims": {"np": 32}} (values parsed as YAML scalars).
    Keys are lowercased, except below an `env` key (GRACE_AL__HPC__ENV__I_MPI_DEBUG=5)."""
    out: dict = {}
    for key, val in (os.environ if environ is None else environ).items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p for p in key[len(ENV_PREFIX):].split("__") if p]
        path = [p if "env" in (q.lower() for q in path[:i]) else p.lower()
                for i, p in enumerate(path)]
        if not path:
            continue
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = yaml.safe_load(val) if val != "" else None
    return out


def read_user_config(path: Path) -> dict:
    """The user YAML, migrated to the current schema if it is a legacy (unversioned) one."""
    d = yaml.safe_load(path.read_text()) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    if "schema_version" not in d:
        d, notes = migrate_legacy(d)
        log.warning("%s has no schema_version: read as an al_workflow.py config (%s). "
                    "`grace-al init` writes the current format.", path.name, "; ".join(notes))
    return d


def merge_layers(user: dict, site: str | None = None,
                 environ: Mapping[str, str] | None = None) -> dict:
    site = site or user.get("site")
    data = site_profile(site) if site else {}
    data = deep_merge(data, user)
    if site:
        data["site"] = site
    return deep_merge(data, env_overrides(environ))


def format_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"] if not str(p).startswith("function-"))
        lines.append(f"  {loc or '<root>'}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: Path | str, site: str | None = None,
                environ: Mapping[str, str] | None = None) -> Config:
    """Validated Config; all paths absolute against the config file's folder."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file {path} not found")
    data = merge_layers(read_user_config(path), site, environ)
    data["home"] = path.parent
    try:
        return Config.model_validate(data)
    except ValidationError as err:
        raise ConfigError(f"invalid config {path}:\n{format_errors(err)}") from None


def dump_config(cfg: Config, path: Path | str) -> None:
    """Write cfg as YAML (paths relative to its home)."""
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
